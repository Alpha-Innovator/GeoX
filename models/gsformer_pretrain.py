import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from models.gsformer import init_tokenizer, GSFormer
from models.vit_encoder import create_vencoder
from lavis.common.registry import registry
from lavis.modules.base_model import all_gather_with_grad, concat_all_gather
from lavis.modules.blip2 import (
    Blip2Base,
    disabled_train,
)
from lavis.modules.blip_outputs import BlipOutput, BlipOutputFeatures


class GSFormerPretrain(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/model.yaml",
    }

    def __init__(
        self,
        vit_model="base",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        path_vit=None,
        num_query_token=8,
        cross_attention_freq=2,
        feature_dim=768,
        embed_dim=256,
        max_txt_len=64,
        loss_weight=1e-2
    ):
        super().__init__()
        self.tokenizer = init_tokenizer()
        self.gsformer = GSFormer(num_query_token, feature_dim, cross_attention_freq, finetune=False)

        self.gsformer.mmtransformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.gsformer.mmtransformer.state_dict()
        for name, param in self.gsformer.mmtransformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.gsformer.mmtransformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.gsformer.mmtransformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.gsformer.mmtransformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len

        self.visual_encoder = create_vencoder(
            path_vit, img_size, vit_precision
        )
        
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        else:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = True
            self.visual_encoder.train()
            logging.info("train vision encoder")
        self.loss_weight = loss_weight


    def forward(self, samples):
        image = samples["image"]
        image_embeds = self.visual_encoder(image).contiguous() # image_embeds: bs * n_patch * dim

        image_atts, query_tokens, query_output = self.gsformer(image_embeds)
        text = samples["text_input"]        
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        ).contiguous()


        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        text_output = self.gsformer.mmtransformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        ).contiguous()


        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        lm_output = self.gsformer.mmtransformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        
        
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        if "image_id" in samples.keys(): #coco retrieval finetuning
            # image_ids = samples["image_id"].view(-1,1)
            image_ids = torch.tensor(samples["image_id"]).to(device=targets.device).view(-1, 1)
            image_ids_all = concat_all_gather(image_ids)
            pos_idx = torch.eq(image_ids, image_ids_all.t()).float()       
            sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
            loss_itc = (loss_t2i+loss_i2t)/2  
        else:                     
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2

        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        image_atts_world = all_gather_with_grad(image_atts)  # Also gather attention masks for image patches
        query_tokens_world = all_gather_with_grad(query_tokens)


        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:    
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        image_atts_neg = []

        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
            image_atts_neg.append(image_atts_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)
        

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        query_tokens_neg = []

        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
            query_tokens_neg.append(query_tokens_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        query_tokens_neg = torch.stack(query_tokens_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_tokens_itm = torch.cat(
            [query_tokens, query_tokens, query_tokens_neg], dim=0
        )  # pos, pos, neg
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        # image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        image_atts_all = torch.cat(
            [image_atts, image_atts_neg, image_atts], dim=0
        )

        output_itm = self.gsformer.mmtransformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)
        loss_align = loss_itc + loss_lm + loss_itm

        pred_attention_masks = query_output.pred_attention_mask

        loss_spr = torch.mean(torch.stack([torch.mean(torch.abs(mask)) for mask in pred_attention_masks]))

        return BlipOutput(
            loss=loss_align + self.loss_weight * loss_spr
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        path_vit = cfg.get("path_vit")
        loss_weight = cfg.get("loss_weight")


        max_txt_len = cfg.get("max_txt_len", 64)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            path_vit=path_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            loss_weight=loss_weight
        )
        model.load_checkpoint_from_config(cfg)

        return model
