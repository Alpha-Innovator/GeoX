
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

WHITE_PIXEL = [(1-0.485)/0.229, (1-0.456)/0.224, (1-0.406)/0.225]

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, pretrained=False, norm_pix_loss=False, mask_strategy="mae", loss_mode="mae", mae_loss_weight=1, loss_weight=0.1, thd=0.95):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_strategy = mask_strategy
        print("mask_strategy", self.mask_strategy)
        self.loss_mode = loss_mode
        self.loss_weight = loss_weight
        self.mae_loss_weight = mae_loss_weight
        self.thd = thd
        print("thd", self.thd)
        if not pretrained:
            self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def patchify_(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2, *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2, 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, unvalid_patch=None):
        N, L, D = x.shape  # batch, length, dim
        if self.mask_strategy == "valid_patch_static" or self.mask_strategy == "mae":
            len_keep = torch.full((N,), int(L * (1 - mask_ratio)), device=x.device) # [N]
        elif self.mask_strategy == "valid_patch_dynamic":
            valid_patch_ratio = (L - unvalid_patch.sum(dim=1))/L  # Count valid patches for each sample
            len_keep = (L * (1 - valid_patch_ratio * mask_ratio)).to(torch.int64)
        else:
            raise ValueError("Mask strategy not supported")

        

        if unvalid_patch is None:
            noise = torch.rand(N, L, device=x.device)
        else:
            noise = torch.full((N, L), float('inf'), device=x.device)
            noise[unvalid_patch] = torch.rand(unvalid_patch.sum(), device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        arange_ids = torch.arange(L, device=x.device).expand(N, L)
        mask_keep = arange_ids < len_keep.unsqueeze(1)

        # ids_keep = ids_shuffle[mask_keep].view(N, -1)
        actual_len_keep = mask_keep.sum(dim=1)
        ids_keep = torch.zeros((N, max(actual_len_keep)), device=x.device, dtype=torch.long)
        for i in range(N):
            ids_keep[i, :actual_len_keep[i]] = ids_shuffle[i][mask_keep[i]]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # 生成二进制掩码
        mask = torch.ones([N, L], device=x.device)
        mask[mask_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)


        return x_masked, mask, ids_restore


    def valid_masking(self, imgs, x, mask_ratio):
        # 使用patchify_转换imgs为补丁[N, L, patch_size**2 * 3]
        imgs_patch = self.patchify_(imgs)  

        N, L, _, _ = imgs_patch.shape 

        # 定义全白阈值并调整形状以匹配补丁的维度
        thresholds = torch.tensor([self.thd * value for value in WHITE_PIXEL], device=imgs_patch.device).view(1, 1, 1, 3)

        # 判断每个补丁是否为非全白
        valid_patch = (imgs_patch.view(N, L, -1, 3) < thresholds).any(dim=-1).any(dim=-1)  # [N, L]
        
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio, ~valid_patch)
        return x_masked, mask, ids_restore
    

   


    def forward_encoder(self, imgs, mask_ratio):
        # embed patches
        x = self.patch_embed(imgs)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.mask_strategy == "mae":
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif self.mask_strategy == "valid_patch_static" or self.mask_strategy == "valid_patch_dynamic":
            x, mask, ids_restore = self.valid_masking(imgs, x, mask_ratio)
        else:
            raise ValueError("Mask strategy not supported")


        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore





    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss_mae(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_mae_np(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Compute MAE loss
        mae_loss = (pred - target) ** 2
        mae_loss = mae_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mae_loss = (mae_loss * mask).sum() / mask.sum()  # mean loss on removed patches

        target_patch = self.patchify_(imgs)  # [N, L, p*p, 3]
        pred_patch = self.patchify_(self.unpatchify(pred))  # [N, L, p*p, 3]

        # Thresholds for considering a pixel non-white are 90% of the WHITE_PIXEL values
        non_white_thresholds = [self.thd * value for value in WHITE_PIXEL]
        
        # Determine non-white pixels using the specific threshold for each channel
        target_non_white = (target_patch < torch.tensor(non_white_thresholds).to(target_patch.device)).any(dim=-1)
        pred_non_white = (pred_patch < torch.tensor(non_white_thresholds).to(pred_patch.device)).any(dim=-1)


        # Count non-white pixels
        target_count = target_non_white.sum(dim=-1)/target_patch.shape[2]*10  # [N, L]
        pred_count = pred_non_white.sum(dim=-1)/pred_patch.shape[2]*10  # [N, L]

        criterion = nn.SmoothL1Loss()
        np_loss = criterion(target_count, pred_count)

        # Combine the MAE loss and the np_loss
        np_loss = (np_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        total_loss = self.mae_loss_weight * mae_loss + self.loss_weight * np_loss
        
        return total_loss




    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        if self.loss_mode == "mae":
            loss = self.forward_loss_mae(imgs, pred, mask)
        elif self.loss_mode == "mae_occ":
            loss = self.forward_loss_mae_occ(imgs, pred, mask)
        elif self.loss_mode == "mae_np":
            loss = self.forward_loss_mae_np(imgs, pred, mask)
        else:
            raise ValueError("Loss mode not supported")
        return loss, pred, mask





def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

