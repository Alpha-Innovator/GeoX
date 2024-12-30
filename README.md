

<div align= "center">
    <h1> Official repo for GeoX</h1>

</div>

<div align="center">
    <h2> <a href="https://arxiv.org/abs/2412.11863">GeoX: Geometric Problem Solving Through Unified Formalized Vision-Language Pre-training</a></h2>

  <p align="center">
    <a href="https://arxiv.org/abs/2412.11863">📃Arxiv Paper</a> •
    <a href="https://huggingface.co/datasets/U4R/GeoX-data">🎒Data</a> •
    <a href="https://huggingface.co/U4R/GeoX">🤗Checkpoint</a> •
    <a href="#-citation">📖Citation
  </p>
  <br>
  <img width="95%" src=./assets/teaser.png>
</div>


## 🏃 Intro GeoX


**GeoX** is a multi-modal large model designed for automatic geometric problem solving, incorporating three progressive training stages to enhance diagram understanding and reasoning. In this paper, we validate that the **formal vision-language training** is a simple-yet-effective paradigm for complex mathematical diagram learning.


<details open="open">
    <summary><b>Abstract</b></summary>
    Despite their proficiency in general tasks, Multi-modal Large Language Models (MLLMs) struggle with automatic Geometry Problem Solving (GPS), which demands understanding diagrams, interpreting symbols, and performing complex reasoning. This limitation arises from their pre-training on natural images and texts, along with the lack of automated verification in the problem-solving process. Besides, current geometric specialists are limited by their task-specific designs, making them less effective for broader geometric problems. To this end, we present GeoX, a multi-modal large model focusing on geometric understanding and reasoning tasks. Given the significant differences between geometric diagram-symbol and natural image-text, we introduce unimodal pre-training to develop a diagram encoder and symbol decoder, enhancing the understanding of geometric images and corpora. Furthermore, we introduce geometry-language alignment, an effective pre-training paradigm that bridges the modality gap between unimodal geometric experts. We propose a Generator-And-Sampler Transformer (GS-Former) to generate discriminative queries and eliminate uninformative representations from unevenly distributed geometric signals. Finally, GeoX benefits from visual instruction tuning, empowering it to take geometric images and questions as input and generate verifiable solutions. Experiments show that GeoX outperforms both generalists and geometric specialists on publicly recognized benchmarks, such as GeoQA, UniGeo, Geometry3K, and PGPS9k. Our data and code will be released soon to accelerate future research on automatic GPS.

</details>



## 🚩 News

- [2024/10/17] Upload paper and init project. Release the data for GeoX. See [here](https://huggingface.co/U4R/GeoX).


## ⚡ Set up

<details>
  <summary><b>Environment Setup</b></summary>

**Step 1. Build Dependencies.** Our code is tested with Python 3.10.14. To run the codes, you should first install the following packages:

```{bash}
conda create -n geox python=3.10
conda activate geox
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn==2.5.9.post1 --no-build-isolation
```
</details>



<details>
  <summary><b>Data and Weights Preparation</b></summary>


**Step 1. Download and Prepare Data.**


1. Follow the instructions [here](https://huggingface.co/datasets/U4R/GeoX-data) and download full dataset for GeoX. 
2. To train the model, you are required to organize the files into the following folders:

```
./data/

  alignment/
    images/
    unified_formal_annotations.json

  geoqa/
    images/
    geoqa_train.json
    geoqa_test.json

  unigeo/
    images/
    unigeo_train.json
    unigeo_test.json

  geometry3k/
    images/
    geometry3k_train.json
    geometry3k_test.json

  pgps9k/
    images/
    pgps9k_train.json
    pgps9k_test.json
```



</details>



## 💻 Train your own model

### (Optional) Uni-modal Pretraining
```{bash}

# Define the base directory and output directory
BASE_DIR="/path/to/your/base/directory"  # Modify this path as necessary
OUTPUT_DIR="/path/to/your/output/directory"  # Modify this path as necessary

# Run the Python script with the specified configurations
python ${BASE_DIR}/pretrain/pretrain_encoder.py \
    --job_dir ${OUTPUT_DIR}/checkpoint/mae \
    --nodes 1 \
    --ngpus 8 \
    --accum_iter 16 \
    --batch_size 256 \
    --use_volta32 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path ${BASE_DIR}/data  # Ensure the data path is correctly parameterized
```


```{bash}

# Define base directory and output model directory
DATA_FILE="/path/to/your/training/data"  # Modify this path as necessary
OUTPUT_DIR="/path/to/your/output/directory"  # Modify this path as necessary
MODEL_DIR="/path/to/LLEMMA/directory"  # Modify this path as necessary
LOG_FILE="${OUTPUT_DIR}/train.log"

# Create output directory if it does not exist
if [ ! -d "${OUTPUT_DIR}" ]; then  
    mkdir -p "${OUTPUT_DIR}"
fi

# Optional: Set environment variable for NCCL
# export NCCL_P2P_DISABLE=1  # Uncomment this if necessary

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    main/train_llm.py \
    --config_name "${MODEL_DIR}/config.json" \
    --tokenizer_name "${MODEL_DIR}" \
    --model_name_or_path "${MODEL_DIR}" \
    --train_files "${DATA_FILE}" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --output_dir "${OUTPUT_DIR}" \
    --evaluation_strategy steps \
    --use_fast_tokenizer false \
    --max_eval_samples 0 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --warmup_ratio 0.1 \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 20000000 \
    --eval_steps 500000000 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --run_name llm_pretrain \
    --bf16 \
    --bf16_full_eval \
    --gradient_checkpointing \
    --deepspeed configs/models/zero3.json \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    | tee -a "${LOG_FILE}"
```

<details>
  <summary><b>Training</b></summary>
</details>




## 📖 Citation

If you find our work helps, please consider starring ⭐ us and citing:

```{bibtex}

```


## Acknowledgments

Thanks to [LLaVA](https://github.com/haotian-liu/LLaVA), [LAVIS](https://github.com/salesforce/LAVIS), [MAE](https://github.com/facebookresearch/mae), and [trasnformers](https://github.com/huggingface/transformers). We borrow some of their codes and checkpoints.



## License

This code is distributed under an [Apache-2.0 license](LICENSE). If there are any problems regarding our project, please open an issue.
