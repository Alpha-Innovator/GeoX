

<div align= "center">
    <h1> Official repo for GeoX</h1>

</div>

<div align="center">
    <h2> <a href="https://arxiv.org/abs/2312.10763">GeoX: Geometric Problem Solving Through Unified Formalized Vision-Language Pre-training</a></h2>

  <p align="center">
    <a href="">💻Project Page</a> •
    <a href="">📃Arxiv Paper</a> •
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

- [2024/10/17] Upload paper and init project. Release the the data for GeoX. 详情请See [here](https://huggingface.co/U4R/GeoX).


## ⚡ Set up

<details>
  <summary><b>Environment Setup</b></summary>

**Step 1. Build Dependencies.** Our code is tested with CUDA 12.2 and Python 3.10.14. To run the codes, you should first install the following packages:

```{bash}

```

After that, 

```{bash}

```

```{bash}

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

<details>
  <summary><b>Training</b></summary>
</details>

<details>
  <summary><b>Evaluation</b></summary>
</details>


## 📖 Citation

If you find our work helps, please consider starring ⭐ us and citing:

```{bibtex}

```


## Acknowledgments

Thanks to [LLaVA](https://github.com/haotian-liu/LLaVA), [LAVIS](https://github.com/salesforce/LAVIS), [MAE](https://github.com/facebookresearch/mae), and [trasnformers](https://github.com/huggingface/transformers). We borrow some of their codes and checkpoints.



## License

This code is distributed under an [Apache-2.0 license](LICENSE). If there are any problems regarding our project, please open an issue.
