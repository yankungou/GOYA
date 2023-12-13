# GOYA - Disentangling Content and Style in Art Paintings

## Introduction

This is the PyTorch code for GOYA, which is introduced in our ICMR 2023 paper: [Not Only Generative Art: Stable Diffusion for Content-Style Disentanglement in Art Analysis](https://arxiv.org/abs/2304.10278).

## Table of Contents

- [GOYA - Disentangling Content and Style in Art Paintings](#goya---disentangling-content-and-style-in-art-paintings)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Evaluation](#evaluation)
    - [Data preparation](#data-preparation)
    - [Distance correlation](#distance-correlation)
    - [Classification](#classification)
    - [Similarity retrieval](#similarity-retrieval)
  - [Training process](#training-process)
  - [Examples](#examples)
  - [Citation](#citation)

## Setup

To download the repository and install the conda environment, please run:

```
git clone https://github.com/yankungou/GOYA.git
cd GOYA
conda env create -f environment.yaml
conda activate GOYA
sh install_apex.sh
```

## Evaluation

### Data preparation

1. Download [WikiArt](https://drive.google.com/file/d/1h13azDWsjup0vxlxxOFcEUUrIpVuOSh3) dataset (about 26 GB), uncompress the file and place the image directory under `data/wikiart`.
2. Obtain content and style features of WikiArt images on GOYA:

   ```
   # Extract CLIP feature for WikiArt images
   python src/data_process/process_wikiart_img.py

   # Obtain content and style features of GOYA. CLIP features will be downloaded automatically if they do not exist.
   python src/eval/get_cs_feature.py --model GOYA
   ```

The following three experiments are computed based on the content and style features.

### Distance correlation

To compute [distance correlation](https://arxiv.org/abs/2008.12378) between content and style features:

```
python src/eval/DC/compute_DC.py --model GOYA
```

### Classification

To examine classification performance:

```
python src/eval/classifier/main.py --model GOYA --space <space>
```

where `space` is either `content` or `style`.

### Similarity retrieval

To visualize the similarity retrieval in content and style space, please run:

```
python src/eval/plot/fig4_retrieval.py --query <query>
```

where `query` refers to the query in Figure 4 from the paper.

## Training process

1. Image generation and process
   We apply [Stable Diffusion](https://github.com/CompVis/stable-diffusion) to generate images for training. You can run the following script to generate images using the same prompts as we use:

   ```
   python src/data_process/img_generation.py --data_type <data_type>
   ```

   where `data_type` is either `train` or `val`. For each prompt, we generate 5 images, resulting in the creation of 1,090,250 synthetic images (about 500GB) from 218,050 prompts. It may cost 57 days. As an alternative, you can download the generated images from [here](https://drive.google.com/drive/folders/1k_x6KoKhyqgNAYZeaGMChLP_kL8toCVR).

   After obtaining the generated images, we extract the pre-trained CLIP features for efficient training:

   ```
   python src/data_process/process_generated_img.py
   ```

   You can also download the features from [here](https://drive.google.com/drive/folders/1v2HTzGpo2L76D7ZJn9wlQDknYKTMFjEg), and place them in `data/generated/img_feature`.
2. Preparation

   To compute the distance matrix of content descriptions for the content contrastive loss:

   ```
   python src/data_process/content_preprocess.py
   ```
3. GOYA training

   We set 4 GPUs to train GOYA with Distributed Data Parallel (DDP):

   ```
   python -m torch.distributed.launch --nproc_per_node=4 --master_port=28898 --use_env src/model/main.py --world_size 4 --nepochs 100 --batch_size 512 --lr 0.0005 --model GOYA
   ```

   If training on a single GPU, it is recommended to set `batch_size` as the effective batch size:

   ```
   python src/model/main.py --nepochs 100 --batch_size 2048 --lr 0.0005 --model GOYA
   ```

## Examples

Retrieved paintings with highest similarity to the query image in the content and style spaces:

![1702393855812](image/README/1702393855812.png)

## Citation

If you find this code useful, please cite our work:

```
@InProceedings{wu2023not,
  author={Yankun Wu and Yuta Nakashima and Noa Garcia},
  title={Not Only Generative Art: Stable Diffusion for Content-Style Disentanglement in Art Analysis},
  booktitle={Proceedings of the 2023 ACM International Conference on Multimedia Retrieval},
  year={2023}
}
```
