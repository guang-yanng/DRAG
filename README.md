# DRAG


This is the official repository of the paper:

> **DRAG: Dynamic Region-Aware GCN for Privacy-Leaking Image Detection**
>
> Guang Yang, Juan Cao, Qiang Sheng, Peng Qi, Xirong Li, and Jintao Li
>
> *To be in the Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI 2022)*
>
> [Preprint](https://arxiv.org/pdf/2203.09121) / [Paper](https://www.aaai.org/AAAI22Papers/AISI-2085.YangG.pdf) / [Code](https://github.com/guang-yanng/DRAG) / [Short Video](https://aaai-2022.virtualchair.net/poster_aisi2085)

We provide the main codes as well as example notebooks to utilize the codes.

# Datasets

The experimental datasets were from our previous papers. Refer to the [paper](https://www.sciencedirect.com/science/article/pii/S0031320320301631) and [code](https://github.com/guang-yanng/Image_Privacy/tree/master/dataset) for details. We provide a sample dataloader for the Image Privacy dataset in the `dataset`.

# Code

## The Adopted Enviroment

```
python==3.6.8
torch==1.4.0
torchvision==0.5.0
```

## Steps

### Step 0: Modify the dataloader


### Step 1: Pretrain the Channel Grouping Layer
The example scripts are in the `channel_grouping_preprocess/ImagePrivacy/`. 
Refer the codes to:

#### 1.1: Pretrain a classification model

`01. classification_pretraining.ipynb`.

#### 1.2: Cluster the feature channels

 `02. channel_grouping.ipynb`.

#### 1.3: Pretrain the Channel Grouping Layer

 `03. channel_grouping_layer_pretraining.ipynb`.



### Step 2: Train the DRAG

Refer to the codes in `DRAG_ImagePrivacy.ipynb`.


# Citation

```
@article{yang2022drag,
  title={DRAG: Dynamic Region-Aware GCN for Privacy-Leaking Image Detection},
  author={Yang, Guang and Cao, Juan and Sheng, Qiang and Qi, Peng and Li, Xirong and Li, Jintao},
  journal={arXiv preprint arXiv:2203.09121},
  year={2022}
}
```
