<div align="center">
    <h2>
        RSBuilding: Towards General Remote Sensing Image Building Extraction and Change Detection with Foundation Model
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/overview.png" width="400"/>
</div>


[![GitHub stars](https://badgen.net/github/stars/Meize0729/opencd_debug)](https://github.com/KyanChen/opencd_debug)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2306.16269-b31b1b.svg)](https://arxiv.org/abs/2306.16269)

## Introduction

This repository is the code implementation of the paper [RSBuilding: Towards General Remote Sensing Image Building Extraction and Change Detection with Foundation Model](https://arxiv.org/abs/2306.16269), which is based on the [Open-cd](https://github.com/likyoo/open-cd) project.

The current branch has been tested under PyTorch 2.0.1 and CUDA 11.7, supports Python 3.7+, and is compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️, your support is our greatest motivation.

## Update Log

🌟 **2024.02.18** Release the source code and make the pre-trained weights publicly available.

🌟 **2024.02.18** Updated the paper content, see [Arxiv](https://arxiv.org/abs/2306.16269) for details.

## Table of Contents

- [Introduction](#Introduction)
- [Update Log](#Update-Log)
- [Table of Contents](#Table-of-Contents)
- [Installation](#Installation)
- [Dataset Preparation](#Dataset-Preparation)
- [Model Training](#Model-Training)
- [Model Testing](#Model-Testing)
- [Image Prediction](#Image-Prediction)
- [Common Problems](#Common-Problems)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)
- [License](#License)
- [Contact](#Contact)

## Installation

### Dependencies

- Linux
- Python 3.7+, recommended 3.9
- PyTorch 2.0 or higher, recommended 2.0
- CUDA 11.7 or higher, recommended 11.7
- MMCV 2.0 or higher, recommended 2.0

### Environment Installation

We recommend using Miniconda for installation. The following command will create a virtual environment named `rsbuilding` and install PyTorch and MMCV.

Note: If you have experience with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow these steps to prepare.

<details>

**Tips**: We recommend installing the version that has been practically tested and proven to work.

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `rsbuilding` and activate it.

```shell
conda create --name rsbuilding python=3.9
conda activate rsbuilding
```

**Step 2**: Install [PyTorch](https://pytorch.org/get-started/locally/), we recommend using conda to install the following version:

```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). We recommend using pip to install the following version:

```shell
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Step 4**: Install other dependencies.

```shell
pip install -r requirements.txt
```

</details>




<!-- 
<div align="center">
  <img src="resources/opencd-logo.png" width="600"/>
</div>

## Introduction
Open-CD is an open source change detection toolbox based on a series of open source general vision task tools.


## News
- 4/21/2023 - Open-CD v1.0.0 is released in 1.x branch, based on OpenMMLab 2.0 ! PyTorch 2.0 is also supported ! Enjoy it !
- 3/14/2023 - Open-CD is upgraded to v0.0.3. Semantic Change Detection (SCD) is supported !
- 11/17/2022 - Open-CD is upgraded to v0.0.2, requiring a higher version of the MMSegmentation dependency.
- 9/28/2022 - The code, pre-trained models and logs of [ChangerEx](https://github.com/likyoo/open-cd/tree/main/configs/changer) are available. :yum:
- 9/20/2022 - Our paper [Changer: Feature Interaction is What You Need for Change Detection](https://arxiv.org/abs/2209.08290) is available!
- 7/30/2022 - Open-CD is publicly available!

## Benchmark and model zoo

Supported toolboxes:

- [x] [OpenMMLab Toolkits](https://github.com/open-mmlab)
- [x] [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [ ] ...

Supported change detection model:
(_The code of some models are borrowed directly from their official repositories._)

- [x] [FC-EF (ICIP'2018)](configs/fcsn)
- [x] [FC-Siam-diff (ICIP'2018)](configs/fcsn)
- [x] [FC-Siam-conc (ICIP'2018)](configs/fcsn)
- [x] [STANet (RS'2020)](configs/stanet)
- [x] [IFN (ISPRS'2020)](configs/ifn)
- [x] [SNUNet (GRSL'2021)](configs/snunet)
- [x] [BiT (TGRS'2021)](configs/bit)
- [x] [ChangeFormer (IGARSS'22)](configs/changeformer)
- [x] [TinyCD (NCA'2023)](configs/tinycd)
- [x] [Changer (TGRS'2023)](configs/changer)
- [x] [HANet (JSTARS'2023)](configs/hanet)
- [x] [TinyCDv2 (Under Review)](configs/tinycd_v2)
- [ ] ...

Supported datasets: | [Descriptions](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
- [x] [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [x] [S2Looking](https://github.com/S2Looking/Dataset)
- [x] [SVCD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- [x] [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)
- [x] [CLCD](https://github.com/liumency/CropLand-CD)
- [x] [RSIPAC](https://engine.piesat.cn/ai/autolearning/index.html#/dataset/detail?key=8f6c7645-e60f-42ce-9af3-2c66e95cfa27)
- [x] [SECOND](http://www.captain-whu.com/PROJECT/)
- [x] [Landsat](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)
- [x] [BANDON](https://github.com/fitzpchao/BANDON)
- [ ] ...

## Usage

[Docs](https://github.com/open-mmlab/mmsegmentation/tree/master/docs)

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) in mmseg.

A Colab tutorial is also provided. You may directly run on [Colab](https://colab.research.google.com/drive/1puZY5R8fwlL6um6pHbgbM1NTYZUXdK2J?usp=sharing). (thanks to [@Agustin](https://github.com/AgustinNormand) for this demo) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1puZY5R8fwlL6um6pHbgbM1NTYZUXdK2J?usp=sharing)

#### simple usage
```
# Install OpenMMLab Toolkits as Python packages
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
```
```
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
```
train
```
python tools/train.py configs/changer/changer_ex_r18_512x512_40k_levircd.py --work-dir ./changer_r18_levir_workdir
```
infer
```
# get .png results
python tools/test.py configs/changer/changer_ex_r18_512x512_40k_levircd.py  changer_r18_levir_workdir/latest.pth --show-dir tmp_infer
# get metrics
python tools/test.py configs/changer/changer_ex_r18_512x512_40k_levircd.py  changer_r18_levir_workdir/latest.pth
```

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@ARTICLE{10129139,
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2023.3277496}}
```

## License

Open-CD is released under the Apache 2.0 license. -->
