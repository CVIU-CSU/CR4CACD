## Contextual Relationships for Abnormal Cervical Cell Detection

This is the code implementation of [Exploring Contextual Relationships for Cervical Abnormal Cell Detection](https://arxiv.org/abs/2207.04693). 
Our code is built on the basis of MMDetection.

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

The master branch works with **PyTorch 1.6** and [**MMDetection v2.6.0**](https://github.com/open-mmlab/mmdetection).

## Datasets

We collected a liquid-based cervical cytology images dataset, called Cervical Cell Detection (CCD) dataset. The CCD dataset consists of 40,000 pathological images, 30,000 training, 5,000 validation and 5,000 testing. All annotated instances belong to 10 categories i.e., negative for intraepithelial lesion for malignancy (NILM), atypical squamous cells-undetermined significance (ASCUS), atypical squamous cells-cannot exclude HSIL (ASCH), low-grade squamous intraepithelial lesion (LSIL), high-grade squamous intraepithelial lesion (HSIL), atypical glandular cells (AGC), vaginalis trichomoniasis (VAG), monilia (MON), dysbacteriosis (DYS) and endocervical cells (EC), etc. Annotated NILM cells are used to assist model training, but are not involved in mAP computation.

## Method

![method](demo/flowchart.jpg)
The implementation of the GRAM and RRAM is in [roi_attention_head.py](mmdet/models/roi_heads/bbox_heads/roi_attention_head.py).

## Conifgs

We set up 5 config files to realize GRAM and RRAM including different combination strategies. Refer to [configs/roi_annention](configs/roi_attention) for details.

## Main Results

### On CCD dataset

Train log of Cascade RRAM and GRAM is in [20220418_170256.log.json](https://drive.google.com/file/d/1EK41ipU_l5cxRWC0AY0rzh1TtVJKqMed/view?usp=sharing). The trained model is available on google driver [here](https://drive.google.com/file/d/1EK41ipU_l5cxRWC0AY0rzh1TtVJKqMed/view?usp=sharing). The trained model with multi-scale training is available on google driver [here](https://drive.google.com/file/d/1u6wWtWsSnieZorDqPrYPTUYuZUWrMh2D/view?usp=sharing).

Model | AP | AP@50 | AP@75 
--- |:---:|:---:|:---:
Faster R-CNN with FPN | 30.6 | 53.6 | 31.7
Double-Head Faster R-CNN (baseline) | 30.9 | 53.9 | 32.2
RRAM | 32.0 | 56.0 | 32.8 
GRAM | 31.9 | 56.2 | 33.1 
Cascade RRAM and GRAM | 32.4 | 56.6 | 33.5
Cascade RRAM and GRAM (multi-scale training) | 34.2 | 58.6 | 36.0 

Model | ASCUS | ASCH | LSIL | HSIL | AGC | VAG | MON | DYS | EC | AP
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Faster R-CNN with FPN | 29.9 | 22.2 | 30.9 | 33.2 | 42.4 | 28.8 | 19.1 | 50.3 | 18.7 | 30.6 
Double-Head Faster R-CNN (baseline) | 30.4 | 22.7 | 31.2 | 33.3 | 42.8 | 29.1 | 18.4 | 51.1 | 19.2 | 30.9
RRAM | 32.1 | 24.1 | 31.8 | 33.7 | 43.9 | 29.5 | 19.5 | 52.9 | 20.3 | 32.0
GRAM | 32.2 | 23.3 | 31.7 | 33.3 | 44.1 | 29.5 | 19.7 | 53.1 | 20.5 | 31.9 
Cascade RRAM and GRAM | 32.2 | 24.1 | 32.9 | 34.0 | 44.4 | 29.7 | 20.6 | 53.4 | 20.7 | 32.4
Cascade RRAM and GRAM (multi-scale training) | 35.2 | 25.3 | 34.5 | 35.6 | 46.2 | 29.8 | 22.0 | 56.0 | 22.8 | 34.2

### On ComparisonDetector dataset

Train log files are in [work_dirs_comparison](/work_dirs_comparison).

Model | AP | AP@50 | AP@75 
--- |:---:|:---:|:---:
Faster R-CNN with FPN | 20.3 | 46.0 | 15.5
Double-Head Faster R-CNN (baseline) | 23.3 | 49.6 | 19.2
Cascade RRAM and GRAM | 28.2 | 56.2 | 25.6
Cascade RRAM and GRAM (multi-scale training) | 29.1 | 56.2 | 27.1 


## Citation
```
@article{liang2023exploring,
  title={Exploring contextual relationships for cervical abnormal cell detection},
  author={Liang, Yixiong and Feng, Shuo and Liu, Qing and Kuang, Hulin and Liu, Jianfeng and Liao, Liyan and Du, Yun and Wang, Jianxin},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}
```
