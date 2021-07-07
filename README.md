## End-to-end Malaria Diagnosis and 3D Cell Rendering with Deep Learning
<a href="https://orcid.org/0000-0002-6521-7898"><img height="15" src="https://github.com/rvignav/CT2Xray/blob/master/docs/orcid.png"></a>&nbsp;Vignav Ramesh

[arXiv](https://arxiv.org/abs/2105.08147) / [Full Paper (PDF)](https://arxiv.org/pdf/2105.08147.pdf) / [Papers With Code](https://paperswithcode.com/paper/covid-19-lung-lesion-segmentation-using-a) / [Website](https://topdocmedicine.wixsite.com/topdoc)

![Header](https://github.com/rvignav/E2EMD/blob/main/docs/sshot.png)

### Pre-trained models

| Training dataset | Train/test split | Data augmentation (y/n) | URL |
| --- | --- | --- | --- |
| X-rays Only | 60/40 | y | [Download](https://drive.google.com/file/d/1Db0NhVCIBOJJTfDHjtmgm3I10-KsUpg-/view?usp=sharing) |
| Mixed | 60/40 | y | [Download](https://drive.google.com/file/d/1nizSK5_RQXsaQ-omKtKL3dwaLL2xJnfC/view?usp=sharing) |
| X-rays Only | 80/20 | y | [Download](https://drive.google.com/file/d/15TBvC-UUYZ4OB_ExNCewHNrZFXdDCPZR/view?usp=sharing) |
| Mixed | 80/20 | y | [Download](https://drive.google.com/file/d/1cO2ck9sJm79tmW-FvawO_ogIL_4yLFpU/view?usp=sharing) |
| X-rays Only | 80/20 | n | [Download](https://drive.google.com/file/d/1fNQndbTef8bu-OPJZHUio4CtTgQMKKxr/view?usp=sharing) |
| Mixed | 80/20 | n | [Download](https://drive.google.com/file/d/11Bs9XbJNKPXaVzKWydvR6r6j9cOFf5ig/view?usp=sharing) |

### Environment setup

Our models were trained on a single CPU (GPUs/TPUs may decrease training time, but are not necessary). The code is implemented using Keras and TensorFlow v2. To install all required dependencies, run the following:

    pip install -r requirements.txt

### Data

All data is stored in this repository and can be accessed [here]().

### Training

Use the following Colab file to train the model: <a href="https://colab.research.google.com/github/rvignav/CT2Xray/blob/master/Segment_Xrays_Only.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Cite
```
@misc{ramesh2021covid19,
      title={COVID-19 Lung Lesion Segmentation Using a Sparsely Supervised Mask R-CNN on Chest X-rays Automatically Computed from Volumetric CTs}, 
      author={Vignav Ramesh and Blaine Rister and Daniel L. Rubin},
      year={2021},
      eprint={2105.08147},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
