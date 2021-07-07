## End-to-end Malaria Diagnosis and 3D Cell Rendering with Deep Learning
<a href="https://orcid.org/0000-0002-6521-7898"><img height="15" src="https://github.com/rvignav/CT2Xray/blob/master/docs/orcid.png"></a>&nbsp;Vignav Ramesh

[arXiv](https://arxiv.org/abs/2105.08147) / [Full Paper (PDF)](https://arxiv.org/pdf/2105.08147.pdf) / [Papers With Code](https://paperswithcode.com/paper/covid-19-lung-lesion-segmentation-using-a) / [Website](https://topdocmedicine.wixsite.com/topdoc)

![Header](https://github.com/rvignav/E2EMD/blob/main/docs/sshot.png)

### Pre-trained models

| Model | Pruned (y/n) | Weights | 
| --- | --- | --- | 
| Vanilla CNN  | n | [v1](https://github.com/rvignav/E2EMD/blob/main/weights/CNN-V1Weights.h5), [v2](https://github.com/rvignav/E2EMD/blob/main/weights/CNN-V2Weights.h5) |
| VGG-19 | n  | [Download](https://github.com/rvignav/E2EMD/blob/main/weights/VGGWeights.h5) |
| VGG-19 | y  | [Download](https://github.com/rvignav/E2EMD/blob/main/weights/finalPrunedWeights.h5) |

### Environment setup

Our models were trained on a single CPU (GPUs/TPUs may decrease training time, but are not necessary). The code is implemented using Keras and TensorFlow v2. To install all required dependencies, run the following:

    pip install -r requirements.txt

### Data

All data is stored in this repository and can be accessed [here](https://github.com/rvignav/E2EMD/tree/main/cell_images).

### Training

Use the following Colab file to train the model: <a href="https://colab.research.google.com/github/rvignav/E2EMD/blob/main/VGG.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
