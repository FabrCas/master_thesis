
# Enhancing Abnormality identification: Robust Out-Of-Distribution strategies for Deepfake Detection
Novel architectures and strategies for **OOD detection** employing the collaborative efforts of In-Distribution classifiers and Out-Of-Distribution detectors.
These techniques are used to improve the robustness of **Deepfake detection**.
Our study integrates *Convolutional Neural Networks* (CNN) and *Vision Transformers* (ViT), presenting two distinct architectures related to a common Pipeline. The first exploits the image reconstruction capabilities of the CNN model, while the second integrates the attention estimation in the study.


## Data

The **CDDB** dataset can be downloaded at the following link: [Download](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view)
## Models
You can download the pre-trained models from the following link: [Download](https://drive.google.com/drive/folders/1S6XIWuHg746lyAacccv6HkgICMlBxOvt?usp=sharing)
## Installation
To install the required dependencies, run the following command:
```bash python -m 
git clone https://github.com/FabrCas/master_thesis.git
cd master_thesis
pip install -r requirements.txt
```
### Workspace File System
```bash python -m 
```

## Launching the Software
To launch the software, run the following command:

    python main.py

## References
### Dataset
**CDDB**: Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials. [paper](https://arxiv.org/abs/2205.05467).

### OOD benchmarks
- **CIFAR-10/CIFAR-100**.
Official page: [link](https://www.cs.toronto.edu/~kriz/cifar.html)
- **MNIST**, The MNIST Database of Handwritten Digit Images for Machine Learning Research.
[paper](https://ieeexplore.ieee.org/document/6296535)
- **FMINST**, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
[paper](https://arxiv.org/abs/1708.07747)
- **SVHN**.
Official page: [link](http://ufldl.stanford.edu/housenumbers/)
- **DTD**, Describing Textures in the Wild.
[paper](https://arxiv.org/abs/1311.3618)
- **Tiny ImageNet**, 80 Million Tiny Images: A Large Data Set for Nonparametric Object and Scene Recognition.
[paper](https://ieeexplore.ieee.org/document/4531741)


### Models

- **ResNet**, Deep Residual Learning for Image Recognition.
[paper](https://arxiv.org/abs/1512.03385v1)
- **AutoEncoder**, Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction
[paper](https://link.springer.com/chapter/10.1007/978-3-642-21735-7_7)
- **VAE**, Auto-Encoding Variational Bayes
[paper](https://arxiv.org/abs/1312.6114)
- **U-net**, Convolutional Networks for Biomedical Image Segmentation.
[paper](https://arxiv.org/abs/1505.04597)
- **ViT** (Vision Transformer), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
[paper](https://arxiv.org/abs/2010.11929)
- **DeiT**, Training data-efficient image transformers & distillation through attention
[paper](https://arxiv.org/abs/2012.12877)

### OOD resources

- A **Baseline** for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.
[paper](https://arxiv.org/abs/1610.02136)
- **CutMix**, Regularization Strategy to Train Strong Classifiers with Localizable Features.
[paper](https://arxiv.org/abs/1905.04899)
- **ODIN**, Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks.
[paper](https://arxiv.org/abs/1706.02690)
- **Confidence Branch**, Learning Confidence for Out-of-Distribution Detection in Neural Networks.
[paper](https://arxiv.org/abs/1802.04865)
- **Visual Attention**, Leveraging Visual Attention for out-of-distribution Detection.
[paper](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Cultrera_Leveraging_Visual_Attention_for_out-of-Distribution_Detection_ICCVW_2023_paper.pdf)

## License
This project is licensed under the MIT License.

## Contact

 - Email: fabriziocasadei97@gmail.com
 - Linkedin: [Page](https://www.linkedin.com/in/fabrizio-casadei-056778179/)
