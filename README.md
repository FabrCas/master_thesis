# Enhancing Abnormality identification: <br>Robust Out-Of-Distribution strategies for Deepfake Detection

Novel architectures and strategies for **OOD detection** employing the collaborative efforts of In-Distribution classifiers and Out-Of-Distribution detectors.

These techniques are used to improve the robustness of **Deepfake detection**. Our study integrates *Convolutional Neural Networks* (CNN) and *Vision Transformers* (ViT), presenting two distinct architectures related to a common Pipeline. The first exploits the image reconstruction capabilities of the CNN model, while the second integrates the attention estimation in the study. Auxiliary data produced by the ID classifier and other components are exploited by the custom Abnormality module to infer whether a sample is Out-Of-Distribution.

The full treatment of this research study is covered in this [pdf file](https://github.com/FabrCas/master_thesis/blob/main/thesis.pdf).

  

## Data

  

The **CDDB** dataset can be downloaded at the following link: [Download](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view)

  

## Models

  

You can download the pre-trained models from the following link: [Download](https://drive.google.com/drive/folders/1S6XIWuHg746lyAacccv6HkgICMlBxOvt?usp=sharing)

  

## Installation

  

To install the required dependencies, run the following command:

  

```bash python -m

  

git  clone  https://github.com/FabrCas/master_thesis.git

  

cd  master_thesis

  

pip  install  -r  ./setup/requirements.txt

  

```

  

Run main file to create the necessary folders.

  

```bash

  

python  main.py

  

```

  

Then move the CDDB dataset in the data folder and pretrained models in the models folder, unzipping files.

  

  

### Workspace File System

  

```bash python -m

├──  data/

├──  models/

├──  results/

├──  scripts/

├──  setup/

├──  static/

├──  bin_classifier.py

├──  bin_ViTClassifier.py

├──  dataset.py

├──  experiments.py

├──  __init__.py

├──  main.py

├──  models_2.py

├──  models.py

├──  multi_classifier.py

├──  ood_detection.py

├──  README.md

├──  LICENSE

├──  test_dataset.py

├──  test_models.py

├──  launch_bin_classifier.py

├──  launch_bin_ViTClassifier.py

├──  launch_experiments.py

├──  launch_ood_detector.py

└──  utilities.py

```

  

## Launching the Software

  

To launch the software and test the approaches proposed, run the following command:

  

python main.py

Use the following main parameters:

 -  **--help**, get execution details.

 -  **--useGPU**, to specify a local GPU runtime. Specify between True and False. Defaults to True.

 -  **--verbose**, give additional information in your console printouts. Specify between True and False. Defaults to True.

 - **--m**, choose which test to run among:
   * **benchmark_synth**, CIFAR10 OOD benchmark using only synthetic data.
   * **benchmark**, CIFAR10 OOD benchmark using synthetic and real outliers.
   * **df_content**, Deepfake detection in Content scenario (Faces) with ViT based approach.
   * **df_group**, Deepfake detection in Group scenario (GANs) with U-Net based approach.
   * **df_mix**, Deepfake detection in Mix scenario with U-Net based approach.
   * **abn_content_synth**, OOD detection in Content scenario (Faces) using only sinthetic data.
   * **abn_content**, OOD detection in Content scenario (Faces) using synthetic and real outliers.
   * **abn_group**, OOD detection in Group scenario (GANs) using synthetic and real outliers.
   * **abn_mix**, OOD detection in Mix scenario using synthetic and real outliers.

  

Example:

  

python main.py --verbose True --useGPU True --m benchmark

  

Single modules for deepfake and OOD detection, i.e. ood_detection.py, can be utilized following procedure in launch_*.py files. An example:

```python

from bin_ViTClassifier import DFD_BinViTClassifier_v7

from ood_detection import Abnormality_module_ViT

  

# Define In-Distributution Classifier

scenario =  "content"

classifier_name =  "faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024"

classifier_type =  "ViTEA_timm"

autoencoder_type =  "vae"

prog_model_timm =  3  # (tiny DeiT)

classifier_epoch =  25

autoencoder_epoch =  25

classifier = DFD_BinViTClassifier_v7(scenario=scenario,  model_type=classifier_type,  model_ae_type  = autoencoder_type,  prog_pretrained_model= prog_model_timm)

# load classifier & autoencoder

classifier.load_both(classifier_name, classifier_epoch, autoencoder_epoch)

  

# Train Abnormality module

type_encoder =  "encoder_v3"

abn = Abnormality_module_ViT(classifier,  scenario  = scenario,  model_type= type_encoder)

abn.train()

```

## References

  

### Dataset

  

**CDDB**: Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials. [paper](https://arxiv.org/abs/2205.05467).

  

  

### OOD benchmarks

  

-  **CIFAR-10/CIFAR-100**.

  

Official page: [link](https://www.cs.toronto.edu/~kriz/cifar.html)

  

-  **MNIST**, The MNIST Database of Handwritten Digit Images for Machine Learning Research.

  

[paper](https://ieeexplore.ieee.org/document/6296535)

  

-  **FMINST**, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.

  

[paper](https://arxiv.org/abs/1708.07747)

  

-  **SVHN**.

  

Official page: [link](http://ufldl.stanford.edu/housenumbers/)

  

-  **DTD**, Describing Textures in the Wild.

  

[paper](https://arxiv.org/abs/1311.3618)

  

-  **Tiny ImageNet**, 80 Million Tiny Images: A Large Data Set for Nonparametric Object and Scene Recognition.

  

[paper](https://ieeexplore.ieee.org/document/4531741)

  

  

### Models

  

  

-  **ResNet**, Deep Residual Learning for Image Recognition.

  

[paper](https://arxiv.org/abs/1512.03385v1)

  

-  **AutoEncoder**, Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

  

[paper](https://link.springer.com/chapter/10.1007/978-3-642-21735-7_7)

  

-  **VAE**, Auto-Encoding Variational Bayes

  

[paper](https://arxiv.org/abs/1312.6114)

  

-  **U-net**, Convolutional Networks for Biomedical Image Segmentation.

  

[paper](https://arxiv.org/abs/1505.04597)

  

-  **ViT** (Vision Transformer), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

  

[paper](https://arxiv.org/abs/2010.11929)

  

-  **DeiT**, Training data-efficient image transformers & distillation through attention

  

[paper](https://arxiv.org/abs/2012.12877)

  

  

### OOD resources

  

  

- A **Baseline** for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.

  

[paper](https://arxiv.org/abs/1610.02136)

  

-  **CutMix**, Regularization Strategy to Train Strong Classifiers with Localizable Features.

  

[paper](https://arxiv.org/abs/1905.04899)

  

-  **ODIN**, Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks.

  

[paper](https://arxiv.org/abs/1706.02690)

  

-  **Confidence Branch**, Learning Confidence for Out-of-Distribution Detection in Neural Networks.

  

[paper](https://arxiv.org/abs/1802.04865)

  

-  **Visual Attention**, Leveraging Visual Attention for out-of-distribution Detection.

  

[paper](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Cultrera_Leveraging_Visual_Attention_for_out-of-Distribution_Detection_ICCVW_2023_paper.pdf)

  

  

## License

  

This project is licensed under the MIT License.

  

  

## Contact

  

  

- Email: fabriziocasadei97@gmail.com

  

- Linkedin: [Page](https://www.linkedin.com/in/fabrizio-casadei-056778179/)