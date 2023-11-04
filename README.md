# PaDiM_anomaly_detection_and_localization_pytorch

This repostory contains a set of functions and classes for performing anomaly detection and localization using an  unofficial implementations of [**PaDiM**](https://arxiv.org/abs/2011.08785) in pytorch.


This code is heavily borrowed from both SPADE-pytorch(https://github.com/byungjae89/SPADE-pytorch), MahalanobisAD-pytorch(https://github.com/byungjae89/MahalanobisAD-pytorch), anodet: anomaly detection on images using features from pretrained neural networks. https://github.com/OpenAOI/anodet, and PaDiM-Anomaly-Detection-Localization-master(https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master#reference) projects.



## Training and testing

From the root folder run

```bash

python main.py --data_path path/to/dataset --save_path path/to/output --backbone wide_resnet50_2

```

or simply run,

```bash
sh run.sh 
```


## Feature extraction & (pre-trained) backbones

The pre-trained backbones come from torchvision. ``Note`` the current code version only handles, resnet based backbones and variants. 

## Discussion and possible improvements:

[1] `More data:` As non-defective images are easier to come by, and each pair will have at least one, means that the solution can be further improved by learning from these images as well. Note that this does not require any annotations or labeling. 

[2] `Fine-tune backbones:` The used backbones were trained on data from the ImageNet dataset, which fundamentally differs from the images used here. A good idea would be to fine-tune the backbones of such data before it is used for anomaly detection.  

[3] `Ensemble models:` Combining predictions from diverse backbones can enhance overall performance by mitigating individual model biases and improving generalization. Additionally, implementing k-fold cross-validation techniques aids in achieving robust and reliable results, as it systematically evaluates the model's performance on different subsets of the data, reducing the risk of overfitting and yielding more trustworthy assessments of its effectiveness.

[4] `Student–Teacher architectures:` In my opinion, the next step will be to incorporate a student–teacher approach to detect anomalous features, in which the student network is trained to predict the extracted features of normal, i.e., anomaly-free training images. The student failure to predict their features enables the detection of anomalies at test time.

## Code Reference

pytorch_cov function in ``utils.py``:
https://github.com/pytorch/pytorch/issues/19037

Code in the directory ``visualization``:
https://github.com/google/active-learning



## Reference

[1] Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. https://arxiv.org/pdf/2011.08785

[2] Niv Cohen and Yedid Hoshen. Sub-Image Anomaly Detection with Deep Pyramid Correspondences (SPADE) in PyTorch. https://github.com/byungjae89/SPADE-pytorch. https://arxiv.org/abs/2005.02357 


[3] Oliver Rippel, Patrick Mertens, and Dorit Merhof. Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection
. https://github.com/byungjae89/MahalanobisAD-pytorch. https://arxiv.org/abs/2005.14140

[4] https://github.com/openvinotoolkit/anomalib

[5] https://medium.com/@niitwork0921/what-is-anomaly-detection-and-what-are-some-of-its-algorithms-735246d265c9

[6] https://medium.com/openvino-toolkit/hands-on-lab-how-to-perform-automated-defect-detection-using-anomalib-5c1cfed666b4










