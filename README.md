# Activity recognition (Auto-labeling) on videos

A python library is used for recognizing human activities on videos and uses the results to label network traffic (PCAP)
automatically. The library includes two parts: features (data representation) and models (detection models).

# Installation

- python 3.7.9
- pip3 install -r requirements.txt

## Features (only videos)

[comment]: <> (### 1&#41; Traffic features &#40;e.g., IAT+SIZE&#41;)

### 1) CNN features

    Using CNN features extracted by the 'CNN' library [?]

### 2) VideoPose3D

    Using 3D keypoints features extracted by the VideoPose3D library [?]

## Models:

### 1) Onevsrest (e.g., logistic regression)

### 2) SVM (linear kernel)

### 3) Random Forest

## Structure:

    TODO

# How

```python
PYTHONPATH =./: python3
.7
examples / classical / detector_feature_A.py.py 
```

# Issues (continuing updating):

## CNN features:

- tensorflow < 2.0, which requires python3.7 or python3.6
- download pretrained models from https://github.com/tensorflow/models/tree/master/research/slim
  E.g., vgg_16_2016_08_28.tar.gz tar -xvf vgg_16_2016_08_28.tar.gz vgg_16.cpkt
- python3.7 feature_extraction.py --video_list data/video.txt --network vgg --framework tensorflow --output_path out/
  --tf_model ./slim/vgg_16.ckpt --video_list: requires a txt file which lists all the full paths of the videos create an
  output directory by 'mkdir out'

## 3D keypoint features (VideoPose3D)

- using detectron2
 
