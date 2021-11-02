# Paddle-VisualAttention

## Results_Compared 

[SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)

<table>
    <tr>
        <th>Methods</th>
        <th>Steps</th>
        <th>GPU</th>
        <th>Batch Size</th>
        <th>Learning Rate</th>
        <th>Patience</th>
        <th>Decay Step</th>
        <th>Decay Rate</th>
        <th>Training Speed (FPS)</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td>PaddlePaddle_SVHNClassifier</td>
        <td>
            <a href="https://drive.google.com/open?id=1DSg3F5GpouEvU9n7YSPdUKH1CSmkdwSw">
                54000
            </a>
        </td>
        <td>GTX 1080 Ti</td>
        <td>1024</td>
        <td>0.01</td>
        <td>100</td>
        <td>625</td>
        <td>0.9</td>
        <td>~1700</td>
        <td>95.65%</td>
    </tr>
    <tr>
        <td>Pytorch_SVHNClassifier</td>
        <td>
            <a href="https://drive.google.com/open?id=1DSg3F5GpouEvU9n7YSPdUKH1CSmkdwSw">
                54000
            </a>
        </td>
        <td>GTX 1080 Ti</td>
        <td>512</td>
        <td>0.16</td>
        <td>100</td>
        <td>625</td>
        <td>0.9</td>
        <td>~1700</td>
        <td>95.65%</td>
    </tr>
</table>

# Introduction

The main idea of this exercise is to study the evolvement of the state of the art and main work along topic of visual attention model. There are two datasets that are studied: augmented MNIST and SVHN. The former dataset focused on canonical problem  —  handwritten digits recognition, but with cluttering and translation, the latter focus on real world problem  —  street view house number (SVHN) transcription. In this exercise, the following papers are studied in the way of developing a good intuition to choose a proper model to tackle each of the above challenges.

For more detail, please refer to this [blog]()

## Recommended environment
```
Python 3.6+
paddlepaddle-gpu 2.0.2
nccl 2.0+
editdistance
visdom
h5py
protobuf
lmdb
```

## Install

### Install env
Install paddle following the official [tutorial](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html).
```shell script
pip install visdom
pip install h5py
pip install protobuf
pip install lmdb
```
## Dataset
1. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

2. Extract to data folder, now your folder structure should be like below:
    ```
    SVHNClassifier
        - data
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
    ```

## Usage

1. (Optional) Take a glance at original images with bounding boxes

    ```
    Open `draw_bbox.ipynb` in Jupyter
    ```

1. Convert to LMDB format

    ```
    $ python convert_to_lmdb.py --data_dir ./data
    ```

1. (Optional) Test for reading LMDBs

    ```
    Open `read_lmdb_sample.ipynb` in Jupyter
    ```

1. Train

    ```
    $ python train.py --data_dir ./data --logdir ./logs
    ```

1. Retrain if you need

    ```
    $ python train.py --data_dir ./data --logdir ./logs_retrain --restore_checkpoint ./logs/model-100.pth
    ```

1. Evaluate

    ```
    $ python eval.py --data_dir ./data ./logs/model-100.pth
    ```

1. Visualize

    ```
    $ python -m visdom.server
    $ python visualize.py --logdir ./logs
    ```

1. Infer

    ```
    $ python infer.py --checkpoint=./logs/model-100.pth ./images/test1.png
    ```

1. Clean

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs_retrain
    ```
