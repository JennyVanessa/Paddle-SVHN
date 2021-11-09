# Paddle-SVHN

This project reproduces [Multi-digit Number Recognition from Street View
Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082.pdf) based on the paddlepaddle framework and participates in the Baidu paper reproduction competition.)


# 1 Introduction

The main idea of this exercise is to study the evolvement of the state of the art and main work along topic of visual attention model. There are two datasets that are studied: augmented MNIST and SVHN. The former dataset focused on canonical problem  —  handwritten digits recognition, but with cluttering and translation, the latter focus on real world problem  —  street view house number (SVHN) transcription. In this exercise, the following papers are studied in the way of developing a good intuition to choose a proper model to tackle each of the above challenges.

### Paper:

* [1] Goodfellow I J, Bulatov Y, Ibarz J, et al. Multi-digit number recognition from street view imagery using deep convolutional neural networks[J]. arXiv preprint arXiv:1312.6082, 2013.

### Reference project

* https://github.com/potterhsu/SVHNClassifier-PyTorch

### The link of aistudio:

* AI Studio: https://aistudio.baidu.com/aistudio/projectdetail/2598446?forkThirdPart=1

# 2 Results_Compared 


<table>
    <tr>
        <th>Methods</th>
        <th>Model Download</th>
        <th>Batch Size</th>
        <th>Learning Rate</th>
        <th>Patience</th>
        <th>Decay Step</th>
        <th>Decay Rate</th>
        <th>Training Speed (FPS)</th>
        <th>Accuracy</th>
    </tr>
    <tr>
        <td>Pytorch_SVHN</td>
        <td>
            <a href="https://drive.google.com/open?id=1DSg3F5GpouEvU9n7YSPdUKH1CSmkdwSw">
                torch_model
            </a>
        </td>
        <td>512</td>
        <td>0.16</td>
        <td>100</td>
        <td>625</td>
        <td>0.9</td>
        <td>~1700</td>
        <td>95.65%</td>
    </tr>
        <tr>
        <td>Paddle_SVHN</td>
        <td>
            <a href="https://pan.baidu.com/s/1g3ZXCF2mCCXvxhTUUsKjYg">
                paddle_model
            </a>
        </td>
        <td>1024</td>
        <td>0.01</td>
        <td>100</td>
        <td>625</td>
        <td>0.9</td>
        <td>~1700</td>
        <td>95.36%</td>
    </tr>
</table>

# 3 Dataset

* [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1
  * test.tar.gz
  * train.tar.gz
  * extra.tar.gz


# 4 Recommended environment

* Python 3.6
* paddlepaddle-gpu 2.0.2
* visdom
* protobuf
* lmdb


# 5 Start

### step1: Clone
```
git clone https://github.com/JennyVanessa/Paddle-SVHN.git
cd Paddle-SVHN
```

### step2: Pip installl

Install paddle following the official [tutorial](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html).
```shell script
pip install visdom
pip install h5py
pip install protobuf
pip install lmdb
```
### step3: Download dataset

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

### step4: Convert Dataset to Lmdb format

    $ python convert_to_lmdb.py --data_dir ./data


### step5: Train
   Save training log in train.log file and save trained model in ./logs directory

    $ python train.py --data_dir ./data --logdir ./logs >> train.log
    
   The output is:
   
    data/test.lmdb
    Start training
    => 2021-11-01 16:47:58.488561: step 100, loss = 7.821150, learning_rate = 0.100000 (2290.5 examples/sec)
    => 2021-11-01 16:48:43.666231: step 200, loss = 7.897614, learning_rate = 0.100000 (2284.5 examples/sec)
    => 2021-11-01 16:49:30.083858: step 300, loss = 7.818493, learning_rate = 0.100000 (2293.1 examples/sec)
    => 2021-11-01 16:50:15.438407: step 400, loss = 7.806008, learning_rate = 0.100000 (2276.1 examples/sec)
    => 2021-11-01 16:51:02.383038: step 500, loss = 7.821648, learning_rate = 0.100000 (2284.2 examples/sec)
    => 2021-11-01 16:51:47.870291: step 600, loss = 7.811975, learning_rate = 0.100000 (2269.1 examples/sec)
    => 2021-11-01 16:52:34.556187: step 700, loss = 7.864832, learning_rate = 0.100000 (2283.2 examples/sec)
    => 2021-11-01 16:53:20.091155: step 800, loss = 7.786717, learning_rate = 0.090000 (2266.6 examples/sec)
    => 2021-11-01 16:54:06.938361: step 900, loss = 7.849339, learning_rate = 0.090000 (2278.9 examples/sec)
    => 2021-11-01 16:54:52.568350: step 1000, loss = 7.795635, learning_rate = 0.090000 (2261.9 examples/sec)
    => Model saved to file: ./logs/model-1000.pdparams
    => patience = 100
    => Evaluating on validation dataset...
    ==> accuracy = 0.022880, best accuracy 0.000000
    ...
    

   Retrain if you need

    
    $ python train.py --data_dir ./data --logdir ./logs_retrain --restore_checkpoint ./logs/model-100.pdparams
    
### step6: Evaluate

    $ python eval.py --data_dir ./data ./logs/model-100.pdparams
    
   The output is:
   
    Start evaluating
    Evaluate /home/aistudio/logs/model-359000.pdparams on /home/aistudio/data/test.lmdb, accuracy = 0.953551
    Done
    
### step7: Infer

    
    $ python infer.py --checkpoint=./logs/model-100.pdparams ./image/test1.png
    
   The test1.png shows:
   
   ![test1.png](https://github.com/JennyVanessa/Paddle-SVHN/blob/main/image/test1.png)
   
   The infer output is:
   
    length: 2
    digits: 7 5 10 10 10
    

### step8: Clean

    
    $ rm -rf ./logs
    or
    $ rm -rf ./logs_retrain
    
    
# 6 Code Structure
```
├─convert_to_lmdb.py                         
├─dataset.py                
├─eval.py                           
├─model.py    
├─evaluator.py
├─draw_bbox.py
├─example_pb2.py
├─infer.py
├─read_lmdb_sample.py
├─visiualize.py
├─train.py
├─train.log
├─images                          
│  test1.png              
                    

```
    
