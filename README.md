# Few-Cost Salient Object Detection with Adversarial-Paced Learning

This repository is the official implementation of **Few-Cost Salient Object Detection with Adversarial-Paced Learning**. 

## Prerequisites

- [Pytorch 1.4.0+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- [Python3.6+](http://python.org/)

## Requirements

> Note: Please **clone this project** and  install required **pytorch** first

To install requirements:

```setup
pip install -r requirements.txt
```

### Backbone  pre-trained model download 
Please select one of the links below to download resnet101  pre-trained model on COCO
- [BaiduDisk](https://pan.baidu.com/s/1kLkA4LT_Af3-TkT2Gz_lRA): (code: 1234)
- [GoogleDisk](https://drive.google.com/file/d/1gLiF0yKByduIyD7MEya-4ZI_yp5613Ot/view?usp=sharing)

After downloading, put it into `pretrained` folder

### Dataset download 
Please select one of the links below to download related  Saliency Object Detection Dataset
- [BaiduDisk](https://pan.baidu.com/s/11uejtGCS2QkaXyy-Wjgbdg): (code: 9ib7)
- [GoogleDisk](https://drive.google.com/file/d/1yTyQOMhNbEh-P8KzezQ838xsjl12vwox/view?usp=sharing)

After downloading, **unzip** them into `dataset` folder


## Train and test

To train the model in the paper, run this command:

```train
python run.py -pretrain ./pretrained/resnet101COCO-41f33a49.pth -d DUTS -save <saved dir name in logs> -gpu <you GPU number> -part 0.1 -idx ./pretrained/train_id.pkl -l_semi_sal 1 -l_pred_adv 0.01 -l_semi_adv 0.007 -proc FC-SOD
```

> Note: 
>
> 1. **This command will automatically test and evaluate the trained model**.
>
>    If you need not to evaluate, you just specify the `-disable_eval` parameter.
>
>    If you need not to test, you just specify the `-disable_test` parameter.
>
> 2. `-gpu` parameter is the GPU number that you use.(e.g. `0` `0,1`...)
>
> 3. Evaluation code is embedded in this project. 
>
>    If you want to evaluate all dataset just specify the `-eval_d All` parameter.
>
> 4. The evaluation result can be found in  `logs/ExperimentalNotes.md` 

## Evaluation

- To evaluate other trained model (eg,our pre-trained model) on the Saliency Object Detection Dataset, run:

```eval
python run.py -pretrain ./pretrained/resnet101COCO-41f33a49.pth -d DUTS -save <saved dir name in logs> -gpu <you GPU number> -part 0.1 -idx ./pretrained/train_id.pkl -l_semi_sal 1 -l_pred_adv 0.01 -l_semi_adv 0.007 -proc AdvSaliency -disable_train -eval_d ALL -test_model <your trained model path>
```

- If you want test generated results(eg,our pre-trained result) using this project, you need to adapt your folder names to our required structure and put it into `logs` directory. Suppose your directory name is "FCSOD", you can specify `-save FCSOD `  and `-disable_train -diabale_test`, then run this command to evaluate

```eval 
python run.py -pretrain ./pretrained/resnet101COCO-41f33a49.pth -d DUTS -save FCSOD -gpu <you GPU number> -part 0.1 -idx ./pretrained/train_id.pkl -l_semi_sal 1 -l_pred_adv 0.01 -l_semi_adv 0.007 -proc AdvSaliency -disable_train -diabale_test -eval_d ALL
```

### Our required structure

```
logs/
|-- <folder name>
|   `-- test
|       |-- SOD
|       |-- DUTS
|       `-- ...
```



## Pre-trained Models

You can download our pretrained models here:

- [BaiduDisk](https://pan.baidu.com/s/1cIp-nz4Ka2DeJ-aJC2BNpg): (code: gtie)
- [GoogleDisk](https://drive.google.com/file/d/1W_ho0bzQZJ9otzlQl3bnRGnTawMNWkuz/view?usp=sharing)

## Results

Our model achieves the following performance on :

| Dataset | F-measure |  MAE  |
| :-----: | :-------: | :---: |
| DUTS-TE |   0.846   | 0.045 |

You can download our testing result here:

- [BaiduDisk](https://pan.baidu.com/s/1MLO2QK7uRXS7OqCVcS7COA): (code: 0011) 
- [GoogleDisk](https://drive.google.com/file/d/1JXuOY_6OL21tLaxQlWFToUaARl4rAjB1/view?usp=sharing)

## Contributing

If you have any questions, feel free to contact me via: `***@163.com`（）.

Thanks list:

- [SalMetric](https://github.com/Andrew-Qibin/SalMetric)
- [Evaluate-SOD](https://github.com/Hanqer/Evaluate-SOD)
- [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet)

### License
[MIT](LICENSE)

