# Crack Detection Service


## 1. 팀원 소개

![image](https://user-images.githubusercontent.com/67961082/197523059-59ec251f-76a8-41a3-8212-f9f54715bcc7.png)<br><br>


## 2. 프로젝트 주제

드론이 촬영한 영상을 분석하여 0.1mm 수준의 균열을 진단하고, 픽셀 단위로 면적 및 폭을 측정하는 모델을 만든다.<br><br>

## 3. 데이터셋

- [Crack500](https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001) (2560X1440)
- [CrackForest](https://www.kaggle.com/code/mahendrachouhanml/crack-forest/data) (480X320)
- [DeepCrack](https://github.com/yhlleo/DeepCrack) (544X384)
- Training set: #615
- Test set: #115


## 3. 모델링

- Library: pytorch, segmentation models, pytorch-lightning
- Model: DeepLabV3+
- Encoder: Efficientnet-b4
- Optimizer: adam
- Scheduler: reducelr
- Learning rate: 0.001
- Epoch: 100
- Batch size: 16
- Loss function: BCEWithLogitLoss
- threshold: 0.5
- Cross validation: Kfold 5


## 4. 모델 성능

|Parameters|Validation|Test|
|:------:|:----:|:----:|
|Accuracy|0.990|0.983|
|Loss|0.028|0.650|
|Precision|0.836|0.780|
|Recall|0.876|0.780|
|F1 score|0.850|0.704|


## 5. Model serving

#### Flow chart
![image](https://user-images.githubusercontent.com/67961082/197526621-32dd00d3-e280-4fda-8c22-1324141e7c7b.png)


## 6. 시연영상

#### Test1
https://user-images.githubusercontent.com/67961082/197527668-db47fc43-8116-4fed-9b12-59ac1654d0c2.mp4


#### Test2
https://user-images.githubusercontent.com/67961082/197527698-73ea2fdc-5e2d-4324-97ee-4e09740d61de.mp4


