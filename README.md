# 기업연계프로젝트(스마트인사이드AI): Crack Detection Service


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


## 7. 어려운 점

#### 7-1. Target이 작은 이미지에서의 Segmantic Segmentation
- target이 작아 segmentation model이 과적합될 가능성이 크다.
- 실제 현장의 이미지에서는 페인트의 벗겨짐 등 균열과 비슷한 부분이 존재하는데, 모델이 이런 부분까지 crack으로 분류하는 문제 발생


#### 7-2. 데이터셋 문제
- 이미지의 정확한 축척을 알지 못하기 때문에 실제 사이즈(cm, mm)로 측정하지 못하고 픽셀로만 나타내줘야 하는 한계가 있다.
- 기업으로부터 드론이 촬영한 이미지를 제공받지 못해 양질의 데이터셋을 구축하는데 어려움이 있다.


## 8. 개선할 점

#### 8-1. 모델 서빙
- 현재는 flask로 model serve구현
- torch serve를 이용해 더 최적화된 서비스 제공이 가능할 것 같다.

#### 8-2. Inference속도 개선
- skeletonize와 canny edge detection으로 균열의 폭을 구할 때 이중 for문의 사용으로 시간이 많이 소요된다.

#### 8-3. Object detection model사용
- 페인트 벗겨짐 등 균열과 유사하게 보이는 부분을 segmentation 모델만으로는 정확하게 분류하기 어렵다.
- Object detection model을 추가로 생성하여 two step으로 균열을 진단하면 보다 정교한 균열진단이 가능할 것 같다.
