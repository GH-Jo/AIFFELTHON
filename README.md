![LOGO](logo.png)



# Socar 차량 파손 영역 및 유형 탐지 딥러닝 모델



Members : 안레오나르도, 조근혜, 이경희

<P>

## Procedure

1. EDA 
2. DataSet Implementation
3. Modeling
4. Validation



## Details

Step 1. EDA

쏘카에서 받은 이미지 폴더 구성
```
보안상 삭제
```
이미지 구성 특징
- scratch와 dent는 이미지 이름이 동일
- spacing은 scratch / dent와 이름이 다르다
- scratch 에는 scratch에 해당되는 mask만 존재
- dent 에는 dent에 해당되는 mask만 존재
- spacing 에는 spacing에 해당되는 mask만 존재

UNET용 csv 파일 만들기 전략
- UNET은 한 모델에서 1개의 segment를 예측
- spacing,dent,scratch 존재유무 컬럼을 만들때 겹치지않도록 해야한다
- scratch 와 dent 둘다 mask_sum >0 값이 존재하면 동일 이름으로 2개의 열(row)이 생성되도록 해야한다
다
Mask RCNN용 csv 파일 만들기 전략
- 한 모델에서 spacing dent, scratch 를 예치
- spacing,dent,scratch 존재유무 컬럼을 만들때 겹치도록 해야한다
- scratch 와 dent 둘다 mask_sum >0 값이 존재하면 동일 이름으로 1개의 열(row)이 생성되고 dent, scratch 컬럼에 True/False 로 표시하도록 한다
```
  <class 'pandas.core.frame.DataFrame'>
RangeIndex: 3113 entries, 0 to 3112
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   filename  3113 non-null   object 
 1   spacing   3113 non-null   bool   
 2   dent      3113 non-null   bool   
 3   scratch   3113 non-null   bool   
 4   prefix    3113 non-null   object 
 5   mask_sum  3113 non-null   float64
 6   label     3113 non-null   bool   
dtypes: bool(4), float64(1), object(2)
```


Step 2. DataSet Implementation

Mask RNN용 Dataset 구현 특이사항
- 스크래치 처럼 작은 영역 10x10 pixel은 제외시키는 부분 구현
- 학습할때 정상 이미지를 제공할지 여부를 선택하는 부분 구현
- mask에서 polygon 영역을 추출하고 영역별 별도의 mask를 생성하는 부분 구항
- 임의로 flip/rotation 기능 적용하는 기능 구현

UNET용 Dataset 과 CNN용 Dataset은 일반적인 Dataset 구현방법이 적용됨


Step 3. Modeling

Classification Model
- resnet50 FineTunning
- resnet50(Freeze) + SVM

UNET Model
- encoder resnet50 + decoder unet
- encoder efficientNetB7 + decoder unet

Mask RCNN Model
- resnet50_fpn + FastRCNNPredictor + MaskRCNNPredictor


Step 4. Validation
- Classification Model
  - normal/problem Val Accuracy : 0.937
- UNET Model 
   - Scratch Val mIoU :0.36
   - Dent Val mIoU    :0.08
   - Spacing Val mIoU :0.10
 - Mask RCNN Model
   - spacing/dent/scratch Val Mask Average Precision: 0.139
   - spacing/dent/scratch Val Mask Average Recall: 0.245



## Reference

1. SOCAR Tech Blog : https://tech.socarcorp.kr/data/2020/02/13/car-damage-segmentation-model.html#1-classification
2. Classification : https://wonwooddo.tistory.com/47
3. Mask RCNN : https://pytorch.org/vision/main/models/mask_rcnn.html
4. Segmentation : https://github.com/qubvel/segmentation_models.pytorch
