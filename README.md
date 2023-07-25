# 주차장 빈공간 개수 찾기 프로젝트
딥러닝 모델 VGG16과 RCNN을 사용해 주차장 사진에서 빈공간의 개수를 찾는 프로젝트


## 데이터 

### 출처 
<https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset>

### 원본 이미지
![2012-09-11_15_16_58](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/731435a9-ce0b-42eb-83d6-5f2169a51e11)

### VGG16용 이미지
- occupied
  
![2012-09-11_15_16_58__058](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/3f3b5aa2-9692-4ef8-bea0-768c2c6b362a)

- empty

![2012-09-11_15_27_08__008](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/236a9712-cf4b-42b8-b09e-064fd98933cc)


### xml 파일
- parking id : 주차장 이름
- occupied : 주차 여부 (비어 있으면 0, 아닐 땐 1)
- space id : 주차 공간 식별 번호
- contour : 주차 공간의 네 꼭짓점 좌표

![image](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/663d7c00-78d0-4896-a759-d269312fd387)


## 데이터 전처리

### xml을 csv로 변환 
[xml_parsing](https://github.com/pabihe0223/Parking-Lot-Object-Detection/blob/d3127da588dfa47e65fe6ddf87b0877d7731f931/xml_parsing.ipynb)

### csv 파일 데이터 전처리 진행

[DataPreProcessingMean](https://github.com/pabihe0223/Parking-Lot-Object-Detection/blob/46dd3c27a253cec57de0d18f748042b678fd1846/DataPreprocessingMean.ipynb)


    PK_data = pd.read_csv('all_data.csv')

![image](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/de0d0847-ce0c-4e8c-8254-ea43188bb2cf)

    #PK_data에서 occupied column의 NULL값이 있는 데이터프레임 생성
    PK_occcupied_na = PK_data.loc[PK_data.occupied.isnull()

---
    #원본 이미지를 기준으로 NULL값의 개수가 평균보다 많은 이미지는 제거
    na_remove=[]
    for i in range(len(cnt)):
      if cnt[i]>na_avg:
         na_remove.append(cnt.index[i])
    PK_data_na_remove=pd.DataFrame()

    for i in range(len(na_remove)):
     PK_data_na_remove=PK_data_na_remove.append(PK_data[PK_data['filename'].str[0:19]==na_remove[i]])

![image](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/9832de44-b15a-4724-a67d-79f5b27f5875)

    exfinal_PK_data=pd.merge(PK_data,PK_data_na_remove,how='outer',indicator=True)
    final_PK_data=exfinal_PK_data.query('_merge=="left_only"').drop(columns=['_merge'])
    final_PK_data
---

    # NULL 값의 개수가 평균보다 적은 파일은 occupied를 0으로 채움
    final_PK_data['occupied']=final_PK_data['occupied'].fillna(0)

![image](https://github.com/pabihe0223/Parking-Lot-Object-Detection/assets/106232513/25a210b2-6382-4e53-b56e-37e0dc4eef38)

- 세 개의 주차장별로 나뉘었던 폴더와 하위 폴더들을 하나의 폴더로 통합
- 데이터 증식 객체를 통해 이미지의 개수를 늘렸기 때문에 모든 이미지를 사용하지 않고 일부만 랜덤으로 선택


## 모델  

### VGG16
- tensorflowmodel keras 모듈 안에 있는 VGG16 모델 사용
- 전체 이미지 중 1800개 학습 450개 검증
- [vgg16 source code](https://github.com/pabihe0223/Parking-Lot-Object-Detection/blob/2903f7f9681dc9200b2889c49148d878b681837d/vgg16_parking_lot.ipynb)

#### 참고 자료
[vgg16](https://github.com/gsadhas/real-time-parking-occupancy-detection/blob/688ea3a5756b329de7ceefb90b163e02f607ca89/cnn_models_vgg16.ipynb)


### RCNN
- torchvisions 모듈 안에 있는 RCNN 모델 사용
- 전체 이미지(12162개) 이미지 모두사용
- [rcnn source code](https://github.com/pabihe0223/Parking-Lot-Object-Detection/blob/leehj/project.ipynb)

#### 참고 자료
- https://hyungjobyun.github.io/machinelearning/FasterRCNN2/
- https://youtu.be/jqNCdjOB15s
