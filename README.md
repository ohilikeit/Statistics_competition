![206994_208214_224](https://user-images.githubusercontent.com/37128004/197730936-312a4914-0429-4c2a-aae0-533d36ff8478.jpg)
# CNN을 활용한 산업분류 자동화 모델 -NLP-
***자연어 기반의 산업 분류 데이터셋을 학습하여 자동화하는 알고리즘을 개발하는 대회***

## Introduction
통계청이 주최하는 2022 통계데이터 인공지능 활용대회에 참여한 결과물이다. 자연어 기반의 산업 분류 모델을 만드는 것이였으나 아직 PLM(Pretrained language model)을 공부하기 전이라 알고 있던 CNN 모델구조를 활용하였다. 

## Environment
- Google Colab Pro(TPU)
- Tensorflow 2.9.2
- keras 2.9.0
- 

## Data
- train(1,000,000 rows X 7 columns)
  - 전국사업체조사 샘플데이터
  - AI_id, digit_1(대분류), digit_2(중분류), digit_3(소분류), text_obj(무엇을 가지고), text_mthd(어떤 방법으로), text_deal(생산,제공하였는가)
- submit(100,000 rows X 4 columns)
  - 예측용 test data
  - digit_1, digit_2, label, document

## Preprocess
- class imbalance
  - few class data(<500) augmentation with EDA(Easy Data Augmentation) & class weights
  - class_weight : 
- tokenizing
  - 불용어 처리 및 okt 형태소 분석기를 통한 형태소 단위 임베딩
- train_test_split
  - train : valid = 9 : 1

## Model
### Structure
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 30)]              0         
                                                                 
 embedding (Embedding)       (None, 30, 1024)          40950784  
                                                                 
 conv1d (Conv1D)             (None, 30, 1024)          3146752   
                                                                 
 batch_normalization (BatchN  (None, 30, 1024)         4096      
 ormalization)                                                   
                                                                 
 global_max_pooling1d (Globa  (None, 1024)             0         
 lMaxPooling1D)                                                  
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 dense (Dense)               (None, 256)               262400    
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 225)               57825     
                                                                 
=================================================================
Total params: 44,421,857
Trainable params: 44,419,809
Non-trainable params: 2,048
_________________________________________________________________


















## Conclusion
처음 참여해본 대회로 구현 자체도 어려움이 많았다. 
