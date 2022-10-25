![206994_208214_224](https://user-images.githubusercontent.com/37128004/197730936-312a4914-0429-4c2a-aae0-533d36ff8478.jpg)
# CNN을 활용한 산업분류 자동화 모델 -NLP-
***자연어 기반의 산업 분류 데이터셋을 학습하여 자동화하는 알고리즘을 개발하는 대회***

## Introduction
통계청이 주최하는 2022 통계데이터 인공지능 활용대회에 참여한 결과물이다. 자연어 기반의 산업 분류 모델을 만드는 것이였으나 아직 PLM(Pretrained language model)을 공부하기 전이라 알고 있던 CNN 모델구조를 활용하였다. 

## Environment
- Google Colab Pro(TPU)
- Tensorflow 2.9.2
- keras 2.9.0

## Data
- train(1,000,000 rows X 7 columns)
  - 전국사업체조사 샘플데이터
  - AI_id, digit_1(대분류), digit_2(중분류), digit_3(소분류), text_obj(무엇을 가지고), text_mthd(어떤 방법으로), text_deal(생산,제공하였는가)
```python
digit_1	digit_2	label	document
0	S	95	952	카센터에서 자동차부분정비 타이어오일교환
1	G	47	472	상점내에서 일반인을 대상으로 채소.과일판매
2	G	46	467	절단하여사업체에도매 공업용고무를가지고 합성고무도매
3	G	47	475	영업점에서 일반소비자에게 열쇠잠금장치
4	Q	87	872	어린이집 보호자의 위탁을 받아 취학전아동보육
...	...	...	...	...
999995	C	13	134	제품입고 워싱 청바지워싱
999996	F	42	424	현장에서 고객의요청에의해 실내인테리어
999997	G	47	474	영업점에서 일반소비자에게 여성의류 판매
999998	P	85	856	사업장에서 일반고객을대상으로 필라테스
999999	I	56	561	사업장에서 접객시설을 갖추고 한식(미역구)판매
1000000 rows × 4 columns
```
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
![image](https://user-images.githubusercontent.com/37128004/197760197-aa4ce5ba-d17d-4f1a-83ff-54caab4bea84.png)

### Parameter settings
- Adam optimizer(learning_rate : 0.001)
- Dropout : 0.5
- Activation function : ELU
- Epochs : 20
- batch_size : 512
- class_weight
- callbacks(model checkpoint, ReduceLROnPlateau, earlystop)

## Result
### train and validate curve
<a href="https://github.com/ohilikeit/Statistics_competition">
  <img align="center" src="https://user-images.githubusercontent.com/37128004/197760910-b2062678-28fb-49b0-a972-778e480cd262.png" width="500" height="400" />
</a>
<a href="https://github.com/anuraghazra/convoychat">
  <img align="center" src="https://user-images.githubusercontent.com/37128004/197760960-16d474dd-4b18-4f59-a5c2-972c4ceb2261.png" width="500" height="400" />
</a>

### score
- macro f1 score : 0.8586(private : 0.79)

## Conclusion
- 처음 참여해본 대회로 구현 자체도 어려움이 많았다. 
- NLP에서 일반적으로 사용하는 PLM이 아닌 CNN으로 접근하여 정확도나 학습 면에서 좋지 못했다. 
- 현업에선 항상 발생하는 class imbalance 문제를 어떻게 해결할 것인가에 대해 많은 고민을 했다.
- tensorflow와 keras를 이용해 딥러닝 모델을 짜는 기본적인 방법이나 text 데이터 전처리, tabulr data 전처리, overfitting 방지법 등 많은 것들을 얻었다. 
