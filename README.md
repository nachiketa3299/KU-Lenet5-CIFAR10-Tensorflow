# Lenet5 - Tensorflow
*2015410115 송재민*



### 제출 파일
- `lenet.py`: Lenet5 모델 구조가 선언되는 파일
- `lenet_eval.py`: Evaluation이 이루어지는 파일
- `lenet_train.py`: Train과 Validation이 이루어지는 파일
- `data_helpers.py`: Data Augmentation 등을 다루는 파일
- `preset.py`: 모델에 사용할 하이퍼파라미터들의 프리셋을 저장
- `tensorboard_run`: Tensorboard를 쉽게 실행하기 위해 작성한 파일
- `./runs`: 학습된 모델들이 저장되는 경로
- `./INFOS/log_eval.txt`: Evaluation 결과가 저장됨
- `./INFOS/train_result.txt`: Train 후 프리셋별 결과 요약되어 저장됨

## Notice
- Learning Rate Decay의 경우 Exponential Decay 방식을 사용. 


## Hyperparameter Tuning
우선 Setting 4까지 모두 훈련 시킨 후, 모든 튜닝 요소가 ON된 상태에서 가장 높은 성능을 보이는 Setting 4를 기준으로 하여 모델을 만들어 나가기로 했다. 

처음으로 Leaning Rate Decay Rate를 조정해 주었다. Setting 4에서 0.96이라는 값으로 시작하였는데, Learning Rate Decay 없이 학습이 진행된 Setting 3보다 근소하게 낮은 성능을 보였기 때문이다
