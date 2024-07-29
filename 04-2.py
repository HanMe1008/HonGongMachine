### 04-2 확률적 경사 하강법

## SGDClassifier

# fish_csv_data 파일에서 판다스 데이터프레임 만들기
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

# Species 열은 타깃 데이터, 나머지 5개 열은 입력 데이터로 사용 
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 사이킷런의 train_test_split() 함수로 훈련 세트와 데이터 세트로 나눔
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 훈련 세트와 테스트 세트의 특성을 표전화 전처리 (훈련 세트에서 학습한 통계 값으로 테스트 세트를 변환해야함)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 사이킷런에서 확률적 경사 하강법을 제공하는 대표적인 분류용 클래스 SGDClassifier
from sklearn.linear_model import SGDClassifier

# SGDClassifier 객체를 만들 때 2개의 매개변수 지정 (loss:손실 함수의 종류, max_iter:수행할 에포크 횟수)
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 확률적 경사 하강법은 점진적 학습이 가능, 모델을 이어서 훈련할 때는 partial_fit() 메서드 사용
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

## 에포크와 과대/과소적합
import numpy as np
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

# 300번의 에포크 동안 훈련 반복
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# 300번의 에포크 동안 기록한 훈련 세트와 테스트 세트의 점수 그래프
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# SGDClassifier의 반복 횟수를 100에 맞추고 모델을 다시 훈련, 최종적으로 점수 출력
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 손실 함수로 힌지 손실(hinge loss)==서포트 벡터 머신(support vector machine)을 사용
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


