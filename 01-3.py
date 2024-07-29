import matplotlib.pyplot as plt

### 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

### 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 산점도로 나타내기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

### 특성 데이터 만들기 
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

### input 데이터 만들기 (2차원 리스트)
fish_data = [[l, w] for l, w in zip(length, weight)]
print(fish_data)

### target 데이터 만들기 (도미(35개)는 1, 빙어(14개)는 0)
fish_target = [1] * 35 + [0] * 14
print(fish_target)

from sklearn.neighbors import KNeighborsClassifier
# import sklearn
# model = sklearn.neighbors.KNeighborsClassifier() 과 동일

kn = KNeighborsClassifier()         # KNeighborsClassifier 클래스의 객체를 만듦
kn.fit(fish_data, fish_target)      # fit() 메서드는 주어진 데이터로 알고리즘을 훈련
accuracy = kn.score(fish_data, fish_target)    # score() 메서드는 훈련된 모델을 평가 (0~1 사이의 값 반환, 1은 정확도 100%)
print(accuracy)

a = kn.predict([[30, 600]])         # predict() 메서드는 새로운 데이터의 정답을 예측
print(a)

print(kn._fit_X)
print(kn._y)

kn49 = KNeighborsClassifier(n_neighbors=49) # 참고 데이터를 49개로 한 kn49 모델
kn49.fit(fish_data, fish_target)
accuracy = kn49.score(fish_data, fish_target)
print(accuracy)
print(35/49)

### 확인문제 4번
for n in range(5, 50):
    # k-최근접 이웃 개수 설정
    kn.n_neighbors = n
    # 점수 계산
    score = kn.score(fish_data, fish_target)
    # 100% 정확도에 미치지 못하는 이웃 개수 출력
    if score < 1:
        print(n, score)
        break