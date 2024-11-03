### 1. Cross Entropy Loss (교차 엔트로피 손실)
Cross Entropy Loss는 분류 문제에서 자주 사용하는 손실 함수로, 모델이 예측한 클래스 확률이 실제 클래스와 얼마나 다른지를 측정합니다. 예측이 실제 값에 가까울수록 Cross Entropy Loss 값이 작아집니다.

#### Cross Entropy Loss 수식
Cross Entropy Loss는 다음과 같이 계산됩니다:
$$
\text{Cross Entropy} = - \sum_{c=1}^{C} Y_{i,c} \log(\hat{Y}_{i,c})
$$
- $Y_{i,c}$: 실제 클래스 (1이면 정답 클래스, 0이면 나머지 클래스)
- $\hat{Y}_{i,c}$: 모델이 예측한 클래스 확률
- $C$: 클래스의 개수

#### 예시로 이해하기
클래스가 세 개인 분류 문제를 가정해보겠습니다. 예를 들어, 강아지, 고양이, 새 세 가지 클래스가 있고, 실제 레이블은 '강아지'입니다. 

모델이 예측한 확률이 다음과 같다면:
- 강아지: 0.7
- 고양이: 0.2
- 새: 0.1

Cross Entropy Loss를 계산하면:
$$
\text{Loss} = -\log(0.7) \approx 0.357
$$
여기서 정답 클래스인 강아지에 대한 확률만 고려합니다. 예측 확률이 1에 가까워질수록, 즉 100% 정확해질수록 Loss가 0에 가까워지며, 반대로 예측이 틀릴수록 값이 커집니다.

---

### 2. KL-Divergence (Kullback-Leibler Divergence)
KL-Divergence는 두 확률 분포 사이의 차이를 측정하는 방법으로, 예측 분포가 실제 분포에 얼마나 가까운지를 측정합니다. Cross Entropy와 유사하지만, 실제 분포를 기준으로 예측 분포와의 차이를 구하는 점에서 조금 다릅니다.

#### KL-Divergence 수식
KL-Divergence는 다음과 같이 계산됩니다:
$$
\text{KL-Divergence} = \sum_{c=1}^{C} Y_{i,c} \cdot \log \left(\frac{Y_{i,c}}{\hat{Y}_{i,c}}\right)
$$
이를 수식적으로 변형하면 두 항으로 나눌 수 있습니다:
$$
\text{KL-Divergence} = \sum_{c=1}^{C} Y_{i,c} \log(Y_{i,c}) - Y_{i,c} \log(\hat{Y}_{i,c})
$$
첫 번째 항은 실제 분포의 엔트로피 (Entropy)이며, 두 번째 항은 Cross Entropy입니다.

#### 기댓값 관점에서 해석하기
위 수식을 기댓값 관점에서 살펴보면 다음과 같이 표현할 수 있습니다:
$$
\text{KL-Divergence} = E_{Y}[\log(Y)] - E_{Y}[\log(\hat{Y})]
$$
이 해석에서 KL-Divergence는 **실제 분포가 예측 분포와 얼마나 다른지**를 나타냅니다.

#### 예시로 이해하기
예를 들어, '강아지'가 실제 정답인 상황에서 모델이 다음과 같은 확률을 예측했다고 해봅시다:
- 강아지: 0.8
- 고양이: 0.1
- 새: 0.1

이 경우 KL-Divergence는 다음과 같이 계산됩니다:
1. **Entropy** (실제 분포의 불확실성): 실제 분포에서 강아지의 확률은 1이므로 $-1 \times \log(1) = 0$.
2. **Cross Entropy** (모델의 예측 불확실성): 모델이 예측한 강아지 확률 0.8을 사용해 $-1 \times \log(0.8) \approx 0.22$.

따라서, KL-Divergence는 $0 - 0.22 = 0.22$로, 예측이 실제 값과 조금 차이가 나고 있다는 정보를 줍니다. 예측 확률이 더 정확해질수록 KL-Divergence 값은 점점 0에 가까워집니다.

---

### 3. Cross Entropy와 KL-Divergence의 관계
- Cross Entropy는 예측과 실제 값 간의 차이를 측정하며, 실제 값이 Hard Label(정확히 0과 1로 구분)일 때 KL-Divergence와 같은 결과를 가집니다.
- KL-Divergence는 실제 분포의 관점에서 예측 분포와의 차이를 측정하는데, 이 차이를 줄이는 것은 모델이 예측을 실제 값에 가깝게 만드는 것과 같습니다.

Cross Entropy는 손실 함수로 주로 사용되며, KL-Divergence는 두 분포 간의 차이를 수치화하여 모델의 학습에 도움을 줍니다.

### 4. Hard Label vs Soft Label
![[Pasted image 20241103220009.png]]

이 그래프는 **Cross Entropy Loss**와 **KL Divergence Loss**가 서로 어떻게 다른지, 그리고 **Hard Label**과 **Soft Label**일 때 각각의 손실 함수 값이 어떻게 변화하는지 보여주려는 것입니다. 이 그래프를 통해 두 손실 함수가 예측 확률에 따라 어떻게 변화하는지 시각적으로 비교할 수 있습니다. 자세히 설명해 볼게요.

**Hard Label vs Soft Label**

- **Hard Label**: 위쪽의 두 그래프는 Hard Label의 예시입니다. 예를 들어, 실제 값이 '0.01' 혹은 '0.99'로 설정된 경우로, Hard Label에서는 클래스 확률이 정확히 0 또는 1로 고정됩니다.
- **Soft Label**: 아래쪽 두 그래프는 Soft Label의 예시입니다. 예를 들어, 실제 값이 '0.3' 또는 '0.7'로 설정된 경우입니다. Soft Label에서는 확률이 0과 1 사이의 값으로, 불확실성이 더 큰 상태를 나타냅니다.

**Loss 함수 비교** 
각 그래프에는 세 가지 손실 함수가 그려져 있습니다.
- **Cross Entropy Loss** (파란색): 모델이 예측한 확률 분포와 실제 값(분포) 간의 차이를 측정합니다.
- **L1 Loss (MAE Loss)** (주황색): 예측값과 실제값 간의 차이의 절댓값을 평균한 것입니다. 분류 문제보다는 주로 회귀 문제에서 사용됩니다.
- **KL Divergence Loss** (녹색): 실제 분포와 예측 분포 간의 차이를 측정합니다. 예측 분포가 실제 분포에 가까워질수록 KL Divergence 값이 줄어듭니다.


**그래프 분석**
**Hard Label (위쪽 그래프)**
- **y true = 0.01**과 **y true = 0.99**로 설정된 경우입니다. 즉, 실제 값이 거의 0이거나 거의 1인 경우를 나타냅니다.
- 여기서 Cross Entropy Loss와 KL Divergence Loss는 예측 확률이 실제 확률에 가까워질수록 손실이 낮아지는 경향을 보여줍니다.
- Hard Label에서는 Cross Entropy Loss와 KL Divergence Loss가 거의 동일하게 작용하고 있음을 볼 수 있습니다. 이는 Hard Label에서는 두 손실 함수가 거의 동일하게 작동함을 의미합니다.

**Soft Label (아래쪽 그래프)**
- **y true = 0.3**과 **y true = 0.7**로 설정된 경우로, 실제 값이 0과 1 사이에 있는 Soft Label입니다.
- 이 경우 KL Divergence Loss와 Cross Entropy Loss가 Hard Label과는 다른 양상을 보입니다. KL Divergence는 Cross Entropy와 달리 **Negative Entropy** 개념이 추가되어, 예측 확률이 실제 값과 가까워지면 Loss가 좀 더 낮아지게 됩니다.
- **Negative Entropy**는 Soft Label에서 나타나는 특징으로, 불확실성이 있는 분포일 때 KL Divergence가 Cross Entropy보다 다르게 행동하게 합니다.

### 결론
이 그래프는 다음 두 가지를 설명하려고 합니다.
1. **Hard Label**일 경우 Cross Entropy Loss와 KL Divergence Loss가 거의 동일하게 작동한다는 점입니다.
2. **Soft Label**일 경우 KL Divergence Loss와 Cross Entropy Loss의 차이가 더 두드러지며, KL Divergence는 Negative Entropy로 인해 다르게 작동할 수 있다는 점입니다. 즉, 불확실성이 큰 Soft Label에서는 KL Divergence가 Cross Entropy와 달리 더 정밀한 정보를 제공할 수 있습니다.

따라서, Soft Label 상황에서는 KL Divergence가 실제 분포와의 차이를 더 잘 반영할 수 있는 지표로 사용될 수 있으며, 이때 Negative Entropy가 중요한 역할을 합니다.

---

### 요약
- **Cross Entropy Loss**: 실제 클래스와 예측 확률 간의 차이를 측정하는 손실 함수로, 주로 분류 문제에서 사용됩니다.
- **KL-Divergence**: 두 확률 분포 간의 차이를 측정하여 예측이 실제 분포에 가까워지도록 유도하는 지표입니다.
- **관계**: Cross Entropy Loss는 KL-Divergence와 같은 개념을 포함하며, Hard Label 상황에서는 두 값이 동일하게 작동합니다.