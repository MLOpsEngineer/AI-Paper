# Differential Transformer

Tianzhu Ye\*†‡, Li Dong\*†, Yuqing Xia\*†, Yutao Sun\*†‡  
Yi Zhu†, Gao Huang‡, Furu Wei†⋄  
†Microsoft Research, ‡Tsinghua University  
[https://aka.ms/GeneralAI](https://aka.ms/GeneralAI)

## 초록

Transformer는 관련 없는 컨텍스트에 과도한 주의를 할당하는 경향이 있습니다. 본 연구에서는 노이즈를 제거하면서 관련 컨텍스트에 대한 주의를 증폭시키는 *DIFF Transformer*를 소개합니다. 구체적으로, *Differential Attention Mechanism*은 두 개의 개별적인 Softmax Attention 맵의 차이로 Attention 점수를 계산합니다. 이 차이는 노이즈를 제거하여 스파스한 Attention 패턴의 출현을 촉진합니다. 언어 모델링에 대한 실험 결과, DIFF Transformer는 모델 크기를 확장하고 학습 토큰을 증가시키는 다양한 설정에서 Transformer를 능가함을 보여줍니다. 더욱 흥미롭게도, DIFF Transformer는 긴 컨텍스트 모델링, 핵심 정보 검색, 환각 완화, In-Context Learning, 활성화 이상치 감소와 같은 실제 응용에서 주목할 만한 이점을 제공합니다. 관련 없는 컨텍스트에 덜 산만해짐으로써, DIFF Transformer는 질문 응답 및 텍스트 요약에서 환각을 완화할 수 있습니다. In-Context Learning의 경우, DIFF Transformer는 정확성을 향상시킬 뿐만 아니라, 만성적인 견고성 문제로 여겨졌던 순서 치환에 대해서도 더 견고합니다. 이러한 결과는 DIFF Transformer가 대형 언어 모델을 발전시키기 위한 매우 효과적이고 유망한 아키텍처임을 보여줍니다.

## 1. 서론

Transformer[41]는 최근 몇 년간 상당한 연구 관심을 끌었으며, Decoder-Only Transformer는 대형 언어 모델(LLMs)의 사실상 표준으로 자리잡았습니다. Transformer의 핵심은 Attention 메커니즘으로, 이는 Softmax 함수를 사용하여 시퀀스 내 다양한 토큰의 중요성을 가중합니다. 하지만 최근 연구[17,23]에 따르면, LLMs는 컨텍스트에서 핵심 정보를 정확히 검색하는 데 어려움을 겪고 있습니다.

그림 1의 왼쪽에서 보이는 것처럼, 우리는 Transformer가 컨텍스트의 서로 다른 부분에 할당한 Normalized Attention Scores를 시각화합니다. 이 작업은 문서 더미 속에 내포된 답변을 검색하는 것입니다. 시각화는 Transformer가 올바른 답변에 대해 적은 비율의 Attention 점수를 할당하면서 불균형적으로 관련 없는 컨텍스트에 집중하는 경향이 있음을 보여줍니다. 섹션 3의 실험은 Transformer가 이러한 기능에서 고군분투한다는 것을 더욱 입증합니다. 이 문제는 결국 올바른 답변을 압도하는 관련 없는 컨텍스트에 할당된 무시할 수 없는 Attention 점수에서 발생합니다. 우리는 이러한 불필요한 점수를 Attention Noise라고 부릅니다.

**그림 1**: Transformer는 종종 관련 없는 컨텍스트(즉, Attention Noise)에 과도하게 주의를 기울입니다. DIFF Transformer는 답변 범위에 대한 주의를 증폭시키고 노이즈를 제거하여 컨텍스트 모델링 능력을 향상시킵니다.


# Differential Transformer: Large Language Models를 위한 기초 아키텍처

이 논문에서는 Differential Transformer(일명 DIFF Transformer)라는 대형 언어 모델을 위한 기초 아키텍처를 소개합니다. Differential Attention 메커니즘은 차별적 잡음 제거를 통해 Attention 노이즈를 상쇄하는 방법을 제안합니다. 구체적으로, 우리는 쿼리 및 키 벡터를 두 그룹으로 나누고 두 개의 개별 Softmax Attention 맵을 계산합니다. 그런 다음, 이 두 맵의 차이를 Attention 점수로 간주합니다. Differential Attention 메커니즘은 Attention 노이즈를 제거하여 모델이 중요한 정보에 집중하도록 유도합니다. 이 접근 방식은 전기공학에서 두 신호 간의 차이가 공통 모드 노이즈를 상쇄하는 노이즈 캔슬링 헤드폰 및 차동 증폭기와 유사합니다. 그림 1의 중앙에서는 DIFF Transformer의 Attention 점수에 대한 정규화된 분포를 제시합니다. DIFF Transformer는 Transformer에 비해 정답에는 훨씬 높은 점수를, 관련이 없는 맥락에는 훨씬 낮은 점수를 할당하는 것을 관찰할 수 있습니다. 그림 1의 오른쪽에서는 제안된 방법이 검색 능력에서 눈에 띄는 향상을 이루었다는 것을 보여줍니다.

우리는 언어 모델링에 대한 광범위한 실험을 수행했습니다. DIFF Transformer를 매개변수 수, 학습 토큰, 맥락 길이 측면에서 확장했습니다. 확장 곡선은 DIFF Transformer가 Transformer가 필요한 모델 크기 또는 학습 토큰의 약 65%만으로도 유사한 언어 모델링 성능을 달성할 수 있음을 나타냅니다. 더욱이, DIFF Transformer는 다양한 다운스트림 작업에서 Transformer를 능가합니다. 긴 시퀀스 평가에서도 DIFF Transformer가 증가하는 맥락을 활용하는 데 매우 효과적임을 보여줍니다. 추가로, 실험 결과는 DIFF Transformer가 대형 언어 모델에 대해 흥미로운 장점을 가지고 있음을 입증합니다. 예를 들어, 제안된 방법은 주요 정보 검색, 환각 완화, 맥락 내 학습에서 Transformer를 상당히 능가합니다. DIFF Transformer는 모델 활성화에서 이상치를 줄여 양자화에 대한 새로운 기회를 제공합니다. 이 발견은 DIFF Transformer가 대형 언어 모델을 위한 효과적이고 독특한 기초 아키텍처로 자리 잡게 합니다.

## 2. Differential Transformer

우리는 대형 언어 모델(LLM)과 같은 시퀀스 모델링을 위한 기초 아키텍처로 Differential Transformer(일명 DIFF Transformer)를 제안합니다. 디코더 전용 모델을 예로 들어 아키텍처를 설명합니다. 모델은 L개의 DIFF Transformer 레이어로 스택됩니다. 입력 시퀀스 $x = x_1 \cdots x_N$가 주어지면, 입력 임베딩을 $X_0 = [x_1, \cdots, x_N] \in \mathbb{R}^{N \times d_{model}}$로 패킹합니다. 여기서 $d_{model}$은 모델의 히든 차원을 나타냅니다. 입력은 추가로 맥락화되어 출력 $X_L$, 즉 $X_l = \text{Decoder}(X_{l-1}), l \in [1, L]$을 얻습니다. 각 레이어는 Differential Attention 모듈과 그 다음의 Feed-Forward 네트워크 모듈로 구성됩니다. Transformer [41]과 비교했을 때, 주요 차이점은 기존 Softmax Attention을 Differential Attention으로 대체한 것이며, 매크로 레이아웃은 동일하게 유지됩니다. 우리는 또한 LLaMA [38]를 따르는 개선 사항으로 Pre-RMSNorm [46]과 SwiGLU [35, 29]를 채택합니다.

### 2.1 Differential Attention

Differential Attention 메커니즘은 쿼리, 키, 값 벡터를 출력으로 매핑합니다. 우리는 쿼리와 키 벡터를 사용하여 Attention 점수를 계산한 다음 값 벡터의 가중합을 계산합니다. 중요한 설계는 Attention 점수의 노이즈를 상쇄하기 위해 한 쌍의 Softmax 함수를 사용하는 것입니다. 구체적으로, 입력 $X \in \mathbb{R}^{N \times d_{model}}$가 주어지면, 이를 먼저 쿼리, 키, 값 벡터로 투영합니다: $Q_1, Q_2, K_1, K_2 \in \mathbb{R}^{N \times d}, V \in \mathbb{R}^{N \times 2d}$. 그런 다음 Differential Attention 연산자인 $\text{DiffAttn}(\cdot)$이 다음과 같이 출력을 계산합니다:

$$
[Q_1; Q_2] = XW_Q, \quad [K_1; K_2] = XW_K, \quad V = XW_V
$$

$$
\text{DiffAttn}(X) = \left(\text{softmax}\left(\frac{Q_1K_1^T}{\sqrt{d}}\right) - \lambda \text{softmax}\left(\frac{Q_2K_2^T}{\sqrt{d}}\right)\right)V
$$

여기서 $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times 2d}$는 매개변수이며, $\lambda$는 학습 가능한 스칼라입니다. 학습 동작을 동기화하기 위해, 우리는 스칼라 $\lambda$를 다음과 같이 재매개변수화합니다:

$$
\lambda = \exp(\lambda_{q1} \cdot \lambda_{k1}) - \exp(\lambda_{q2} \cdot \lambda_{k2}) + \lambda_{init}
$$

여기서 $\lambda_{q1}, \lambda_{k1}, \lambda_{q2}, \lambda_{k2} \in \mathbb{R}^d$는 학습 가능한 벡터이고, $\lambda_{init} \in (0, 1)$는 $\lambda$의 초기화를 위한 상수입니다. 우리는 $\lambda_{init} = 0.8 - 0.6 \times \exp(-0.3 \cdot (l-1))$ 설정이 실전에서 잘 작동함을 경험적으로 발견했으며, $l \in [1, L]$은 레이어 인덱스를 나타냅니다. 이는 기본 전략으로 사용됩니다.


### Linear
```python
def DiffAttn(X, W_q, W_k, W_v, λ):
    Q1, Q2 = split(X @ W_q)
    K1, K2 = split(X @ W_k)
    V = X @ W_v
    s = 1 / sqrt(d)
    A1 = Q1 @ K1.transpose(−1, −2) * s
    A2 = Q2 @ K2.transpose(−1, −2) * s
    return (softmax(A1) − λ * softmax(A2)) @ V
```

### MultiHead
```python
def MultiHead(X, W_q, W_k, W_v, W_o, λ):
    O = GroupNorm([DiffAttn(X, W_qi, W_ki, W_vi, λ) for i in range(h)])
    O = O * (1 − λ_init)
    return Concat(O) @ W_o
```

그림 2: Multi-head differential attention. 각 헤드는 두 개의 softmax attention map의 차이를 사용하여 attention 노이즈를 상쇄합니다. λ는 학습 가능한 스칼라로, λ_init으로 초기화됩니다. GroupNorm은 각 헤드에 독립적으로 정규화를 적용합니다. 고정된 곱셈 값 (1−λ_init)은 GroupNorm 이후에 사용되며, 이는 Transformer와의 그래디언트 흐름을 정렬합니다. 코드 구현은 [https://aka.ms/Diff-Transformer](https://aka.ms/Diff-Transformer)에서 확인할 수 있습니다.

실험에서는 모든 레이어에 동일한 λ (예: 0.8)를 사용하는 초기화 전략도 탐구합니다. 절편 연구(Section 3.8)에서 보여지듯이, 성능은 다양한 초기화 전략에 비교적 강건합니다.

Differential attention은 두 개의 softmax attention 함수의 차이를 취하여 attention 노이즈를 제거합니다. 이 아이디어는 전기공학에서 제안된 differential amplifiers [19]와 유사하여, 두 신호의 차이를 출력으로 사용함으로써 입력의 공통 모드 노이즈를 제거합니다. 또한, 노이즈 캔슬링 헤드폰의 설계도 유사한 아이디어에 기반합니다. 우리는 Appendix A에 설명된 대로 FlashAttention [8]을 직접 재사용할 수 있으며, 이는 모델 효율성을 크게 향상시킵니다.

### Multi-Head Differential Attention

Differential Transformer에서는 multi-head 메커니즘 [41]도 사용합니다. h는 attention 헤드의 수를 나타냅니다. 우리는 각 헤드에 대해 서로 다른 projection matrix인 $W_Q, W_K, W_V, i \in [1, h]$를 사용합니다. 스칼라 λ는 동일한 레이어 내의 헤드 간에 공유됩니다. 그런 다음 헤드 출력은 다음과 같이 정규화되고 최종 결과로 투영됩니다:

$$
\text{head}_i = \text{DiffAttn}(X; W_{Qi}, W_{Ki}, W_{Vi}, λ)
$$

$$
\text{head}_i = (1−λ_{\text{init}}) \cdot \text{LN}(\text{head}_i) \tag{3}
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \cdots, \text{head}_h) W_O
$$

여기서 $λ_{\text{init}}$은 방정식 (2)의 상수 스칼라이며, $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$은 학습 가능한 projection matrix입니다. $\text{LN}(·)$은 각 헤드에 대해 RMSNorm [46]을 사용하고, $\text{Concat}(·)$은 채널 차원을 따라 헤드를 함께 결합합니다. 우리는 고정된 곱셈 값 $(1−λ_{\text{init}})$을 $\text{LN}(·)$의 스케일로 사용하여 Transformer와 그래디언트를 정렬합니다. Appendix F는 전체 그래디언트 흐름이 Transformer와 유사하게 유지됨을 증명합니다. 이 뛰어난 특성 덕분에 유사한 하이퍼파라미터를 직접 상속받고 훈련 안정성을 보장할 수 있습니다. 우리는 헤드 수를 $h = d_{\text{model}} / 2d$로 설정하며, 여기서 $d$는 Transformer의 헤드 차원과 동일합니다. 따라서 파라미터 수와 계산 복잡성을 정렬할 수 있습니다.

### Headwise Normalization

그림 2는 GroupNorm(·) [44]을 사용하여 LN(·)이 각 헤드에 독립적으로 적용되었음을 강조합니다. Differential attention은 더 희소한 패턴을 가지는 경향이 있으므로 통계 정보가 헤드 간에 더 다양합니다. LN(·) 연산자는 각 헤드를 결합하기 전에 정규화하여 그래디언트 통계를 개선합니다 [43, 28].

## 2.2 전체 아키텍처

전체 아키텍처는 $L$개의 레이어를 쌓아 구성되며, 각 레이어는 멀티-헤드 차별적 어텐션 모듈과 피드-포워드 네트워크 모듈을 포함합니다. 차별적 Transformer 레이어는 다음과 같이 설명됩니다:

$$
Y_l = \text{MultiHead}(\text{LN}(X_l)) + X_l
$$

$$
X_{l+1} = \text{SwiGLU}(\text{LN}(Y_l)) + Y_l
$$

여기서 $\text{LN}(·)$은 RMSNorm [46]이고, $\text{SwiGLU}(X) = (\text{swish}(XW_1) \odot XW_2)W_3$이며, $W_1, W_2 \in \mathbb{R}^{d_{\text{model}} \times \frac{8}{3}d_{\text{model}}}, W_3 \in \mathbb{R}^{\frac{8}{3}d_{\text{model}} \times d_{\text{model}}}$는 학습 가능한 행렬입니다.

## 3 실험

우리는 차별적 Transformer를 대형 언어 모델에 대해 다음과 같은 관점에서 평가합니다. 첫째, 다양한 다운스트림 작업(섹션 3.1)에서 제안된 아키텍처와 Transformer를 비교하고 모델 크기와 학습 토큰을 확장하는 특성을 연구합니다(섹션 3.2). 둘째, 길이를 64K로 확장하고 긴 시퀀스 모델링 능력을 평가합니다(섹션 3.3). 셋째, 핵심 정보 검색, 맥락적 환상 평가, 맥락 내 학습의 결과를 제시합니다(섹션 3.4–3.6). 넷째, 차별적 Transformer가 Transformer에 비해 모델 활성화에서 이상치를 줄일 수 있음을 보여줍니다(섹션 3.7). 다섯째, 다양한 설계 선택에 대한 광범위한 ablation 연구를 수행합니다(섹션 3.8).

### 3.1 언어 모델링 평가

우리는 1T 토큰으로 3B 크기의 DIFF Transformer 언어 모델을 훈련하고, 다양한 다운스트림 작업에서 이전에 잘 훈련된 Transformer 기반 모델 [13,39,40]과 비교합니다. 부록 B에 설명된 대로, 우리는 350B 토큰으로 3B 크기의 Transformer 언어 모델을 동일한 설정으로 훈련합니다. 체크포인트는 공정한 비교를 보장하기 위해 다음 실험 및 분석에서도 사용됩니다.

#### 설정

우리는 StableLM-3B-4E1T [40]와 유사한 레시피를 따릅니다. 숨겨진 크기는 3072로 설정합니다. 레이어의 수는 28입니다. 헤드 차원 d는 128입니다. Transformer의 헤드 수는 24이고, DIFF Transformer의 헤드 수는 12로, 계산 FLOPs와 모델 크기를 맞춥니다. 총 매개변수 수는 약 2.8B입니다. 학습 시퀀스 길이는 4096입니다. 배치 크기는 4M 토큰입니다. 우리는 1T 토큰으로 모델을 훈련합니다. 우리는 AdamW [24] 옵티마이저를 사용하며, $\beta = 0.9, 0.95$입니다. 최대 학습률은 $3.2 \times 10^{-4}$이며, 1000단계의 워밍업 후 선형으로 감소하여 $1.28 \times 10^{-5}$까지 내려갑니다. 훈련 코퍼스는 StableLM-3B-4E1T [40]을 따릅니다. 우리는 tiktoken-cl100k_base 토크나이저를 사용합니다. 자세한 하이퍼파라미터는 부록 C에 제공됩니다.

#### 결과

표 1은 LMEvalHarness 벤치마크 [12]에서의 제로-샷 결과를 보고합니다. 우리는 DIFF Transformer를 OpenLLaMA-v2-3B [13], StableLM-base-alpha-3B-v2 [39], StableLM-3B-4E1T [40]을 포함한 잘 훈련된 Transformer 기반 언어 모델과 비교합니다. OpenLLaMA-v2-3B와 StableLM-base-alpha-3B-v2도 1T 토큰으로 훈련되었습니다. StableLM-3B-4E1T의 1T 결과는 기술 보고서 [40]에서 가져왔습니다. 실험 결과, DIFF Transformer는 이전의 잘 튜닝된 Transformer 언어 모델에 비해 유리한 성능을 달성함을 보여줍니다. 추가로, 부록 B는 DIFF Transformer가 동일한 설정으로 3B 크기 언어 모델을 훈련하여 다양한 작업에서 Transformer를 능가함을 보여줍니다.

표 1: EvalHarness [12] 정확도, 잘 훈련된 Transformer 언어 모델과 비교 [40, 39, 13]. 우리는 3B 모델을 1조 개의 학습 토큰으로 확장했습니다. StableLM-3B-4E1T의 1T 결과는 기술 보고서 [40]에서 가져왔습니다.


3.15  
3.10  
3.05  
3.00  
2.95  
2.90  
100 101  
#Parameters (B) (log scale)  
ssoL  
Transformer  
2.9  
Diff (Ours)  
2.8  
2.7  
2.6  
38% Fewer Params 2.5  
26 27 28 29  
#Tokens (B) (log scale)  
(a) Scaling model size ranging from 830M to 13B.  
ssoL  
Transformer  
Diff (Ours)  
36% Fewer Tokens  
(b) Scaling number of training tokens for 3B models.  
그림 3: 매개변수 수와 학습 토큰 수 증가에 따른 언어 모델링 손실. DIFF Transformer는 Transformer의 성능을 맞추기 위해 모델 크기나 학습 토큰 수의 약 65%만 필요로 합니다.

### 3.2 Transformer와의 확장성 비교
우리는 언어 모델링에서 DIFF Transformer와 Transformer의 확장성을 비교합니다. 모델 크기와 학습 토큰 수를 각각 확장합니다. LLaMA [38]에서와 같이 보강된 Transformer 아키텍처를 따르고 있으며, 공정한 비교를 위해 동일한 설정을 사용합니다. 구체적으로, "Transformer" 모델에는 RMSNorm [46], SwiGLU [35, 29]의 개선 및 바이어스 제거가 포함되어 있습니다.

**모델 크기 확장**  
그림 3a에서 보이듯이, 우리는 830M, 1.4B, 2.8B, 6.8B, 13.1B 매개변수를 가진 언어 모델을 학습시킵니다. 모델은 시퀀스 길이 2048과 배치 크기 0.25M 토큰으로 학습됩니다. 우리는 모델을 40K 스텝 동안 학습합니다. 자세한 하이퍼파라미터는 부록 D에 설명되어 있습니다. 이 구성에서 확장 법칙[18]이 경험적으로 잘 맞습니다. 그림 3a는 DIFF Transformer가 다양한 모델 크기에서 Transformer를 능가함을 보여줍니다. 결과는 DIFF Transformer가 매개변수 수의 측면에서 확장 가능하다는 것을 나타냅니다. 맞춰진 곡선에 따르면, 6.8B 크기의 DIFF Transformer는 11B 크기의 Transformer와 비슷한 검증 손실을 달성하며, 매개변수는 62.2%만 필요합니다. 유사하게, 7.8B 크기의 DIFF Transformer는 13.1B 크기의 Transformer 성능을 맞추며, 매개변수는 59.5%만 필요합니다.

**학습 토큰 확장**  
그림 3b에서 보이듯이, 우리는 부록 B에 제시된 대로 3B 언어 모델을 360B 토큰(즉, 90K 스텝)까지 매 40B 토큰(즉, 10K 스텝)마다 평가합니다. 맞춰진 곡선은 160B 토큰으로 학습된 DIFF Transformer가 251B 토큰으로 학습된 Transformer와 비슷한 성능을 달성하며, 학습 토큰은 63.7%만 소비함을 나타냅니다.

### 3.3 긴 컨텍스트 평가

100 1K 10K 100K  
Sequence Position  
doohilekiL-goL  
evitageN  
우리는 부록 B에 설명된 3B 크기의 언어 모델을 64K 컨텍스트 길이로 확장합니다. 우리는 3B 체크포인트를 추가로 1.5B 토큰으로 계속 학습합니다. 대부분의 하이퍼파라미터는 섹션 3.1과 동일하게 유지됩니다. 학습률은 8e-5입니다. RoPE[36] θ는 640,000으로 증가됩니다. 학습 말뭉치는 시퀀스 길이에 따라 업 샘플링됩니다[11].

**결과**  
그림 4는 다양한 위치에서 토큰의 누적 평균 음의 로그 가능성(NLL)을 제공합니다[32], 여기서 낮은 NLL은 더 나은 성능을 나타냅니다. 평가는 64K 길이 내의 책 데이터에서 수행됩니다. 우리는 컨텍스트 길이가 증가함에 따라 NLL이 일관되게 감소하는 것을 관찰합니다. DIFF Transformer는 Transformer보다 낮은 NLL 값을 달성합니다. 결과는 DIFF Transformer가 증가하는 컨텍스트를 효과적으로 활용할 수 있음을 보여줍니다.

3.4 중요 정보 검색

Needle-In-A-Haystack [17] 테스트는 넓은 문맥에서 중요한 정보를 추출하는 능력을 평가하는 데 널리 사용됩니다. 우리는 LWM [22] 및 Gemini1.5 [32]의 multi-needle 평가 프로토콜을 따릅니다. 바늘은 다양한 길이의 문맥 내에서 여러 깊이에 삽입됩니다. 각 바늘은 특정 도시에 고유한 매직 넘버를 할당하는 간결한 문장으로 구성됩니다. 목표는 쿼리 도시와 일치하는 매직 넘버를 검색하는 것입니다. 우리는 정답 바늘을 문맥 내 다섯 가지 깊이, 즉 0%, 25%, 50%, 75%, 100%에 배치하고, 다른 혼란스러운 바늘은 무작위로 배치합니다. 각 깊이와 길이 조합은 50개의 샘플을 사용하여 평가됩니다. 평균 정확도가 보고됩니다. 여기서 N은 총 숫자-도시 쌍의 수를 나타내고, R은 쿼리 도시의 수를 나타냅니다.

4K 문맥 길이에서의 검색

표 2에서 보여주듯이, 우리는 4K 길이의 문맥에 N = 1, 2, 4, 6개의 바늘을 삽입하고 R = 1, 2개의 바늘을 검색합니다. 우리는 4K 입력 길이로 훈련된 3B 크기 모델을 평가합니다 (부록 B). 우리는 두 모델 모두 N = 1 및 N = 2에 대해서는 좋은 정확도를 얻는 것을 발견하였습니다. N과 R이 증가함에 따라, DIFFTransformer는 일관된 정확도를 유지하는 반면, Transformer의 성능은 크게 저하됩니다. 특히, N = 6, R = 2에서 두 모델 간의 정확도 차이는 30%에 이릅니다. 이 결과는 DIFFTransformer가 혼란스러운 문맥에서 중요한 정보를 검색하는 데 뛰어난 능력을 가지고 있음을 나타냅니다.

64K 문맥 길이에서의 검색

그림 5에서 보여주듯이, 평가된 문맥 길이는 N = 8, R = 1 설정에서 8K에서 64K까지 범위가 있습니다. 우리는 길이 확장을 가진 3B 크기 모델을 평가합니다 (섹션 3.3). 우리는 다양한 정답 바늘 깊이(y축)와 문맥 길이(x축)에 걸쳐 정확도를 보고합니다. 하단 행은 모든 깊이에 대한 평균 정확도입니다. DIFFTransformer는 다양한 문맥 길이에 걸쳐 안정적인 성능을 유지합니다. 반면, Transformer의 평균 정확도는 문맥 길이가 최대 길이인 64K까지 증가함에 따라 점진적으로 감소합니다. 게다가, DIFFTransformer는 특히 중요한 정보가 문맥의 첫 번째 절반(즉, 0%, 25%, 50% 깊이)에 위치할 때 Transformer보다 우수한 성능을 보입니다. 특히, 바늘이 64K 문맥의 25% 깊이에 배치될 때, DIFFTransformer는 Transformer에 비해 76%의 정확도 향상을 달성합니다.

Attention Score 분석

표 3은 중요한 정보 검색 작업을 위해 정답 범위와 노이즈 문맥에 할당된 Attention Score를 제시합니다. 이 점수는 유용한 정보를 Attention Noise로부터 보존하는 모델의 능력을 나타냅니다. 우리는 문맥 내 서로 다른 위치(즉, 깊이)에 삽입된 중요한 정보를 비교할 때의 정규화된 Attention Score를 비교합니다. Transformer와 비교했을 때, DIFFTransformer는 정답 범위에 더 높은 Attention Score를 할당하고, Attention Noise는 낮습니다.


## AttentiontoAnswer↑ AttentionNoise↓

### 모델

| Model       | 0%  | 25% | 50% | 75% | 100% |
|-------------|-----|-----|-----|-----|------|
| Transformer | 0.03 | 0.03 | 0.03 | 0.07 | 0.09 |
| DIFF        | 0.27 | 0.30 | 0.31 | 0.32 | 0.40 |

| Model       | 0%  | 25% | 50% | 75% | 100% |
|-------------|-----|-----|-----|-----|------|
| Transformer | 0.51 | 0.54 | 0.52 | 0.49 | 0.49 |
| DIFF        | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 |

표 3: 주요 정보 검색 작업에서 답변 영역과 노이즈 컨텍스트에 할당된 Attention 점수. 대상 답변은 컨텍스트의 다양한 위치(즉, 깊이)에 삽입됩니다. DIFF Transformer는 유용한 정보에 더 많은 Attention 점수를 할당하고 Attention 노이즈를 효과적으로 제거합니다.

### 3.5 In-Context Learning

우리는 In-Context Learning을 다중 샷 분류와 In-Context Learning의 강건성이라는 두 가지 관점에서 평가합니다. In-Context Learning은 언어 모델의 기본적인 능력으로, 모델이 입력 컨텍스트를 얼마나 잘 활용할 수 있는지를 나타냅니다.

**Many-Shot In-Context Learning**

그림 6에서 보이듯이, 우리는 Transformer와 우리의 아키텍처 간의 다중 샷 분류 정확도를 비교합니다. 우리는 64K 입력 길이를 지원하는 3B 크기의 언어 모델을 평가합니다(섹션 3.3). 우리는 [3]의 평가 프로토콜을 따르고, 제한된 디코딩을 사용합니다[30]. 우리는 1-shot에서 시작하여 총 길이가 64K에 도달할 때까지 데모 샘플의 수를 점진적으로 증가시킵니다. 구체적으로, TREC[15] 데이터셋은 6개의 클래스를 가지고 있으며, TREC-fine[15]은 50개의 클래스를, Banking-77[5]은 77개의 클래스를, Clinic-150[20]은 150개의 클래스를 가지고 있습니다. 결과는 DIFF Transformer가 데이터셋과 데모 샘플의 수에 관계없이 일관되게 Transformer보다 우수함을 보여줍니다. 또한, 평균 정확도의 향상은 5.2%에서 21.6%까지 상당합니다.

**그림 6**: 네 개의 데이터셋에 대한 다중 샷 In-Context Learning 정확도. 데모 예제가 1-shot에서 시작하여 전체 길이가 64K 토큰에 도달할 때까지 증가합니다. 점선은 성능이 안정화된 후의 평균 정확도를 나타냅니다.

그림 7: TREC 데이터셋에 대한 in-context learning의 강건성 평가. 정확도는 예시 순서를 무작위로 변경하며 랜덤 시드를 조정하여 평가합니다. 점선은 최상의 결과와 최악의 결과 사이의 마진을 나타냅니다. 작은 마진은 우수한 강건성을 의미합니다. 두 가지 프롬프트 형식이 검토되었습니다.

In-Context Learning의 강건성 그림 7은 Transformer와 DIFF Transformer 간의 in-context learning 강건성을 비교합니다. 동일한 예시가 주어졌을 때, 순서 변경에 따른 성능 변동성을 분석합니다. 낮은 변동성은 더 높은 강건성과 성능의 치명적인 저하 위험이 적음을 나타냅니다. 평가 프로토콜은 위와 동일합니다. 그림 7은 TREC 데이터셋에 대한 분석을 나타냅니다. 더 많은 결과는 부록 E에서 제공됩니다. 우리는 두 가지 프롬프트 형식을 평가합니다. 즉, 예시가 무작위로 배열된 경우(그림 7a)와 클래스별로 번갈아 배열된 경우(그림 7b)입니다. 두 설정 모두에서 DIFF Transformer는 Transformer에 비해 성능 변동성이 훨씬 작습니다. 결과는 우리의 접근 방식이 in-context learning에 대해 더 강건함을 나타냅니다. 반면, Transformer는 순서 변경에 의해 주의를 산만하게 만들기 쉬워 최상의 결과와 최악의 결과 사이에 큰 마진이 발생합니다[25].

3.6 Contextual Hallucination 평가

우리는 3B 크기의 언어 모델(부록 B에 설명됨)의 텍스트 요약 및 질문 응답에 대한 contextual hallucination을 평가합니다. 입력 컨텍스트가 올바른 사실을 포함하지만 모델이 여전히 정확한 출력을 생성하지 못하는 경우에 초점을 맞춥니다.

우리는 [6]의 평가 프로토콜을 따릅니다. 우리는 모델 출력을 실제 정답과 함께 GPT-4o[27]에 입력합니다. 그런 다음 GPT-4o에 모델 출력이 정확하고 환각이 없는지에 대한 이진 판단을 요청합니다. 이전 연구[6,31]는 위의 환각 평가 프로토콜이 GPT-4o 판단과 인간 주석 간의 높은 일치를 보인다고 했습니다. 자동 메트릭은 신뢰할 수 있으며 인간 평가를 반영합니다. 각 데이터셋에 대해 정확도는 100개의 샘플에 대해 평균화됩니다.

표 4a는 요약 데이터셋 XSum[26], CNN/DM[33], MultiNews[10]에 대한 환각 평가를 나타냅니다. 이 작업의 목표는 입력 문서에 대한 요약을 생성하는 것입니다.

표 4: 텍스트 요약 및 질문 응답에 대한 contextual hallucination 평가. 높은 정확도는 환각이 적음을 나타냅니다. 우리는 Chuang et al.[6]의 방식을 따라 GPT-4o를 사용하여 이진 판단을 내리며, 이는 인간 주석과 비교적 높은 일치를 보입니다.

## 모델 활성화 유형 및 통계

표 5: Attention logits 및 hidden states에서 가장 큰 활성화 값. 상위 활성화 값은 그들의 중간값에 비해 현저히 높은 크기를 가지므로 활성화 이상치로 간주됩니다. DIFF Transformer는 Transformer에 비해 이상치를 완화합니다.

## 질문 응답

표 4b에 나타난 것처럼, 우리는 DIFF Transformer와 Transformer의 단일 문서 및 다중 문서 질문 응답에서의 환각률을 비교합니다. Qasper [9] 데이터셋은 단일 문서 질문 응답입니다. 반면, HotpotQA [45]와 2WikiMultihopQA [14]는 다중 문서 질문 응답입니다. 목표는 주어진 컨텍스트에 대한 질문에 답하는 것입니다. 모든 평가 예제는 LongBench [2]에서 가져왔습니다.

Transformer와 비교했을 때, 우리의 방법은 요약 및 질문 응답에서의 맥락적 환각을 완화합니다. 성능 향상은 DIFF Transformer가 작업에 필요한 필수 정보에 더 잘 집중하는 것에서 비롯될 수 있습니다. 이는 Transformer에서의 맥락적 환각의 주요 원인이 attention 점수의 잘못된 할당이라는 이전 관찰[16]과 일치합니다.

## 3.7 활성화 이상치 분석

대형 언어 모델에서는 활성화의 일부분이 대다수에 비해 현저히 큰 값으로 나타나는데, 이는 일반적으로 활성화 이상치라고 불리는 현상입니다 [4, 37]. 이상치는 훈련 및 추론 중 모델의 양자화를 어렵게 만듭니다. 우리는 DIFF Transformer가 활성화 이상치의 크기를 줄일 수 있으며, 잠재적으로 양자화를 위한 낮은 비트 폭을 허용할 수 있음을 보여줍니다.

### 가장 큰 활성화 값의 통계

표 5는 부록 B에서 훈련된 Transformer와 DIFF Transformer 모델에서 수집된 활성화 값의 통계를 보여줍니다. 우리는 attention logits (즉, softmax 이전 활성화)과 hidden states (즉, layer outputs)의 두 가지 유형의 활성화를 분석합니다. 통계는 0.4M 토큰에서 수집되었습니다. 표 5에서 볼 수 있듯이, 중간값은 유사한 크기를 가지지만, DIFF Transformer는 Transformer에 비해 훨씬 낮은 상위 활성화 값을 나타냅니다. 결과는 우리의 방법이 활성화 이상치를 적게 생성함을 보여줍니다.

## Attention Logits의 양자화

그림 8: HellaSwag [12] 데이터셋에서의 Zero-shot 정확도. 우리는 attention logits을 16비트(즉, 양자화되지 않음)에서 8비트, 6비트, 4비트로 양자화합니다.

Figure 8에 나타난 것처럼, 우리는 attention logits을 더 낮은 비트로 양자화합니다. 우리는 absmax 양자화[42]를 사용하여 동적 사후 훈련 양자화를 적용합니다. 16비트 구성은 양자화되지 않은 원본 결과를 나타냅니다. 모델은 8비트, 6비트, 4비트로 점진적으로 양자화됩니다. Figure 8은 HellaSwag [12]에서의 zero-shot 성능을 보고합니다. 다른 데이터셋도 유사한 경향을 따릅니다. DIFF Transformer는 16비트에서 6비트까지 비트 폭이 줄어들더라도 높은 성능을 유지합니다. 이에 비해, Transformer의 정확도는 6비트 양자화에서 크게 떨어집니다. 4비트 DIFF Transformer는 6비트 Transformer와 유사한 정확도를 달성하며, 4비트 Transformer에 비해 약 25% 더 높은 정확도를 보입니다. 결과는 DIFF Transformer가 attention scores에서 활성화 이상치를 자연스럽게 완화하여 낮은 비트 FlashAttention [8] 구현을 위한 새로운 기회를 제공함을 나타냅니다.


## Fine-Grained Slices

### 3.8 Ablation Studies

우리는 1.4B 크기의 언어 모델을 사용하여 ablation 연구를 수행합니다. 훈련 설정은 3.2절의 1.4B 모델과 동일합니다. 모델은 Transformer의 경우 L = 24 레이어, h = 16 heads, DIFF Transformer의 경우 h = 8 heads를 가집니다. head의 차원은 d = 128입니다. 자세한 하이퍼파라미터는 부록 D에 설명되어 있습니다.

표 6은 검증 세트에 대한 세밀한 손실을 보고합니다. 우리는 Zoology [1]를 따라 손실을 "Ar-Hit"과 "Others"로 나눕니다. 특히, "Ar-Hit"은 문맥에서 이전에 본 n-gram의 마지막 토큰을 고려하여 연관 회상 능력을 평가합니다. "Others" 슬라이스는 문맥에서 회상할 수 없거나 빈번한 토큰을 나타냅니다.

표 6에서 볼 수 있듯이, 우리는 DIFF Transformer의 다양한 설계 선택을 ablate하고 여러 Transformer 변형을 제시합니다. 모든 모델이 공정한 비교를 위해 유사한 크기와 훈련 FLOPs를 가지고 있음을 주의하세요. 첫 번째와 네 번째 행은 각각 Transformer와 DIFF Transformer의 기본 설정으로, 이는 Figure 3a에서 직접 가져온 것입니다. 우리의 방법은 전반적인 손실과 세밀한 손실 측면에서 Transformer보다 우수합니다. DIFF Transformer가 모델 크기를 맞추기 위해 head의 수를 절반으로 줄였으므로, 두 번째 행은 구성 변경이 큰 영향을 미치지 않음을 보여줍니다. 우리는 DIFF Transformer에서 GroupNorm을 ablate하였고, 이는 훈련 불안정성으로 인해 성능이 저하되었습니다. 우리의 방법에서는 여러 head가 서로 다른 통계를 가지는 경향이 있기 때문에 GroupNorm이 이들을 유사한 값으로 정규화하는 데 중요한 역할을 합니다. 반면에, 세 번째와 첫 번째 행을 비교할 때, Transformer에 GroupNorm을 추가하는 것은 성능에 거의 영향을 미치지 않습니다. 결과는 우리의 방법의 개선이 구성이나 정규화 모듈이 아닌, 차별적 주의 메커니즘에서 기인함을 시사합니다. 더욱이, 우리는 λ를 초기화하는 다양한 전략을 비교합니다. 2.1절에 설명된 대로, 기본 설정은 지수 초기화를 사용합니다. 즉, $$\lambda = 0.8 - 0.6 \times \exp(-0.3 \cdot (l-1))$$이며, 여기서 l은 레이어 인덱스입니다. 마지막 두 행은 λ = 0.8, 0.5로 일정한 초기화를 사용합니다. 검증 손실의 미미한 변화는 모델이 λ 초기화 선택에 대해 견고함을 시사합니다.

### 4 결론

이 연구에서는 Differential Transformer (a.k.a. DIFF Transformer)를 소개하며, 이는 관련 문맥에 대한 주의를 증폭시키면서 노이즈를 제거합니다. 언어 모델링에 대한 실험 결과는 DIFF Transformer가 Transformer에 비해 스케일링 특성, 긴 문맥 모델링, 주요 정보 검색, 환각 완화, 문맥 내 학습 및 활성화 이상치 감소 측면에서 우수하다는 것을 보여줍니다. 결과는 주의 노이즈를 줄이는 것의 중요성을 강조합니다. 더욱이, 차별적 주의 메커니즘은 FlashAttention [8]과 쉽게 구현될 수 있습니다. 이 발견은 DIFF Transformer를 대형 언어 모델을 위한 독특하고 유망한 기초 아키텍처로 자리매김합니다. 미래에는 활성화 이상치의 크기 감소로 인해 효율적인 저비트 주의 커널을 개발할 수 있습니다. 주의 패턴이 훨씬 더 희소해짐에 따라 키-값 캐시를 압축하는 데 이 특성을 활용하고자 합니다.


**Reference**

1. Arora, S., et al.

## 감사의 말씀

Ben Huntley가 GPU 클러스터를 유지 관리해 주신 것에 대해 감사드립니다. 긴 시퀀스 훈련은 [21]의 내부 버전인 CUBE를 사용합니다.

## 참고문헌

[1] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. Zoology: Measuring and improving recall in efficient language models. arXiv preprint arXiv:2312.04927, 2023.  
[2] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023.  
[3] Amanda Bertsch, Maor Ivgi, Uri Alon, Jonathan Berant, Matthew R. Gormley, and Graham Neubig. In-context learning with long-context models: An in-depth exploration. arXiv:2405.00200, 2024.  
[4] Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Quantizable transformers: Removing outliers by helping attention heads do nothing. Advances in Neural Information Processing Systems, 36, 2024.  
[5] Iñigo Casanueva, Tadas Temcˇinas, Daniela Gerz, Matthew Henderson, and Ivan Vulic´. Efficient intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pp. 38–45, 2020.  
[6] Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James Glass. Lookback lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps. arXiv preprint arXiv:2407.07071, 2024.  
[7] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.  
[8] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.  
[9] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4599–4610, 2021.  
[10] Alexander Richard Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 1074–1084, 2019.  
[11] Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hanna Hajishirzi, Yoon Kim, and Hao Peng. Data engineering for scaling language models to 128k context. ArXiv, abs/2402.10171, 2024.  
[12] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 122023.  
[13] Xinyang Geng and Hao Liu. OpenLLaMA: An open reproduction of LLaMA. https://github.com/openlm-research/open_llama, 2023.  
[14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 6609–6625, 2020.


# Differential Attention 구현

우리는 `DiffAttn(·)`과 기존의 소프트맥스 어텐션에 대한 의사 코드를 제시합니다.

```python
def Attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    # Q, K, V: [b, n, d]
    s = 1 / sqrt(d)
    A = Q @ K.transpose(-1, -2) * s
    return softmax(A) @ V

def DiffAttn(X, W_q, W_k, W_v, λ):
    Q1, Q2 = split(X @ W_q)
    K1, K2 = split(X @ W_k)
    V = X @ W_v
    # Qi, Ki: [b, n, d]; V: [b, n, 2d]
    s = 1 / sqrt(d)
    A1 = Q1 @ K1.transpose(-1, -2) * s
    A2 = Q2 @ K2.transpose(-1, -2) * s
    return (softmax(A1) - λ * softmax(A2)) @ V
```

## Flash Attention을 사용한 구현

추가적으로, 우리는 Flash Attention[8]을 사용한 구현을 제공합니다. 구현은 Q, K, V 간의 차원을 다르게 사용할 수 있는지 여부에 따라 두 가지 유형으로 분류됩니다. 구체적으로, `FlashDiffAttn_1(·)`은 서로 다른 차원을 지원하는 패키지 (예: xformers1)를 나타내고, `FlashDiffAttn_2(·)`는 그렇지 않은 패키지 (예: flash-attention2)를 나타냅니다. 우리는 또한 공식 Flash Attention2[7]을 기반으로 수정된 맞춤형 flash-attention3 패키지를 구현하여 Q, K, V 간의 서로 다른 차원을 지원합니다.

코드 구현은 [https://aka.ms/Diff-Transformer](https://aka.ms/Diff-Transformer)에서 확인할 수 있습니다.

```python
def FlashDiffAttn_1(X, W_q, W_k, W_v, λ):
    Q1, Q2 = split(X @ W_q)
    K1, K2 = split(X @ W_k)
    V = X @ W_v
    A1 = flash_attn(Q1, K1, V)
    A2 = flash_attn(Q2, K2, V)
    return A1 - λ * A2

def FlashDiffAttn_2(X, W_q, W_k, W_v, λ):
    Q1, Q2 = split(X @ W_q)
    K1, K2 = split(X @ W_k)
    V1, V2 = split(X @ W_v)
    A11 = flash_attn(Q1, K1, V1)
    A12 = flash_attn(Q1, K1, V2)
    A1 = Concat(A11, A12)
    A21 = flash_attn(Q2, K2, V1)
    A22 = flash_attn(Q2, K2, V2)
    A2 = Concat(A21, A22)
    return A1 - λ * A2
```

## 효율성

표 7은 DIFFTransformer와 Transformer 간의 처리량을 비교합니다. 공정한 비교를 위해, 우리는 위에서 언급한 맞춤형 flash-attention 구현을 두 방법 모두에 사용합니다. 실험은 Nvidia H100-80GB GPU 카드를 사용하여 수행되었습니다.

**표 7: 처리량은 초당 토큰 수로 측정됩니다.**

표 7에서 보이는 것처럼, 우리는 다른 모델 크기 (3B, 13B)와 컨텍스트 길이 (2K, 4K)로 설정을 평가합니다. 3B 모델의 경우, DIFFTransformer에는 12개의 헤드가 있고 Transformer에는 24개의 헤드가 있습니다.


**Reference:**

1. https://github.com/facebookresearch/xformers
2. https://github.com/Dao-AILab/flash-attention
3. https://aka.ms/flash-diff


13B 모델에는 DIFF Transformer에 20개의 헤드가 있고, Transformer에는 40개의 헤드가 있습니다. 모든 모델은 동일한 헤드 차원 $d = 128$을 가지고 있습니다. 학습 효율성은 순전파와 역전파로 구성됩니다. Prefill 효율성은 순전파만 포함합니다. 표 7에서 볼 수 있듯이, 처리량 결과는 허용 범위 내에서 비교 가능합니다. 맞춤형 플래시 어텐션 구현은 FlashAttention2[7]에 기반을 두고 있음을 주목하세요. 최근 FlashAttention3[34]의 출시로 인해 처리량의 격차가 더욱 줄어들 수 있습니다. 차별적 어텐션을 위해 특별히 설계된 보다 진보된 커널 구현은 처리량을 향상시킬 수 있습니다.

### B. 언어 모델링 평가

3.1절과 동일한 설정을 따라, 350B 토큰에서 3B 크기의 언어 모델을 훈련하고 다양한 다운스트림 작업에서 DIFF Transformer와 Transformer [41]을 비교합니다. LLaMA [38]에서와 같이 증강된 Transformer 아키텍처를 사용합니다. 구체적으로, "Transformer" 모델은 RMSNorm[46], SwiGLU[35,29] 개선 사항과 편향 제거를 포함합니다.

표 8은 LMEvalHarness 벤치마크[12]에서의 제로샷 및 5샷 결과를 보고합니다. 결과는 제로샷과 몇샷 설정 모두에서 DIFF Transformer가 Transformer를 다양한 작업에서 능가함을 보여줍니다.

#### 표 8: LMEvalHarness[12]에서 잘 훈련된 Transformer 언어 모델과 DIFF Transformer의 비교. DIFF Transformer는 제로샷 및 몇샷 설정에서 더 나은 정확도를 달성합니다.


## C. Section 3.1의 하이퍼파라미터

표 9는 Section 3.1의 DIFFTransformer-3B 모델에 대한 상세한 하이퍼파라미터를 보여줍니다. Transformer-3B의 유일한 차이점은 24개의 헤드를 가진다는 것입니다. Transformer-3B와 DIFFTransformer-3B는 유사한 FLOPs를 가지고 있음을 주목하세요.

| 매개변수 | 값 |
| --- | --- |
| Layers | 28 |
| Hidden size | 3072 |
| FFN size | 8192 |
| Vocab size | 100,288 |
| Heads | 12 |
| Adam β | (0.9, 0.95) |
| LR | $3.2 \times 10^{-4}$ |
| Batch size | 4M |
| Warmup steps | 1000 |
| Weight decay | 0.1 |
| Dropout | 0.0 |

표 9: Section 3.1에서 DIFFTransformer-3B 모델에 사용된 하이퍼파라미터.

## D. Section 3.2의 하이퍼파라미터

표 10은 다양한 모델 크기에 대한 DIFFTransformer의 히든 차원, 레이어 수, 헤드 수를 보고합니다. 모든 Transformer 모델 크기에 대해 DIFFTransformer와 비교하여 파라미터를 정렬하기 위해 헤드의 수를 두 배로 늘렸습니다. FFN 크기는 히든 차원 $d_{model}$의 $\frac{8}{3}$배입니다. 학습 길이는 2048로 설정되어 있으며, 배치 크기는 0.25M 토큰으로 설정되어 있습니다. 우리는 AdamW[24]를 사용하며, $\beta_1 = 0.9$, $\beta_2 = 0.98$입니다. 학습률은 830M에서 2.8B 크기에서는 $1.5 \times 10^{-4}$이고, 6.8B에서 13.1B 크기에서는 $7.5 \times 10^{-5}$입니다. 워밍업 스텝은 375이며, 선형 비율 감소를 따릅니다. Weight decay는 0.05로 설정되어 있습니다. 모델은 40k 스텝, 즉 10B 토큰으로 학습됩니다.

| 크기 | 히든 차원 | 레이어 수 | 헤드 수 |
| --- | --- | --- | --- |
| 830M | 1536 | 24 | 8 |
| 1.4B | 2048 | 24 | 8 |
| 2.8B | 2560 | 32 | 10 |
| 6.8B | 4096 | 32 | 16 |
| 13.1B | 5120 | 40 | 20 |

표 10: Section 3.2에서 DIFFTransformer에 사용된 모델 크기와 하이퍼파라미터.

# In-Context Learning의 강건성 평가

Section 3.5에서 설명한 바와 같이, 우리는 동일한 in-context 예제들의 순열을 통해 Transformer와 DIFF Transformer의 in-context learning의 강건성을 평가합니다. 우리는 64K 길이로 확장된 3B 크기의 언어 모델을 평가합니다 (Section 3.3).

그림 9는 네 개의 데이터셋에 대한 비교를 제공하며, in-context 예제들이 무작위로 배열되었습니다. 평가 프로토콜은 Section 3.5와 동일합니다. DIFF Transformer의 정확도 변동성이 Transformer보다 일관되게 낮아, in-context learning에 있어서 DIFF Transformer의 더 큰 강건성을 나타냅니다.

**그림 9: 네 개의 데이터셋에 대한 in-context learning의 강건성 평가. 정확도는 랜덤 시드를 조정하여 예제의 순서를 변경함으로써 평가됩니다. 점선은 최상의 결과와 최악의 결과 간의 차이를 나타냅니다. 예제들은 프롬프트 내에서 무작위로 배열됩니다.**

## F GradientFlowof DIFF Transformer

우리는 Differential Attention의 Gradient Flow가 기존 Softmax Attention과 유사하다는 것을 보여줍니다. 이 특성 덕분에, Transformer에서 사용되는 동일한 하이퍼파라미터를 DIFF Transformer에 직접 적용할 수 있으며, 학습 불안정성에 대한 염려가 없습니다.

Differential Attention을 위해, 증명에서 단일 헤드를 선택하고 Equation (1)과 Equation (3)을 다음과 같이 확장합니다. 입력으로 X ∈ $\mathbb{R}^{N \times d_{\text{model}}}$, Q_1, Q_2, K_1, K_2 ∈ $\mathbb{R}^{N \times d}$, V ∈ $\mathbb{R}^{N \times 2d}$, 그리고 출력으로 O ∈ $\mathbb{R}^{N \times d_{\text{model}}}$를 가집니다:

$$[Q_1; Q_2] = [XW_{Q1}; XW_{Q2}], [K_1; K_2] = [XW_{K1}; XW_{K2}], V = XW_V$$

$$A_1 = \text{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right), A_2 = \text{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)$$

$$O = \text{GroupNorm}((A_1 - \lambda A_2)V)W_O$$

여기서 $W_{Q1}, W_{Q2}, W_{K1}, W_{K2} \in \mathbb{R}^{d_{\text{model}} \times d}$, $W_V \in \mathbb{R}^{d_{\text{model}} \times 2d}$, $W_O \in \mathbb{R}^{2d \times d_{\text{model}}}$는 파라미터입니다. λ는 학습 가능한 스칼라이고, GroupNorm은 고정된 배율로서의 스케일을 가집니다: $γ = 1 - λ_{\text{init}}$. 초기 학습 단계에서 (A_1 - \lambda A_2)V의 $x$에 대해, 우리는 다음과 같이 정의합니다:

$$\frac{\partial \text{GN}(x)}{\partial x} = \Theta\left(\frac{1}{||x||}\right) = \Theta(1) \text{ as } \sqrt{\frac{2}{2d}} = \Theta(1 - λ_{\text{init}})$$

이 표현을 바탕으로 O의 Gradient를 $\frac{\partial L}{\partial O}$로 주어졌을 때, 파라미터의 Gradient를 다음과 같이 형성합니다:

$$\frac{\partial L}{\partial W_O} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial W_O} = ((A_1 - \lambda A_2)V)^T \frac{\partial L}{\partial O}$$

$$\frac{\partial L}{\partial W_V} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial V} \frac{\partial V}{\partial W_V} = X^T (A_1 - \lambda A_2)^T W_O^T \frac{\partial L}{\partial O}$$

$$\frac{\partial L}{\partial W_{Q1}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A_1} \frac{\partial A_1}{\partial Q_1} \frac{\partial Q_1}{\partial W_{Q1}} = \frac{1}{\sqrt{d}} X^T [A_1 \odot (\frac{\partial L}{\partial O} W_O^T V^T - (A_1 \odot (\frac{\partial L}{\partial O} W_O^T V^T))J)]K_1$$

$$\frac{\partial L}{\partial W_{Q2}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A_2} \frac{\partial A_2}{\partial Q_2} \frac{\partial Q_2}{\partial W_{Q2}} = -\lambda \frac{1}{\sqrt{d}} X^T [A_2 \odot (\frac{\partial L}{\partial O} W_O^T V^T - (A_2 \odot (\frac{\partial L}{\partial O} W_O^T V^T))J)]K_2$$

$$\frac{\partial L}{\partial W_{K1}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A_1} \frac{\partial A_1}{\partial K_1} \frac{\partial K_1}{\partial W_{K1}} = \frac{1}{\sqrt{d}} X^T [A_1 \odot (\frac{\partial L}{\partial O} W_O^T V^T - (A_1 \odot (\frac{\partial L}{\partial O} W_O^T V^T))J)]^T Q_1$$

$$\frac{\partial L}{\partial W_{K2}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A_2} \frac{\partial A_2}{\partial K_2} \frac{\partial K_2}{\partial W_{K2}} = -\lambda \frac{1}{\sqrt{d}} X^T [A_2 \odot (\frac{\partial L}{\partial O} W_O^T V^T - (A_2 \odot (\frac{\partial L}{\partial O} W_O^T V^T))J)]^T Q_2$$

여기서 J ∈ $\mathbb{R}^{N \times N}$는 모두 1로 구성된 행렬입니다.

비교를 위해, 우리는 기존 Softmax Attention을 다시 형성합니다. 2d 차원의 Attention에 대해, 입력으로 X ∈ $\mathbb{R}^{N \times d_{\text{model}}}$, Q_1, Q_2, K_1, K_2 ∈ $\mathbb{R}^{N \times d}$, V ∈ $\mathbb{R}^{N \times 2d}$, 그리고 출력으로 O ∈ $\mathbb{R}^{N \times d_{\text{model}}}$를 가집니다.



$$
[Q_1;Q_2]=[XW_{Q1};XW_{Q2}], [K_1;K_2]=[XW_{K1};XW_{K2}], V =XW_V
$$

$$
A=softmax\left(\frac{Q_1 K_1^T + Q_2 K_2^T}{\sqrt{2d}}\right) \tag{8}
$$

$$
O =(AV)W_O
$$

여기서 $W_{Q1}, W_{Q2}, W_{K1}, W_{K2} \in \mathbb{R}^{d_{model} \times d}, W_V \in \mathbb{R}^{d_{model} \times 2d}, W_O \in \mathbb{R}^{2d \times d_{model}}$는 파라미터입니다.

$$
\frac{\partial L}{\partial O}
$$

$O$의 그래디언트를 나타내며, 파라미터의 그래디언트를 다음과 같이 정리합니다:

$$
\frac{\partial L}{\partial W_O} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial W_O} = \left(AV\right)^T \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial W_V} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial V} \frac{\partial V}{\partial W_V} = X^T A^T (W_O)^T \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial W_{Q1}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A} \frac{\partial A}{\partial Q_1} \frac{\partial Q_1}{\partial W_{Q1}} = \frac{1}{\sqrt{2d}} X^T \left[A \odot \left( (W_O)^T V^T - \left(A \odot \left( (W_O)^T V^T \right)\right)J \right)\right]K_1 \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial W_{Q2}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A} \frac{\partial A}{\partial Q_2} \frac{\partial Q_2}{\partial W_{Q2}} = \frac{1}{\sqrt{2d}} X^T \left[A \odot \left( (W_O)^T V^T - \left(A \odot \left( (W_O)^T V^T \right)\right)J \right)\right]K_2 \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial W_{K1}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A} \frac{\partial A}{\partial K_1} \frac{\partial K_1}{\partial W_{K1}} = \frac{1}{\sqrt{2d}} X^T \left[A \odot \left( (W_O)^T V^T - \left(A \odot \left( (W_O)^T V^T \right)\right)J \right)\right]^T Q_1 \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial W_{K2}} = \frac{\partial L}{\partial O} \frac{\partial O}{\partial A} \frac{\partial A}{\partial K_2} \frac{\partial K_2}{\partial W_{K2}} = \frac{1}{\sqrt{2d}} X^T \left[A \odot \left( (W_O)^T V^T - \left(A \odot \left( (W_O)^T V^T \right)\right)J \right)\right]^T Q_2 \frac{\partial L}{\partial O}
$$

Softmax의 특성을 사용하여, 우리는 $A_1 = A_2 = A_1 = A_2 - \lambda A_1$로 그래디언트의 크기를 고려할 수 있습니다. 따라서, Attention과 Differential Attention의 해당 파라미터의 그래디언트는 크기에서 동일하며, Equation (7)과 Equation (9)에서와 같이 일정한 상수 인자들로만 차이가 납니다. 

Gradient 크기에 불변인 AdamW[24]와 같은 Optimizer를 사용할 때, DIFFTransformer에서의 파라미터 업데이트는 Transformer와 유사합니다. 이는 Transformer 하이퍼파라미터를 훈련 불안정성을 위험에 빠뜨리지 않고 재사용할 수 있게 해줍니다.

