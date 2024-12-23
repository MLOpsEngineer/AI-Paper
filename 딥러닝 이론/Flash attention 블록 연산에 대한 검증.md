플래시어텐션에서 블록 단위로 Q (Query)와 K (Key) 행렬을 나누어 연산하는 과정이 전체 행렬 곱셈과 동일한 결과를 낼 수 있는지에 대해 구체적인 예시를 통해 검증하면 다음과 같다.

---

### **예시 설정**

우리가 사용할 예시는 두 개의 $4 \times 4$ 행렬 $Q$와 $K$입니다. 이 행렬들을 $2 \times 2$ 블록으로 분할하여 블록 단위로 곱셈을 수행한 후, 이를 다시 합쳐서 전체 $QK$ 행렬을 재구성하는 과정을 살펴보겠습니다.

#### **행렬 $Q$와 $K$ 정의**

$$
Q = \begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16 \\
\end{bmatrix}
$$

$$
K = \begin{bmatrix}
17 & 18 & 19 & 20 \\
21 & 22 & 23 & 24 \\
25 & 26 & 27 & 28 \\
29 & 30 & 31 & 32 \\
\end{bmatrix}
$$

---

### **1. 전체 행렬 곱셈 (블록을 나누지 않았을 때)**

먼저, 블록을 나누지 않고 전체 행렬을 곱하는 방법을 통해 최종 결과를 확인해보겠습니다.

$$
QK = Q \times K
$$

각 원소 $(i,j)$는 $Q$의 $i$번째 행과 $K$의 $j$번째 열의 내적입니다.

예를 들어, $QK$의 첫 번째 원소 $(1,1)$은 다음과 같이 계산됩니다:

$$
(QK)_{1,1} = 1 \times 17 + 2 \times 21 + 3 \times 25 + 4 \times 29 = 17 + 42 + 75 + 116 = 250
$$

이와 같은 방식으로 모든 원소를 계산하면 다음과 같은 $QK$ 행렬을 얻습니다:

$$
QK = \begin{bmatrix}
250 & 260 & 270 & 280 \\
618 & 644 & 670 & 696 \\
986 & 1028 & 1070 & 1112 \\
1354 & 1412 & 1470 & 1528 \\
\end{bmatrix}
$$

---

### **2. 블록 단위로 행렬을 나누어 곱하기**

이제 $Q$와 $K$ 행렬을 $2 \times 2$ 블록으로 분할하고, 블록 단위로 곱셈을 수행한 후 이를 합산하여 전체 $QK$ 행렬을 재구성해보겠습니다.

#### **블록 분할**

$$
Q = \begin{bmatrix}
Q_1 & Q_2 \\
Q_3 & Q_4 \\
\end{bmatrix}, \quad
K = \begin{bmatrix}
K_1 & K_2 \\
K_3 & K_4 \\
\end{bmatrix}
$$

각 블록은 다음과 같습니다:

$$
Q_1 = \begin{bmatrix}
1 & 2 \\
5 & 6 \\
\end{bmatrix}, \quad
Q_2 = \begin{bmatrix}
3 & 4 \\
7 & 8 \\
\end{bmatrix}, \quad
Q_3 = \begin{bmatrix}
9 & 10 \\
13 & 14 \\
\end{bmatrix}, \quad
Q_4 = \begin{bmatrix}
11 & 12 \\
15 & 16 \\
\end{bmatrix}
$$

$$
K_1 = \begin{bmatrix}
17 & 18 \\
21 & 22 \\
\end{bmatrix}, \quad
K_2 = \begin{bmatrix}
19 & 20 \\
23 & 24 \\
\end{bmatrix}, \quad
K_3 = \begin{bmatrix}
25 & 26 \\
29 & 30 \\
\end{bmatrix}, \quad
K_4 = \begin{bmatrix}
27 & 28 \\
31 & 32 \\
\end{bmatrix}
$$

#### **블록 단위 행렬 곱셈**

블록 단위로 $Q$와 $K$를 곱하여 부분 결과를 계산합니다. 전체 행렬 곱셈은 다음과 같이 4개의 블록 곱셈으로 구성됩니다:

$$
QK = \begin{bmatrix}
Q_1K_1 + Q_2K_3 & Q_1K_2 + Q_2K_4 \\
Q_3K_1 + Q_4K_3 & Q_3K_2 + Q_4K_4 \\
\end{bmatrix}
$$

각 부분을 하나씩 계산해보겠습니다.

---

#### **블록 곱셈 상세 계산**

1. **$Q_1K_1$:**

$$
Q_1K_1 = \begin{bmatrix}
1 & 2 \\
5 & 6 \\
\end{bmatrix} \times \begin{bmatrix}
17 & 18 \\
21 & 22 \\
\end{bmatrix} = \begin{bmatrix}
1 \times 17 + 2 \times 21 & 1 \times 18 + 2 \times 22 \\
5 \times 17 + 6 \times 21 & 5 \times 18 + 6 \times 22 \\
\end{bmatrix} = \begin{bmatrix}
17 + 42 & 18 + 44 \\
85 + 126 & 90 + 132 \\
\end{bmatrix} = \begin{bmatrix}
59 & 62 \\
211 & 222 \\
\end{bmatrix}
$$

2. **$Q_2K_3$:**

$$
Q_2K_3 = \begin{bmatrix}
3 & 4 \\
7 & 8 \\
\end{bmatrix} \times \begin{bmatrix}
25 & 26 \\
29 & 30 \\
\end{bmatrix} = \begin{bmatrix}
3 \times 25 + 4 \times 29 & 3 \times 26 + 4 \times 30 \\
7 \times 25 + 8 \times 29 & 7 \times 26 + 8 \times 30 \\
\end{bmatrix} = \begin{bmatrix}
75 + 116 & 78 + 120 \\
175 + 232 & 182 + 240 \\
\end{bmatrix} = \begin{bmatrix}
191 & 198 \\
407 & 422 \\
\end{bmatrix}
$$

3. **합산 $Q_1K_1 + Q_2K_3$:**

$$
Q_1K_1 + Q_2K_3 = \begin{bmatrix}
59 & 62 \\
211 & 222 \\
\end{bmatrix} + \begin{bmatrix}
191 & 198 \\
407 & 422 \\
\end{bmatrix} = \begin{bmatrix}
250 & 260 \\
618 & 644 \\
\end{bmatrix}
$$

4. **$Q_1K_2$:**

$$
Q_1K_2 = \begin{bmatrix}
1 & 2 \\
5 & 6 \\
\end{bmatrix} \times \begin{bmatrix}
19 & 20 \\
23 & 24 \\
\end{bmatrix} = \begin{bmatrix}
1 \times 19 + 2 \times 23 & 1 \times 20 + 2 \times 24 \\
5 \times 19 + 6 \times 23 & 5 \times 20 + 6 \times 24 \\
\end{bmatrix} = \begin{bmatrix}
19 + 46 & 20 + 48 \\
95 + 138 & 100 + 144 \\
\end{bmatrix} = \begin{bmatrix}
65 & 68 \\
233 & 244 \\
\end{bmatrix}
$$

5. **$Q_2K_4$:**

$$
Q_2K_4 = \begin{bmatrix}
3 & 4 \\
7 & 8 \\
\end{bmatrix} \times \begin{bmatrix}
27 & 28 \\
31 & 32 \\
\end{bmatrix} = \begin{bmatrix}
3 \times 27 + 4 \times 31 & 3 \times 28 + 4 \times 32 \\
7 \times 27 + 8 \times 31 & 7 \times 28 + 8 \times 32 \\
\end{bmatrix} = \begin{bmatrix}
81 + 124 & 84 + 128 \\
189 + 248 & 196 + 256 \\
\end{bmatrix} = \begin{bmatrix}
205 & 212 \\
437 & 452 \\
\end{bmatrix}
$$

6. **합산 $Q_1K_2 + Q_2K_4$:**

$$
Q_1K_2 + Q_2K_4 = \begin{bmatrix}
65 & 68 \\
233 & 244 \\
\end{bmatrix} + \begin{bmatrix}
205 & 212 \\
437 & 452 \\
\end{bmatrix} = \begin{bmatrix}
270 & 280 \\
670 & 696 \\
\end{bmatrix}
$$

7. **$Q_3K_1$:**

$$
Q_3K_1 = \begin{bmatrix}
9 & 10 \\
13 & 14 \\
\end{bmatrix} \times \begin{bmatrix}
17 & 18 \\
21 & 22 \\
\end{bmatrix} = \begin{bmatrix}
9 \times 17 + 10 \times 21 & 9 \times 18 + 10 \times 22 \\
13 \times 17 + 14 \times 21 & 13 \times 18 + 14 \times 22 \\
\end{bmatrix} = \begin{bmatrix}
153 + 210 & 162 + 220 \\
221 + 294 & 234 + 308 \\
\end{bmatrix} = \begin{bmatrix}
363 & 382 \\
515 & 542 \\
\end{bmatrix}
$$

8. **$Q_4K_3$:**

$$
Q_4K_3 = \begin{bmatrix}
11 & 12 \\
15 & 16 \\
\end{bmatrix} \times \begin{bmatrix}
25 & 26 \\
29 & 30 \\
\end{bmatrix} = \begin{bmatrix}
11 \times 25 + 12 \times 29 & 11 \times 26 + 12 \times 30 \\
15 \times 25 + 16 \times 29 & 15 \times 26 + 16 \times 30 \\
\end{bmatrix} = \begin{bmatrix}
275 + 348 & 286 + 360 \\
375 + 464 & 390 + 480 \\
\end{bmatrix} = \begin{bmatrix}
623 & 646 \\
839 & 870 \\
\end{bmatrix}
$$

9. **합산 $Q_3K_1 + Q_4K_3$:**

$$
Q_3K_1 + Q_4K_3 = \begin{bmatrix}
363 & 382 \\
515 & 542 \\
\end{bmatrix} + \begin{bmatrix}
623 & 646 \\
839 & 870 \\
\end{bmatrix} = \begin{bmatrix}
986 & 1028 \\
1354 & 1412 \\
\end{bmatrix}
$$

10. **$Q_3K_2$:**

$$
Q_3K_2 = \begin{bmatrix}
9 & 10 \\
13 & 14 \\
\end{bmatrix} \times \begin{bmatrix}
19 & 20 \\
23 & 24 \\
\end{bmatrix} = \begin{bmatrix}
9 \times 19 + 10 \times 23 & 9 \times 20 + 10 \times 24 \\
13 \times 19 + 14 \times 23 & 13 \times 20 + 14 \times 24 \\
\end{bmatrix} = \begin{bmatrix}
171 + 230 & 180 + 240 \\
247 + 322 & 260 + 336 \\
\end{bmatrix} = \begin{bmatrix}
401 & 420 \\
569 & 596 \\
\end{bmatrix}
$$

11. **$Q_4K_4$:**

$$
Q_4K_4 = \begin{bmatrix}
11 & 12 \\
15 & 16 \\
\end{bmatrix} \times \begin{bmatrix}
27 & 28 \\
31 & 32 \\
\end{bmatrix} = \begin{bmatrix}
11 \times 27 + 12 \times 31 & 11 \times 28 + 12 \times 32 \\
15 \times 27 + 16 \times 31 & 15 \times 28 + 16 \times 32 \\
\end{bmatrix} = \begin{bmatrix}
297 + 372 & 308 + 384 \\
405 + 496 & 420 + 512 \\
\end{bmatrix} = \begin{bmatrix}
669 & 692 \\
901 & 932 \\
\end{bmatrix}
$$

12. **합산 $Q_3K_2 + Q_4K_4$:**

$$
Q_3K_2 + Q_4K_4 = \begin{bmatrix}
401 & 420 \\
569 & 596 \\
\end{bmatrix} + \begin{bmatrix}
669 & 692 \\
901 & 932 \\
\end{bmatrix} = \begin{bmatrix}
1070 & 1112 \\
1470 & 1528 \\
\end{bmatrix}
$$

---

### **3. 전체 $QK$ 행렬 재구성**

이제 각 블록 곱셈의 합산 결과를 이용해 전체 $QK$ 행렬을 재구성합니다.

$$
QK = \begin{bmatrix}
Q_1K_1 + Q_2K_3 & Q_1K_2 + Q_2K_4 \\
Q_3K_1 + Q_4K_3 & Q_3K_2 + Q_4K_4 \\
\end{bmatrix} = \begin{bmatrix}
250 & 260 \\
618 & 644 \\
\end{bmatrix} \quad \begin{bmatrix}
270 & 280 \\
670 & 696 \\
\end{bmatrix} \quad \begin{bmatrix}
986 & 1028 \\
1354 & 1412 \\
\end{bmatrix} \quad \begin{bmatrix}
1070 & 1112 \\
1470 & 1528 \\
\end{bmatrix}
$$

전체 $QK$ 행렬은 다음과 같습니다:

$$
QK = \begin{bmatrix}
250 & 260 & 270 & 280 \\
618 & 644 & 670 & 696 \\
986 & 1028 & 1070 & 1112 \\
1354 & 1412 & 1470 & 1528 \\
\end{bmatrix}
$$

---

### **4. 결과 비교**

앞서 블록을 나누지 않고 전체 행렬을 곱했을 때 얻은 $QK$ 행렬과 블록 단위로 곱셈을 수행한 후 재구성한 $QK$ 행렬을 비교해보면 동일한 결과를 얻을 수 있음을 확인할 수 있습니다.

**블록을 나누지 않았을 때:**

$$
QK = \begin{bmatrix}
250 & 260 & 270 & 280 \\
618 & 644 & 670 & 696 \\
986 & 1028 & 1070 & 1112 \\
1354 & 1412 & 1470 & 1528 \\
\end{bmatrix}
$$

**블록 단위로 곱셈 후 재구성:**

$$
QK = \begin{bmatrix}
250 & 260 & 270 & 280 \\
618 & 644 & 670 & 696 \\
986 & 1028 & 1070 & 1112 \\
1354 & 1412 & 1470 & 1528 \\
\end{bmatrix}
$$

두 결과는 완전히 동일합니다.


---

### **5. 소프트맥스 연산

소프트맥스는 주어진 벡터의 각 요소를 확률로 변환하는 함수로, 다음과 같이 계산됩니다:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

여기서:
- $x_i$는 벡터 $X$의 $i$번째 요소입니다.
- $N$은 벡터 $X$의 요소 개수입니다.

소프트맥스 연산의 핵심은 **분모**인 모든 지수의 합을 계산하는 것입니다. 이 분모는 벡터 전체의 정보를 필요로 합니다.

---

### **6. 플래시어텐션에서의 소프트맥스 블록 단위 처리**

플래시어텐션은 소프트맥스 연산을 블록 단위로 분할하여 수행함으로써 메모리 접근 시간을 줄이고 연산 속도를 향상시킵니다. 블록 단위로 소프트맥스를 계산하려면 전체 소프트맥스 값을 정확히 재현할 수 있는 방법이 필요합니다. 이를 위해 **블록별 최대값**을 사용한 정규화와 **블록별 지수의 합**을 활용합니다.

---

### **7. 구체적인 예시를 통한 단계별 설명**

#### **예시 설정**

- **전체 벡터 $X$**: $[1, 2, 3, 4, 5, 6]$
- **블록 크기**: 3개 요소씩 두 개의 블록으로 분할
  - **블록 1 ($X_1$)**: $[1, 2, 3]$
  - **블록 2 ($X_2$)**: $[4, 5, 6]$

#### **1단계: 전체 소프트맥스 계산**

먼저, 블록 단위로 나누지 않고 전체 소프트맥스를 계산하여 기준 값을 설정해보겠습니다.

$$
\text{softmax}(X) = \left[\frac{e^1}{S}, \frac{e^2}{S}, \frac{e^3}{S}, \frac{e^4}{S}, \frac{e^5}{S}, \frac{e^6}{S}\right]
$$

여기서 $S$는 전체 지수의 합입니다.

$$
S = e^1 + e^2 + e^3 + e^4 + e^5 + e^6
$$

#### **2단계: 블록 단위로 벡터 분할**

벡터 $X$를 두 개의 블록으로 분할합니다.

$$
X_1 = [1, 2, 3], \quad X_2 = [4, 5, 6]
$$

#### **3단계: 블록별 최대값 계산 및 전체 최대값 도출**

소프트맥스 계산의 수치 안정성을 위해 각 블록의 최대값을 찾고, 이를 전체 최대값으로 사용하여 정규화를 수행합니다.

- **블록 1 ($X_1$)의 최대값**: $\max(X_1) = 3$
- **블록 2 ($X_2$)의 최대값**: $\max(X_2) = 6$
- **전체 최대값**: $\max(X) = \max(3, 6) = 6$

#### **4단계: 블록별 정규화 및 지수 계산**

각 블록의 요소에서 전체 최대값을 빼고 지수를 계산합니다.

- **블록 1 ($X_1$) 정규화 및 지수 계산**:
  $$
  X_1' = [1 - 6, 2 - 6, 3 - 6] = [-5, -4, -3]
$$
  $$
  e^{X_1'} = [e^{-5}, e^{-4}, e^{-3}]
$$
  
- **블록 2 ($X_2$) 정규화 및 지수 계산**:
  $$
  X_2' = [4 - 6, 5 - 6, 6 - 6] = [-2, -1, 0]
$$
  $$
  e^{X_2'} = [e^{-2}, e^{-1}, e^{0}]
$$

#### **5단계: 블록별 지수의 합 계산**

각 블록별로 지수의 합을 계산하고, 이를 전체 지수의 합에 포함시킵니다.

- **블록 1 ($X_1$) 지수 합**:
  $$
  S_1 = e^{-5} + e^{-4} + e^{-3}
$$
  
- **블록 2 ($X_2$) 지수 합**:
  $$
  S_2 = e^{-2} + e^{-1} + e^{0}
$$
  
- **전체 지수 합**:
  $$
  S = S_1 + S_2 = (e^{-5} + e^{-4} + e^{-3}) + (e^{-2} + e^{-1} + e^{0})
$$

#### **6단계: 블록별 소프트맥스 결과 계산**

각 블록의 정규화된 지수를 전체 합 $S$으로 나누어 소프트맥스를 계산합니다.

- **블록 1 ($X_1$) 소프트맥스**:
  $$
  \text{softmax}(X_1) = \left[\frac{e^{-5}}{S}, \frac{e^{-4}}{S}, \frac{e^{-3}}{S}\right]
$$
  
- **블록 2 ($X_2$) 소프트맥스**:
  $$
  \text{softmax}(X_2) = \left[\frac{e^{-2}}{S}, \frac{e^{-1}}{S}, \frac{e^{0}}{S}\right]
$$

#### **7단계: 블록별 소프트맥스 결과 결합하여 전체 소프트맥스 구성**

블록별로 계산된 소프트맥스 결과를 원래의 벡터 순서에 맞게 결합하여 전체 소프트맥스를 완성합니다.

$$
\text{softmax}(X) = \left[\frac{e^{-5}}{S}, \frac{e^{-4}}{S}, \frac{e^{-3}}{S}, \frac{e^{-2}}{S}, \frac{e^{-1}}{S}, \frac{e^{0}}{S}\right]
$$

---

### **8. 구체적인 숫자 예시**

좀 더 명확히 이해하기 위해 실제 숫자를 대입하여 계산해보겠습니다.

#### **예시 벡터**

$$
X = [1, 2, 3, 4, 5, 6]
$$

#### **블록 분할**

$$
X_1 = [1, 2, 3], \quad X_2 = [4, 5, 6]
$$

#### **블록별 최대값 및 전체 최대값**

- $\max(X_1) = 3$
- $\max(X_2) = 6$
- $\max(X) = 6$

#### **블록별 정규화 및 지수 계산**

- **블록 1 ($X_1$) 정규화**:
  $$
  X_1' = [1 - 6, 2 - 6, 3 - 6] = [-5, -4, -3]
$$
  $$
  e^{X_1'} = [e^{-5} \approx 0.0067, \, e^{-4} \approx 0.0183, \, e^{-3} \approx 0.0498]
$$
  
- **블록 2 ($X_2$) 정규화**:
  $$
  X_2' = [4 - 6, 5 - 6, 6 - 6] = [-2, -1, 0]
$$
  $$
  e^{X_2'} = [e^{-2} \approx 0.1353, \, e^{-1} \approx 0.3679, \, e^{0} = 1.0]
$$

#### **블록별 지수의 합 계산**

- **블록 1 ($X_1$) 지수 합**:
  $$
  S_1 = 0.0067 + 0.0183 + 0.0498 = 0.0748
$$
  
- **블록 2 ($X_2$) 지수 합**:
  $$
  S_2 = 0.1353 + 0.3679 + 1.0 = 1.5032
$$
  
- **전체 지수 합**:
  $$
  S = S_1 + S_2 = 0.0748 + 1.5032 = 1.578
$$

#### **블록별 소프트맥스 결과 계산**

- **블록 1 ($X_1$) 소프트맥스**:
  $$
  \text{softmax}(X_1) = \left[\frac{0.0067}{1.578}, \frac{0.0183}{1.578}, \frac{0.0498}{1.578}\right] \approx [0.0042, 0.0116, 0.0316]
$$
  
- **블록 2 ($X_2$) 소프트맥스**:
  $$
  \text{softmax}(X_2) = \left[\frac{0.1353}{1.578}, \frac{0.3679}{1.578}, \frac{1.0}{1.578}\right] \approx [0.0857, 0.2328, 0.6345]
$$

#### **전체 소프트맥스 결과 결합**

$$
\text{softmax}(X) = [0.0042, 0.0116, 0.0316, 0.0857, 0.2328, 0.6345]
$$

#### **전체 소프트맥스 직접 계산과 비교**

이제 전체 소프트맥스를 직접 계산하여 블록 단위 계산과 비교해보겠습니다.

1. **전체 지수 계산**:
   $$
   e^1 \approx 2.7183, \quad e^2 \approx 7.3891, \quad e^3 \approx 20.0855
$$
   $$
   e^4 \approx 54.5982, \quad e^5 \approx 148.4132, \quad e^6 \approx 403.4288
$$
   
2. **전체 지수 합**:
   $$
   S = 2.7183 + 7.3891 + 20.0855 + 54.5982 + 148.4132 + 403.4288 \approx 636.6321
$$
   
3. **소프트맥스 계산**:
   $$
   \text{softmax}(X) = \left[\frac{2.7183}{636.6321}, \frac{7.3891}{636.6321}, \frac{20.0855}{636.6321}, \frac{54.5982}{636.6321}, \frac{148.4132}{636.6321}, \frac{403.4288}{636.6321}\right]
$$
   $$
   \approx [0.0043, 0.0116, 0.0316, 0.0856, 0.2330, 0.6344]
$$
   
#### **결과 비교**

- **블록 단위 소프트맥스**:
  $$
  [0.0042, 0.0116, 0.0316, 0.0857, 0.2328, 0.6345]
$$
  
- **전체 소프트맥스 직접 계산**:
  $$
  [0.0043, 0.0116, 0.0316, 0.0856, 0.2330, 0.6344]
$$

두 결과는 소수점 아래 몇 자리에서 약간의 오차가 있지만, 이는 소수점 근사값의 차이로 인해 발생한 것입니다. 실제로 블록 단위 소프트맥스 연산은 전체 소프트맥스 연산과 동일한 결과를 정확히 재현합니다.

---

### **9. 플래시어텐션에서의 블록 단위 소프트맥스 연산의 핵심 포인트**

1. **블록별 최대값 사용**:
   - 각 블록의 최대값을 구하여 전체 최대값을 도출함으로써 수치 안정성을 확보합니다.
   
2. **정규화**:
   - 각 블록의 요소에서 전체 최대값을 빼고 지수를 계산하여 큰 수로 인한 오버플로우를 방지합니다.
   
3. **블록별 지수의 합**:
   - 각 블록의 지수 합을 구하고, 이를 전체 지수 합에 포함시켜 전체 소프트맥스의 분모를 정확히 계산합니다.
   
4. **블록별 소프트맥스 계산**:
   - 정규화된 지수를 전체 지수 합으로 나누어 각 블록의 소프트맥스 값을 계산합니다.
   
5. **결과 결합**:
   - 블록별로 계산된 소프트맥스 결과를 원래의 벡터 순서에 맞게 결합하여 전체 소프트맥스 결과를 완성합니다.

---


### **결론**

**블록 단위 행렬 곱셈이 전체 행렬 곱셈과 동일한 결과를 내는 이유:**

- **수학적 동등성:** 행렬을 블록으로 분할하여 곱한 후 합산하는 과정은 전체 행렬을 직접 곱하는 것과 수학적으로 동일한 결과를 보장합니다. 이는 행렬 곱셈의 분배법칙에 기반합니다.
  
- **메모리 최적화:** 블록 단위로 나누어 처리하면, 각 블록이 SRAM과 같은 빠른 메모리에 적재될 수 있어 데이터 접근 속도를 높일 수 있습니다. 큰 행렬을 한 번에 처리할 때 발생하는 메모리 병목을 줄일 수 있습니다.
  
- **병렬 처리:** 각 블록 단위의 연산은 독립적으로 수행될 수 있어 GPU의 병렬 처리 능력을 최대한 활용할 수 있습니다. 이는 전체 연산 속도를 크게 향상시킵니다.

**FlashAttention의 장점:**

1. **메모리 접근 최적화:** 블록 단위로 데이터를 처리함으로써 HBM (High Bandwidth Memory)과 SRAM 간의 데이터 이동을 최소화하고, SRAM의 빠른 속도를 최대한 활용합니다.
   
2. **연산 속도 향상:** 블록 단위로 병렬적으로 연산을 수행하여 전체 어텐션 연산 시간을 단축합니다.
   
3. **메모리 사용 효율성:** 큰 어텐션 행렬을 한 번에 저장하거나 읽지 않고, 블록 단위로 처리하여 메모리 사용을 최적화합니다.
   
4. **수치 안정성 유지:** 소프트맥스 연산 시 블록 단위로 정규화를 수행하여 수치 안정성을 유지합니다.