
Sharath Turuvekere Sreenivas*, Saurav Muralidharan*, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz and Pavlo Molchanov

## 초록

우리는 Llama 3.1 8B 모델과 Mistral NeMo 12B 모델을 각각 4B와 8B 파라미터로 압축하는 방법에 대한 종합적인 보고서를 제시합니다. 이 과정에서는 가지치기(pruning)와 증류(distillation)를 사용합니다 [1]. 두 가지의 가지치기 전략을 탐구합니다: (1) 깊이 가지치기(depth pruning)와 (2) 숨겨진/Attention/MLP (너비) 가지치기(joint hidden/attention/MLP (width) pruning)입니다. 그리고 이것을 LM Evaluation Harness [2]의 공통 벤치마크에서 평가합니다. 모델들은 NeMo Aligner로 정렬된 후, 인스트럭션 튜닝된 버전으로 테스트됩니다. 이 접근법은 Llama 3.1 8B에서 매력적인 4B 모델을 생성하고, Mistral NeMo 12B에서 최첨단 Mistral-NeMo-Minitron-8B (간단히 MN-Minitron-8B) 모델을 생성합니다. 원본 데이터를 사용할 수 없는 경우, 증류 데이터셋에 대해 교사 모델을 약간 미세 조정하는 것이 유익하다는 것을 발견했습니다. 우리는 Hugging Face에 허용적인 라이선스로 기본 모델 가중치를 공개합니다.

Hugging Face의 모델: Mistral-NeMo-Minitron-8B-Base | Llama-3.1-Minitron-4B-Width-Base | Llama-3.1-Minitron-4B-Depth-Base

## 소개

대형 언어 모델(LLM) 제공업체는 보통 서로 다른 크기(예: Llama 3.1 8B, 70B, 405B)의 모델 군을 처음부터 훈련합니다. 이는 사용자가 다양한 배포 규모, 크기 및 계산 예산을 목표로 할 수 있도록 돕기 위해서입니다. 그러나 수십억 개의 파라미터를 가진 여러 모델을 처음부터 훈련하는 것은 시간, 데이터 및 자원이 매우 많이 소모됩니다.

최근의 연구 [1]는 가지치기와 지식 증류를 결합하여 LLM 모델 군의 훈련 비용을 크게 줄일 수 있음을 보여주었습니다. 여기서, 모델 군 중 가장 큰 모델만 처음부터 훈련되고, 다른 모델들은 더 큰 모델을 연속적으로 가지치기하고, 그 후 지식 증류 [3]를 수행하여 가지치기된 모델의 정확성을 회복합니다.

이 보고서에서, 우리는 Minitron 압축 전략 [1]을 두 개의 최첨단 모델인 Llama 3.1 8B [4]와 Mistral NeMo 12B [5]에 성공적으로 적용하여 각각 4B와 8B 파라미터로 압축합니다. 그림 1은 우리의 접근법에 대한 고수준 개요를 제공합니다.

원본 논문 [1]을 따르는 동안, 우리는 중요한 변경을 합니다: 원본 훈련 데이터에 접근할 수 없기 때문에, 우리는 가지치기와 증류 전에 교사 모델을 자체 데이터셋으로 미세 조정합니다. 우리는 이 단계를 교사 수정(teacher correction)이라고 부릅니다. 그림 4는 교사 수정을 생략할 경우 데이터 분포 불일치가 발생하여 증류에 부정적인 영향을 미친다는 것을 보여줍니다.

표 1은 우리의 결과를 요약합니다: 우리의 압축 전략은 최첨단 8B 모델 (MN-Minitron-8B)을 생성하며, 이는 일반적인 언어 모델링 벤치마크에서 모든 유사 크기의 모델보다 뛰어납니다. 우리의 Llama-3.1-Minitron-4B 모델들(깊이와 너비 가지치기 변형 모두)도 교사 모델에 비해 강력한 정확성을 보여줍니다. 두 변형 중 너비 가지치기 변형이 깊이 가지치기 변형보다 우수합니다. TensorRT-LLM을 사용하여 측정한 실행 시간 추론 성능을 기준으로 할 때, Llama-3.1-Minitron-4B 모델은 깊이와 너비 가지치기 변형에서 각각 2.7배와 1.8배의 평균 속도 향상을 제공합니다.

## 방법론

우리의 접근법에 대한 고수준 개요는 그림 1에 설명되어 있습니다. 여기서, 교사 모델은 먼저 증류에 사용할 대상 데이터셋에 가볍게 미세 조정됩니다 - 우리는 이 단계를 교사 수정(teacher correction)이라고 부릅니다.

## 표 1: MN-Minitron-8B 및 Llama-3.1-Minitron-4B 모델의 정확도 수치
우리의 모델을 다양한 일반적인 언어 모델링 벤치마크에서 유사한 크기의 SoTA 오픈 모델과 비교합니다. 모든 평가는 우리가 수행했으며, *로 표시된 항목은 해당 논문에서 가져왔습니다.

## 표 2: 정렬된 Llama-3.1-Minitron 모델의 정확도 수치
우리의 모델을 다양한 벤치마크에서 유사한 크기의 SoTA 오픈 정렬 모델과 비교합니다. 모든 평가는 우리가 수행했습니다. *는 벤치마크의 대표적인 하위 집합에서 얻은 결과를 나타냅니다. 최상위는 굵게, 두 번째는 밑줄로 표시됩니다. MN-Minitron-8B의 정렬은 진행 중이며, 준비가 완료되면 게시될 예정입니다.

다음으로, 모델을 압축하기 위해 Pruning이 적용된 후, Distillation을 사용하여 손실된 모델 정확도를 복구합니다. Pruning 및 Distillation 방법에 대한 전체 설명은 Minitron 논문 [1]을 참조하십시오.

### Pruning
Weight pruning은 모델 크기를 줄이기 위한 강력하고 잘 알려진 기법입니다. 이 보고서에서는 모델 가중치에서 비영 요소 블록(또는 채널)이 한 번에 제거되는 구조적 Pruning에 초점을 맞춥니다; 구조적 Pruning 기법의 예로는 neuron, attention head, convolutional filter, depth pruning이 있습니다 [1]. LLM의 경우, 그림 2에 표시된 바와 같이, 각 layer, neuron, head 및 embedding 차원의 중요도를 계산하여 Pruning 프로세스를 시작합니다. 그런 다음 이러한 중요도 점수를 정렬하여 해당 중요도 순위를 계산합니다.

#### 중요도 추정:
우리는 깊이, neuron, head 및 embedding channel을 고려하여 모든 축에 대한 민감도 정보를 동시에 계산하는 순수하게 activation 기반의 중요도 추정 전략을 사용합니다. 이 목적을 위해 소량의 Calibration 데이터셋(1024 샘플)을 사용합니다.

깊이 Pruning은 특별한 경우로 간주하며 다른 차원을 압축하는 것과 결합하지 않습니다. 우리는 Multi-Head Attention(MHA), Multi-Layer Perceptron(MLP) 및 LayerNorm 레이어에 의해 생성된 activation을 조사하여 각 head, neuron 및 embedding channel의 중요도를 계산합니다. 깊이 Pruning의 경우, layer 중요도를 평가하기 위한 세 가지 개별적인 지표를 고려합니다: (1) LM validation loss, (2) Block Importance (BI) [6] 및 (3)

## 그림 2: 원본 논문 [1]에 설명된 Pruning 및 Distillation 프로세스
우리는 이 작업에서 동일한 접근 방식을 따릅니다.

- Pruning과 Distillation 과정은 원본 논문 [1]에 설명된 대로 진행됩니다.
- 우리는 소량의 Calibration 데이터셋과 오직 Forward Propagation 패스를 사용합니다.


다운스트림 작업의 정확도를 위해, 손실 기반의 순위 매기기에서는 단일 또는 연속적인 레이어 블록을 제거하고 LM 손실에 미치는 영향을 계산합니다. 이는 레이어의 "중요성" 또는 민감도로 작용합니다. BI는 레이어 또는 레이어 블록의 입력과 출력 간의 코사인 거리를 사용합니다. 우리는 BI와 LM 손실 지표가 높은 상관관계를 가지고 있지만, 그림 8과 9에서 보여주듯 다운스트림 작업에서 가장 정확한 가지치기 모델을 생성하지 않는다는 것을 발견했습니다. 따라서 우리는 Winogrande 벤치마크 [7]를 사용하여 레이어 중요성을 평가합니다.

## 모델 다듬기

그림 2에 나타난 바와 같이, 주어진 아키텍처 구성에 대해 각 축의 요소를 계산된 중요도에 따라 먼저 순위 매기고 해당 가중치 행렬을 직접 다듬기(재구성)합니다. 뉴런과 헤드 가지치기의 경우, 각각 MLP와 MHA 레이어 가중치를 다듬습니다. 임베딩 채널의 경우, MLP, MHA, 그리고 LayerNorm 레이어의 가중치 행렬의 임베딩 차원을 다듬습니다. 원래 방법([1])은 최적의 아키텍처를 찾기 위해 Neural Architecture Search (NAS)를 사용합니다. 이 작업에서는 이 단계를 생략하고 대신 원래 논문에서 얻은 네트워크 아키텍처 관련 학습을 활용합니다.

## Distillation을 통한 재훈련

우리는 가지치기 이후의 정확도 회복 과정을 재훈련이라고 부릅니다. 이 작업에서는 두 가지 재훈련 전략을 탐구합니다: (1) 진실 라벨을 활용한 전통적인 훈련, (2) 언프룬된 모델(교사)의 감독을 사용한 Knowledge Distillation. Knowledge Distillation (KD) [3]은 더 크거나 복잡한 모델인 교사로부터 더 작거나 간단한 모델인 학생에게 지식을 전이하는 것을 포함합니다. 학생 모델이 교사 모델의 출력 및/또는 중간 상태를 모방하게 하여 지식 전이를 달성합니다. 우리의 경우, 압축되지 않은 모델과 가지치기 모델이 각각 교사와 학생에 해당합니다. Distillation을 위해, 우리는 이전 작업[1]에서의 모범 사례를 따르며 교사와 학생의 로짓에 대해서만 Forward KL Divergence 손실[8]을 사용합니다 ([3]을 따름). 이는 그림 3에 설명되어 있습니다.

## 훈련 세부 사항

### 사전 훈련

Llama 3.1 8B [4]와 Mistral NeMo [5] 12B는 우리가 접근할 수 없는 여러 독점 데이터셋에서 사전 훈련됩니다. Llama 3.1 기술 보고서 [4]에 따르면, 8B 모델은 15T 토큰으로 사전 훈련됩니다. 우리는 Hugging Face에서 공개적으로 사용할 수 있는 해당 Base 모델로 시작합니다.

### Pruning

우리의 단순화된 Pruning 레시피는 Minitron 논문 [1]에서 설명된 모범 사례를 기반으로 하며 Methodology 섹션에서 설명됩니다. 구체적으로, 너비 가지치기의 경우, 우리는 (1) 배치 및 시퀀스 차원 전체에서 l2-norm 및 평균을 집계 함수로 사용하고, (2) 반복적인 접근 방식을 피하고 단일 샷 가지치기를 수행합니다. 깊이 가지치기의 경우, Methodology 섹션에서 설명된 대로, Gromov et al. [11]의 관찰을 따르고 Winogrande [7]에서 정확도 감소가 가장 적은 연속적인 레이어 하위 그룹을 제거합니다. 이 작업에서는 경량 Neural Architecture Search (NAS) 단계를 생략하고, Llama-3.1-Minitron-4B와 MN-Minitron-8B 모두에 대해 수동 아키텍처 구성을 사용합니다. 우리가 고안한 아키텍처는 Minitron-4B와 Minitron-8B 모델에서 영감을 받았으며, 표 3에 자세히 설명되어 있습니다. 이제 각 대상 압축 모델에 대한 가지치기 레시피를 설명합니다.

## 그림 3 | Distillation 개요: 원래의 훈련 데이터가 없는 경우, 교사 모델의 약간의 미세 조정을 권장합니다. Distillation은 원래 모델을 교사로, 가지치기 모델을 학생으로 하여 로짓에서 KL divergence를 최소화함으로써 수행됩니다.

### 데이터셋

우리는 모든 가지치기와 Distillation 실험을 위해 Nemotron-4 curated continued training (CT) 데이터셋 [9][10]을 사용합니다.

## Llama-3.1-Minitron-4B-Width

- 시작 모델: Llama 3.1 8B 2.0
- Hidden dimension: 4096 → 3072
- MLP hidden dimension: 14336 → 9216
- Attention heads: 변화 없음
- Depth: 변화 없음

## Llama-3.1-Minitron-4B-Depth

- 시작 모델: Llama 3.1 8B
- Hidden dimension: 변화 없음
- MLP hidden dimension: 변화 없음
- Attention heads: 변화 없음
- Depth: 32 → 16

## MN-Minitron-8B

- 시작 모델: Mistral NeMo 12B
- Hidden dimension: 5120 → 4096
- MLP hidden dimension: 14336 → 11520
- Attention heads: 변화 없음
- Depth: 변화 없음

## Distillation

**Teacher Correction:** Mistral NeMo 12B 모델을 직접적으로 교사로 사용하는 것은 우리 데이터셋에서 최적의 성능을 발휘하지 못합니다. 이는 교사 모델이 훈련된 원본 데이터셋과 증류에 사용되는 데이터셋 간의 서브-워드 토큰 분포의 변화 때문입니다. 이를 해결하기 위해, 약 127B개의 토큰을 사용하여 데이터셋에서 교사를 미세 조정합니다. 그림 4에 나타난 바와 같이, 원본 데이터셋이 증류 시 사용 가능하지 않을 경우 이러한 수정이 필수적입니다. 우리는 이 기술을 Mistral-NeMo와 Llama-3.1 교사 모델 모두에 적용합니다. 미세 조정 과정은 교사 모델의 다운스트림 작업에 대한 정확성에 약간의 영향을 미치며, 일부 작업은 개선되고 일부는 감소합니다. 이는 테이블 1에 나타나 있습니다. 우리는 이것이 미세 조정에 사용된 데이터셋의 특성으로 인한 것이라고 가정합니다.

## Retraining

Minitron 작업 [1]에서 배운 내용을 바탕으로, 우리는 로짓-온리 증류를 선택하여 교사와 학생의 확률 간의 forward KL Divergence [8] 손실을 최소화하고, LM cross-entropy 손실은 무시합니다. 여기서, 가지치기되지 않은 모델과 가지치기된 모델은 각각 교사와 학생에 해당합니다. 우리는 증류 중 테이블 4에 나열된 하이퍼파라미터를 사용합니다. 우리의 훈련 작업에는 32개의 NVIDIA DGX H100 노드를 사용합니다.

## Instruction Tuning

증류된 모델의 지시 따름 능력을 평가하기 위해, 우리는 Nemotron-4 340B [13]에 사용된 지시 튜닝 데이터셋을 사용하여 NeMo-Aligner [12]로 Llama-3.1-Minitron 4B 모델에 대해 감독된 미세 조정(SFT)을 수행합니다. 테이블 2에서 보이는 바와 같이, 우리는 IFEval [14]와 MT-Bench [15]를 통해 지시 따름 및 역할 놀이 능력을 평가하고, ChatRAG-Bench [16]를 통해 RAG QA를, BFCL [17]을 통해 함수 호출 능력을 평가합니다.

**표 3 | 압축된 모델의 아키텍처 세부 사항**

**그림 4 | 압축된 8B 학생 모델의 훈련 수렴 그래프. 원본 교사와 수정된 교사로부터의 감독을 비교합니다.**

**표 4 | 증류 기반 재훈련 중 사용된 하이퍼파라미터**


## 분석

우리는 이 최신 모델들의 압축 특성을 더 잘 이해하기 위해 일련의 소거 연구를 수행합니다. 이 섹션에서 우리의 결과를 보고합니다.

### 너비 대 깊이 가지치기

그림 5는 너비와 깊이에 따라 가지치기된 Llama-3.1-Minitron-4B의 학습 곡선을 보여줍니다. 너비 가지치기가 초기 손실이 더 작고, 두 변형이 동일한 수의 매개변수를 가지고 있음에도 불구하고 깊이 가지치기 모델보다 일관되게 뛰어난 성능을 보이는 것을 확인할 수 있습니다.

### 가지치기 및 증류

그림 6은 가지치기와 증류를 통한 우리의 제안된 접근 방식의 직교적 이점을 보여줍니다. 우리는 (1) 임의의 가중치 초기화와 증류, (2) 중요도 점수를 무시하고 임의로 구성 요소가 가지치기되는 경우의 임의 가지치기 및 증류, (3) 일반적인 교차 엔트로피 기반 LM 손실 학습과 함께 우리의 제안된 가지치기, (4) 증류 기반 학습과 함께 우리의 제안된 가지치기를 비교합니다. 우리는 가지치기가 임의의 초기화에 비해 현저히 더 나은 시작점을 제공하며, 증류 기반 학습이 일반적인 학습 방법을 능가하면서도 상당히 적은 학습 토큰(우리의 경우 최대 50배)을 필요로 한다는 것을 확인했습니다.

### 교사 수정

우리는 교사 수정에 대한 두 가지 접근 방식을 비교합니다: (1) 수정된 교사를 가지치기하고 증류하는 방법과 (2) 원본 교사를 가지치기하고 지속적으로 수정된 교사로부터 증류하는 방법. 그림 7의 결과는 교사 수정이 가지치기의 최적성에 영향을 미치지 않으며, 수정된 교사로부터의 증류가 중요하다는 것을 시사합니다. 교사 수정은 증류와 병행하여 수행하여 격차를 줄일 수 있습니다.

### 깊이 가지치기 지표

연속적인 레이어 블록이 제거될 때 LM 검증 손실이 어떻게 증가하는지를 살펴보면(그림 8), 시작과 끝 부분의 레이어가 가장 중요하다는 것을 알 수 있습니다. 비연속적인 레이어를 제거하면 LM 검증 손실이 더 좋아질 수 있습니다(점선). 그러나 이 관찰은 다운스트림 작업 성능을 평가할 때 반드시 적용되지 않습니다. 그림 9에서는 레이어별 중요도에 따라 선택된 16개의 레이어를 제거했을 때 Winogrande 정확도가 0.5로 나타나는 반면, 레이어 16부터 31까지 연속적으로 제거했을 때는 0.595의 정확도가 나타납니다. 이 차이는 증류 기반 재훈련 동안에도 유지되며, 우리는 후자의 접근 방식을 선택합니다.

## 평가

Touvron et al. [19]을 따르는 벤치마크를 통해, 우리는 일련의 다운스트림에서 우리의 압축 모델을 평가합니다.


## Base Models
기본 모델의 평가 결과는 표 1에 나타나 있습니다. 유사한 크기의 모델과 비교하여, MN-Minitron-8B는 전반적으로 우수한 정확도를 보여주며, 최근의 Llama 3.1 8B 모델을 40배 적은 학습 토큰(380B 대 15T)으로 능가합니다. 마찬가지로, Llama-3.1-Minitron 4B 모델은 150배 적은 학습 토큰(94B 대 15T)으로 교사 모델인 Llama 3.1 8B와 비교해 유리한 성능을 보입니다. 또한, 우리의 가지치기된 Llama 모델은 이전 세대의 Minitron 4B 모델을 능가합니다. 표 1에서 보듯이, 폭을 줄인 변형이 깊이를 줄인 변형보다 더 우수합니다. 이러한 결과는 우리의 방법론이 최첨단의 정확도와 학습 효율성의 크기 향상을 함께 제공하는 장점을 명확히 보여줍니다.

## Instruct Models
명령어 조정된 Llama-3.1-Minitron 4B 변형의 성능은 표 2에 나타나 있습니다. 우리는 Llama-3.1-Minitron 4B 변형을 유사한 크기의 기본값과 비교하여, 우리의 모델이 강력한 명령어 수행 능력과 역할 놀이 능력을 보여주며, IFEval [14]와 MT-Bench [15]에서는 Gemma2에 뒤처지지만, 검색 기반 질문 응답(ChatRAG-Bench [16])과 함수 호출(BFCL [17])에서 최첨단 성능을 달성함을 확인할 수 있었습니다.

## Runtime Performance Analysis
우리는 NVIDIA TensorRT-LLM을 사용하여 Llama 3.1 8B 및 Llama-3.1-Minitron 4B 변형을 최적화했으며, 이는 최적화된 LLM 추론을 위한 오픈 소스 툴킷입니다. 이러한 최적화는 MMLU [20], Python 코드 생성용 HumanEval [21], 상식적 추론을 위한 여러 질문 응답 데이터셋(Arc-C [22], HellaSwag [23], TruthfulQA [24], WinoGrande [7]) 및 요약을 위한 XL-Sum English [25] 등 다양한 작업에서 테스트되었습니다. 우리는 MMLU에서 5-shot 성능, Winogrande에서 5-shot, ARC-Challenge에서 25-shot, HellaSwag에서 10-shot, XL-Sum의 20%에서 0-shot 성능을 보고하며, HumanEval 및 MBPP의 pass@1 점수를 평균하여 보고합니다. pass@1 점수 계산에는 온도 0.2 및 핵심 샘플링 [26]을 사용하며, top-p = 0.95를 적용합니다. 명령어 조정 모델의 경우, 우리는 MT-Bench [15], Instruction-Following Eval (IFEval) [14], ChatRAG-Bench [16], 및 Berkeley Function Calling Leaderboard (BFCL) [17]를 사용합니다.


H100 80 GB GPU 하나에서 단일로 학습되었습니다. 서로 다른 사용 사례는 입력 시퀀스 길이/출력 시퀀스 길이 (ISL/OSL) 조합을 통해 표현되며, 배치 크기는 8B-12B 모델에서는 32로, 4B 모델에서는 64로 설정되었습니다. 4B 모델의 더 작은 메모리 풋프린트는 더 큰 배치를 가능하게 합니다.

## 감사의 말

이 작업은 NVIDIA의 많은 사람들의 기여 없이는 불가능했을 것입니다. 몇몇을 언급하자면:

- 기초 모델: Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Pavlo Molchanov, Mostofa Patwary, Daniel Korzekwa, Ashwath Aithal, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz
- 정렬: Ameya Sunil Mahabaleshwarkar, Hayley Ross, Brandon Rowlett, Oluwatobi Olabiyi, Shizhe Diao, Yoshi Suhara
- 데이터셋: Sanjeev Satheesh, Jupinder Parmar, Shengyang Sun, Jiaqi Zeng, Zhilin Wang, Yi Dong, Zihan Liu, Rajarshi Roy, Wei Ping, Makesh Narsimhan Sreedhar, Oleksii Kuchaiev
- TensorRT-LLM: Bobby Chen, James Shen, Chenhan Yu
- Hugging Face 지원: Ao Tang, Yoshi Suhara, Greg Heinrich

## 통찰

이 섹션에서는 흥미롭고 놀라운 관찰 결과를 요약합니다.

### 일반

1. Teacher correction은 새로운, 보지 않은 데이터셋에서 distillation이 최적으로 작동하기 위해 중요합니다. 이 방식으로 distillation에 사용된 데이터셋으로 teacher를 fine-tuning하면 LM validation loss가 6% 이상 감소합니다. Teacher correction은 pruning의 최적성에 영향을 주지 않으며, distillation과 병행하여 수행할 수 있습니다.
2. Minitron 논문의 관찰과 일치하게, 우리는 pruning과 distillation 후에 state-of-the-art 정확성을 달성하기 위해 380B 토큰만 필요합니다.
3. Width pruning의 경우, Attention heads를 유지하고 다른 차원 (MLP intermediate dimension, embedding channels)을 가지치기함으로써 더 강력한 정확성을 달성합니다.

### Mistral NeMo 12B to MN-Minitron-8B

1. 우리 압축 모델은 pruning과 distillation 이후 두 가지 벤치마크, GSM8k와 HumanEval에서 teacher를 능가합니다: GSM8k는 55.7%에서 58.5%로, HumanEval은 23.8%에서 36.2%로 증가합니다. 이 향상은 데이터셋에 의해 영향을 받을 가능성이 큽니다. 그러나 retraining은 distillation loss만을 사용하여 수행됩니다.

### Llama 3.1 8B to Llama-3.1-Minitron 4B

1. Width pruning은 MMLU에서 60.5%의 정확성을 제공하며, depth pruning은 58.7%의 정확성을 제공합니다.
2. Reasoning 능력은 더 크게 영향을 받아, GSM8K의 정확성은 width에서 41.24%, depth에서 16.8%를 기록합니다.
3. Depth pruning은 처리량을 증가시켜 Llama-3.1 8B에 비해 2.7배의 속도 향상을 달성하며, width pruning은 1.7배의 속도 향상을 제공합니다.

