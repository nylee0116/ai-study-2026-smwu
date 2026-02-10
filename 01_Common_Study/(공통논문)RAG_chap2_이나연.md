# [2026-1 Winter] AI Study

# 🎯 Common Study (공통논문)
논문 : Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)   
파트 : 2.2 Retriever (DPR) ~ 2.5 Decoding   
발표자 : 이나연 (nylee0116)   

**[Full PDF]((공통논문)RAG_chap2_이나연.pdf)에서 더 자세한 내용을 확인하실 수 있습니다.**

## 1. 스터디 개요
- **RAG(Retrieval-Augmented Generation)** 는 **정보 검색(Retrieval)** 과 **생성 모델(Generation)** 을 결합한 하이브리드 모델입니다.
- **Parametric Memory**: 모델 내부 가중치에 저장된 지식 (예: GPT, BART). 꺼내 쓰기는 빠르지만 최신 정보 반영이 어렵습니다.
- **Non-parametric Memory**: 외부 데이터베이스(예: Wikipedia, 검색DB)에 저장된 지식. 최신 정보를 반영하고 명시적으로 참조하는 데 효율적입니다. 

<img width="1326" height="412" alt="rag_architecture" src="https://github.com/user-attachments/assets/b2eb8c69-a83b-4799-aee6-5909b5d4839c" />

## 💡 주요 개념 정리

* **Marginalize(주변화)**: 특정 문서 하나에 의존하지 않고, 여러 문서가 정답에 기여할 모든 확률을 수학적으로 '합산'하는 과정을 뜻합니다.
* **End-to-End Fine-tuning**: 답변이 틀렸을 때 생성 모델뿐만 아니라 문서를 가져온 Retriever에게도 책임을 물어 동시에 업데이트하는 방식을 뜻합니다.

---

## 2. 모델 구성 요소 (Models)

### 2.2 Retriever: DPR (Dense Passage Retrieval)
질문에 적절한 문서를 찾아내는 역할을 수행하며, **Bi-encoder** 구조를 사용합니다.
* bi-encoder는 context encoder와 candidate encoder로 구성되었으며 각각 context query와 candidate document들을 encoding합니다.

* **핵심 수식**: $p_{\eta}(z|x) \propto \exp(d(z)^{\top}q(x))$
* $d(z) = \text{BERT}_d(z), \quad q(x) = \text{BERT}_q(x)$
* **상세 설명**:
    * **$d(z)$ (Document Encoder)**: 문서를 밀집 벡터로 변환 (BERT 기반).
    * **$q(x)$ (Query Encoder)**: 사용자 질문을 벡터로 변환 (BERT 기반).
    * **MIPS (Maximum Inner Product Search)**: 수백만 개의 문서 중 유사도가 높은 문서를 빠르게 검색.

### 2.3 Generator: BART
검색된 문서를 참고하여 최종 답변을 생성합니다.

* **핵심 수식**: $p_{\theta}(y_i | x, z, y_{1:i-1})$
* **모델 구조**: 400M 파라미터의 **BART-large** (seq2seq transformer)를 활용.
* **입력 방식**: 원문 질문 $x$와 검색된 문서 $z$를 단순히 **결합(Concatenate)** 하여 입력.

---

## 3. 학습 및 디코딩 (Training & Decoding)

### 2.4 Training
Retriever와 Generator를 직접적인 가이드 없이 동시에 학습(Joint Training)시키는 **End-to-End** 방식을 취합니다.

* **학습 목표**: 각 타겟의 **Negative Marginal Log-likelihood**를 최소화.
  $$\sum_{i} -\log p(y_j|x_j)$$.
* **효율화 전략**: 연산 비용을 줄이기 위해 문서 인코더($BERT_d$)와 인덱스는 고정하고, **질문 인코더($BERT_q$)와 생성 모델만 파인튜닝**함.

### 2.5 Decoding
RAG-Sequence와 RAG-Token 모델은 테스트 시 서로 다른 방식으로 정답을 도출합니다.

| 모델 | 디코딩 방식 | 특징 |
| :--- | :--- | :--- |
| **RAG-Token** | Standard Beam Search | 각 토큰 생성 시마다 다른 문서를 참조하여 확률 합산. |
| **RAG-Sequence** | Thorough / Fast Decoding | 하나의 문서를 끝까지 참조하여 문장 전체를 생성 후 확률 합산. |

* **Thorough Decoding**: 모든 문서에 대해 확률을 정밀하게 계산하는 방식.
* **Fast Decoding**: 효율성을 위해 생성되지 않은 가설의 확률을 0으로 간주하는 근사법.

---

