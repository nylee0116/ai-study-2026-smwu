# Triton-based FlashAttention Implementation & Analysis

**Author** : 이나연 (Sookmyung Women's University, Dept. of AI Engineering)

본 프로젝트는 **<FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness>** 논문의 **Algorithm 1**을 OpenAI Triton 및 Python을 이용해 직접 구현하고, 기존 방식(Naive Attention) 및 PyTorch Native API와의 성능 및 메모리 효율성을 비교 분석한 프로젝트입니다.

> 💡 모든 실험 코드는 `.ipynb` 파일 또는 Google Colab 환경에서 확인하실 수 있습니다.
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](colab.research.google.com/github/nylee0116/ai-study-2026-smwu/blob/main/03_Individual_Project/(개인미니프로젝트)이나연/FlashAttention_이나연.ipynb)

---

## 1. Project Objective
* **Memory 효율성** : O(N^2)의 메모리 복잡도를 가지는 표준 attention 기법의 한계를 이해하고, SRAM을 활용한 Tiling 기법으로 메모리 효율성을 극복하기
* **Triton 커널 구현** : GPU 하드웨어 가속을 위한 Triton 언어를 학습하고, Online Softmax 알고리즘을 커널 수준에서 구현하기
* **성능 분석** : 다양한 sequence(N) 길이에 따른 실행 시간 및 peak memory 점유율 분석

## 2. Implementation
### 2-1. Core Algorithm: Tiling & Online Softmax
표준 attention 기법은 Q와 K를 내적한 행렬을 전부 HBM에 기록해야하지만, 본 구현은 블록 단위로 쪼개어 SRAM에서 연산합니다.
* **Tiling** : BLOCK_M, BLOCK_N 단위로 데이터를 로드하여, HBM상에서의 O(N^2)의 중간 행렬 생성을 방지합니다. 그리고 이를 O(N)의 메모리 접근으로 치환합니다.
* **Online Softmax** : 지수 함수의 합을 계산하기 위해 보정 계수 (alpha, beta)를 이용해 수치 연산을 수행합니다.
    
### 2-2. Triton Kernel Logic
```python
@triton.jit
def flash_attn_kernel(Q, K, V, L, M, Out, ... BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # [Logic Flow]
    # 1. Load Q block from HBM to SRAM
    # 2. Outer loop for K, V blocks
    # 3. Compute S = QK^T, update running max(m) and sum(l)
    # 4. Rescale accumulator(acc) using correction factors (alpha, beta)
    # 5. Store final result back to HBM (O(N) write)
```
    
## 3. Performance Analysis
### 3-1. Benchmark Environment
- GPU : NVIDIA T4 (Colab)
- Data Type : Float16
- Comparison : Naive Attention (표준 attention pytorch 연산) / FlashAttention / PyTorch Native (FlashAttention-2 최적화 버전)
    
### 3-2.Results & Insight
**(1) Execution Time**
<img width="1012" height="642" alt="image" src="https://github.com/user-attachments/assets/b38a2bb5-5d4c-4c34-b8b8-b273e64aa317" />
- sequence 길이가 짧은 구간 (4096 이하) 에서는 Naive attention 방식이 더 빠르나, N이 커질수록 FlashAttention이 더 효율적입니다.
- 이는 FlashAttention에 Recomputation 오버헤드가 존재하기 때문입니다. 중간 값을 저장하지 않는 대신 Online Softmax으로 인해 절대적인 연산량이(FLOPs)이 많기 때문입니다.
- 이에 따라 N이 작은 구간에서는 Naive 방식이 더 유리합니다. 반면, N이 커질수록 메모리 대역폭 한계에 부딪히는 메모리 병목 현상이 발생합니다. 
- 여기서부터는 연산량이 조금 많더라도, HBM 접근을 줄인 FlashAttention이 더 효율적입니다.

**(2) Memory Usage & OOM analysis**
<img width="989" height="607" alt="image" src="https://github.com/user-attachments/assets/57ba3a2c-bdf8-4839-85c3-4c654e9a5e48" />

- sequence 길이가 16384에 도달했을 때, Naive 방식은 Q와 K의 내적 결과를 HBM에 저장하기 위해 약 8GB 메모리 할당을 시도하다 Cuda Out of Memory (OOM) 에러를 발생시켰습니다.
- Execution Time이 다소 안정적으로 보임에도, sequence 길이가 일정 수준을 초과하면 제대로 작동하지 못하는 모습을 보였습니다.
- 반면 본 프로젝트에서 구현한 FlashAttention은 동일한 N에 대해서 수십 MB의 메모리만 점유하며, 안정적인 메모리 효율성을 보이고 있습니다. 연산량 자체로 보면 FlashAttention이 비효율적으로 보이지만, sequence길이가 길어지는 현상에 대해서 더 안정적임을 확인할 수 있는 프로젝트였습니다.

## 4. Conclusion
본 프로젝트를 통해 최신 딥러닝 가속의 핵심인 IO-Awareness의 중요성을 체감할 수 있었습니다.
특히, T4 GPU 환경에서 OOM 현상이 발생한 Naive attention 방식을 대신해, FlashAttention 커널이 긴 토큰도 처리하는 과정에서 HW 최적화의 중요성을 확인했습니다.
이를 통해 알고리즘 설계가 AI 하드웨어 성능을 어떻게 극대화하는지 이해할 수 있는 프로젝트였습니다.

---
## Author
* **이나연 (Lee Na-yeon)** * Sookmyung Women's University, Dept. of AI Engineering
* [GitHub Profile](https://github.com/nylee0116) | [Email](nylee16@sookmyung.ac.kr)
