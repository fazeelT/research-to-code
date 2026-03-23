# Efficient Memory Management for Large Language Model Serving with PagedAttention

## Paper Information
- **Source**: semantic-scholar
- **URL**: https://www.semanticscholar.org/paper/83b90f4a0ae4cc214eb3cc140ccfef9cd99fac05
- **Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, Ion Stoica
- **Year**: 2023

## Abstract
High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2--4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM's source code is publicly available at https://github.com/vllm-project/vllm.

## Implementation
This folder contains an implementation of the paper above.

### Files
- `implementation.py` - Python implementation using PyTorch
- `implementation.cu` - CUDA implementation (if applicable)
- `requirements.txt` - Python dependencies

### Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run implementation
python implementation.py
```

## Citation
```bibtex
@article{efficient-memory-management-for-large-language-mod,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Haotong Zhang and Ion Stoica},
  journal={semantic-scholar},
  year={2023},
  url={https://www.semanticscholar.org/paper/83b90f4a0ae4cc214eb3cc140ccfef9cd99fac05}
}
```

## Notes
- This is a reference implementation for educational purposes
- TODO: Add performance benchmarks
- TODO: Add additional experiments
