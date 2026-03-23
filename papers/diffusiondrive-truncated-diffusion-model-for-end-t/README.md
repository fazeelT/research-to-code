# DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving

## Paper Information
- **Source**: semantic-scholar
- **URL**: https://www.semanticscholar.org/paper/71d62a6c0b84604a376174cf728e36d6f3c23f1e
- **Authors**: Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, Xinggang Wang
- **Year**: 2024

## Abstract
Recently, the diffusion model has emerged as a powerful generative technique for robotic policy learning, capable of modeling multi-mode action distributions. Leveraging its capability for end-to-end autonomous driving is a promising direction. However, the numerous denoising steps in the robotic diffusion policy and the more dynamic, open-world nature of traffic scenes pose substantial challenges for generating diverse driving actions at a real-time speed. To address these challenges, we propose a novel truncated diffusion policy that incorporates prior multi-mode anchors and truncates the diffusion schedule, enabling the model to learn denoising from anchored Gaussian distribution to the multi-mode driving action distribution. Additionally, we design an efficient cascade diffusion decoder for enhanced interaction with conditional scene context. The proposed model, DiffusionDrive, demonstrates 10× reduction in denoising steps compared to vanilla diffusion policy, delivering superior diversity and quality in just 2 steps. On the planning-oriented NAVSIM dataset, with aligned ResNet-34 backbone, DiffusionDrive achieves 88.1 PDMS without bells and whistles, setting a new record, while running at a real-time speed of 45 FPS on an NVIDIA 4090. Qualitative results on challenging scenarios further confirm that DiffusionDrive can robustly generate diverse plausible driving actions.

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
@article{diffusiondrive-truncated-diffusion-model-for-end-t,
  title={DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving},
  author={Bencheng Liao and Shaoyu Chen and Haoran Yin and Bo Jiang and Cheng Wang and Sixu Yan and Xinbang Zhang and Xiangyu Li and Ying Zhang and Qian Zhang and Xinggang Wang},
  journal={semantic-scholar},
  year={2024},
  url={https://www.semanticscholar.org/paper/71d62a6c0b84604a376174cf728e36d6f3c23f1e}
}
```

## Notes
- This is a reference implementation for educational purposes
- TODO: Add performance benchmarks
- TODO: Add additional experiments
