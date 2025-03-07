# Brain-inspired Lp-Convolution: Enhancing Large Kernels and Aligning with the Visual Cortex  
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue)](https://openreview.net/forum?id=0LSAmFCc4p)  

This repository contains the official implementation of **"Brain-inspired Lp-Convolution Benefits Large Kernels and Aligns Better with Visual Cortex"**, accepted at **ICLR 2025**. Our work introduces **Lp-Convolution**, a novel approach that enhances **large kernel convolutional neural networks (CNNs)** by integrating biologically-inspired trainable Gaussian sparsity, improving both **performance and alignment with neural representations in the visual cortex**.

## How it works?
Lp-convolution is new type of convolutional layer, conventional kernels are overlaid with Lp-Mask, channel-wise. Lp-Mask is trainable mask and designed with multivariate p-generalized normal distribution (MPND).

## MPND
The MPND PDF is given by:

$$ 
\text{MPND}(x, y) = \beta e^{-\sum |\mathbf{C} \Delta \mathbf{s}|^p}
 $$

Where:
- $\Delta \mathbf{s} = [x - x_0, y - y_0]$ is the displacement from the center,
- $\mathbf{C}$ is the inverse of covariance matrix,
- $p$ is the power parameter of the $L_p$-norm,
- $\beta$ is a normalization constant (set to 1 in these visualizations).
By varying $p$ and $\mathbf{C}$, Lp-Mask dynamically shapes its contour.

### Visualization
In practice, Lp-mask is trained by standard backpropagation. The trained parameters are $p$ and $\mathbf{C}$, channel-wisely.
Following is examples of changes in mask shapes on varying conditions. Note, $\alpha$, $\gamma$, $\theta$ represent scale, distortion, rotation, which are calcuated from $\mathbf{C}$ using singular value decomposition, for interpretability purpose. 

<table>
  <tr>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_p_only_no_distortion_loop.gif?raw=true" width="180" height="180" alt="P Only No Distortion"></td>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_p_only_with_distortion_loop.gif?raw=true" width="180" height="180" alt="P Only With Distortion"></td>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_p2_distortion_rotation_loop.gif?raw=true" width="180" height="180" alt="P2 Distortion Rotation"></td>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_p16_distortion_rotation_loop.gif?raw=true" width="180" height="180" alt="P16 Distortion Rotation"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_scale_up_90deg_horiz_loop.gif?raw=true" width="180" height="180" alt="Scale Up 90deg Horizontal"></td>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_scale_up_180deg_vert_loop.gif?raw=true" width="180" height="180" alt="Scale Up 180deg Vertical"></td>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_widen_loop.gif?raw=true" width="180" height="180" alt="Widen"></td>
    <td><img src="https://github.com/jeakwon/lpconv/blob/main/assets/mpnd_3d_narrow_loop.gif?raw=true" width="180" height="180" alt="Narrow"></td>
  </tr>
</table>

## Paper  
🔗 **[ICLR 2025 OpenReview Link](https://openreview.net/forum?id=0LSAmFCc4p)**  

📑 **Citation (BibTeX):**  
```bibtex
@inproceedings{kwon2025brain,
  author    = {Jea Kwon and Sungjun Lim and Kyungwoo Song and C. Justin Lee},
  title     = {Brain-inspired Lp-Convolution Benefits Large Kernels and Aligns Better with Visual Cortex},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=0LSAmFCc4p},
  note      = {Accepted at ICLR 2025}
}
```

## Acknowledgement
This code is based on [CycleMLP ICLR 2022](https://github.com/ShoufaChen/CycleMLP), [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

### Install
- PyTorch 1.7.0+ and torchvision 0.8.1+
- [timm](https://github.com/rwightman/pytorch-image-models/tree/c2ba229d995c33aaaf20e00a5686b4dc857044be):
```
pip install 'git+https://github.com/rwightman/pytorch-image-models@c2ba229d995c33aaaf20e00a5686b4dc857044be'

or

git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models
git checkout c2ba229d995c33aaaf20e00a5686b4dc857044be
pip install -e .
```
- fvcore (optional, for FLOPs calculation)
- mmcv, mmdetection, mmsegmentation (optional)

### Data preparation

For tiny imagenet, download http://cs231n.stanford.edu/tiny-imagenet-200.zip. run val_format.py for validation set.


### Training
To train on Tiny ImageNet on a single node with 4 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ACMLP --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save
```
