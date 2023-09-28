# Welcome to Our Project Repository!

Thank you for visiting our code repository for Lp-Convolution! We are excited to share our work with the community and welcome any feedback or questions you might have.

Please note that we are currently in the process of organizing and refining this repository to ensure that it is as user-friendly and informative as possible. We apologize for any inconvenience this may cause and appreciate your understanding and patience as we work to improve the repository.

We are actively working on updating and optimizing the codebase, documentation, and other resources, and we aim to have a more organized version available as soon as possible.

In the meantime, if you have any questions, concerns, or need clarification on any aspect of the project, please do not hesitate to raise an issue or contact us directly. We value your input and are committed to addressing any inquiries promptly.

Thank you once again for your interest in our work, and we look forward to your valuable feedback!

Best regards, 
Authors


### Acknowledgement
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
