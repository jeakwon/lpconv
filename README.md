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

Download and extract ImageNet-1k train and val images from http://image-net.org/.
(For tiny imagenet, download http://cs231n.stanford.edu/tiny-imagenet-200.zip.
and additionally run val_format.py for validation set.)
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Training
To train on Tiny ImageNet on a single node with 4 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model ACMLP --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save
```
