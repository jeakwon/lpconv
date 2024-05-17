import numpy as np
import os.path as osp
import datetime, time
from collections import OrderedDict

import wandb
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F

from my_dassl.modeling import build_backbone
from my_dassl.optim import build_optimizer, build_lr_scheduler
from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.utils import (MetricMeter, AverageMeter, set_random_seed, count_num_param)
from my_dassl.metrics import compute_accuracy


class ConvNextV2(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )        
        ## convnext v2 -----------------------------
        if self.backbone.out_features is None:
            fdim = self.backbone.head.in_features
        else:
            fdim = self.backbone.out_features
        self.backbone.head = nn.Linear(fdim, num_classes)
        ## -----------------------------------------

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x):
        y = self.backbone(x)
        return y



@TRAINER_REGISTRY.register()
class CONVNET(TrainerX):    
    def __init__(self, cfg):
        super().__init__(cfg)


    def build_model(self):
        cfg = self.cfg

        print("Building model")
        self.model = ConvNextV2(cfg, cfg.MODEL, self.num_classes)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        """Generic training loops."""
        self.before_train()
        set_random_seed(self.cfg.SEED)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train() 


    def before_epoch(self):
        pass

    
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        self.total_length = self.num_batches * self.max_epoch
        self.warmup_length = self.total_length * 0.1
        end = time.time()

        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label



    def forward_backward(self, batch):
        self.optim.zero_grad()
        ## Training Phase
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, label)[0].item(),
        }

        if self.cfg.use_wandb:
            wandb.log({'tr_loss (batch)': loss_summary["loss"],
                    'tr_acc (batch)' : loss_summary["acc_train"]})

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary



    def after_train(self):
        print("Finish training")
        # all_last_acc = self.test()
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()



    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        curr_result = 0.0
        if self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best.pth.tar"
                )
        else:
            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

        if self.cfg.use_wandb:
            wandb.log({'val_acc':curr_result})