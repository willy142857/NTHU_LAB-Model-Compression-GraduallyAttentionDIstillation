import os
import numpy as np
from tqdm import tqdm
from abc import abstractmethod

from helpers.utils import (
    get_average_meters,
    save_model
)

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, args, model, train_loader, eval_loader, optimizer, save_dir, device, logger, sampler=None, writer=None):
        self.args = args
        self.model = model
        self.train_loader = train_loader  # Train data loader
        self.eval_loader = eval_loader  # Eval data loader
        self.sampler = sampler
        self.optimizer = optimizer
        self.scaler = GradScaler()
        self.save_dir = save_dir
        self.device = device  # Device name
        self.logger = logger
        self.writer = writer
        self.cur_epoch = None
        self.cur_lr = None
        self.global_step = None

    def _get_save_model_path(self):
        return os.path.join(self.save_dir, f'model_best.pt')

    def _train_epoch(self):
        self.model.train()  # Train mode
        e_loss, e_top1, e_top5 = get_average_meters(n=3)
        self._adjust_learning_rate()
        text = str()
        is_sub_process = self.args.distributed and dist.get_rank() != 0
        with tqdm(self.train_loader, disable=is_sub_process) as iter_bar:
            for batch in iter_bar:
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                with autocast():
                    b_loss, b_top1, b_top5 = self._get_loss_and_backward(batch)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.global_step += 1
                e_loss.update(b_loss.item(), len(batch))
                e_top1.update(b_top1.item(), len(batch))
                e_top5.update(b_top5.item(), len(batch))
                text = f'Iter (loss={e_loss.mean:5.3f} | top1={e_top1.mean:5.3} | top5={e_top5.mean:5.3})'
                iter_bar.set_description(text)
                
        text = f'[ Epoch {self.cur_epoch} (Train) ] : {text}'
        self.logger.log(text, verbose=True)

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()  # Evaluation mode
        e_vals = None  # Epoch result array
        b_dict = None  # Batch result dict
        for batch in tqdm(self.eval_loader, desc='Test'):
            batch = [t.to(self.device) for t in batch]
            b_dict = self._evaluate(batch)  # Accuracy to print
            b_vals = np.array(list(b_dict.values()))
            if e_vals is None:
                e_vals = [0] * len(b_vals)
            e_vals += b_vals
            
        e_dict = dict(zip(b_dict.keys(), e_vals/len(self.eval_loader)))
        text = f'[ Epoch {self.cur_epoch} (Test) ] : {e_dict}'
        self.logger.log(text, verbose=True)
        if self.writer is not None:
            self.writer.add_scalars('eval/scalar_group', e_dict, self.cur_epoch)
        return e_dict

    def _adjust_learning_rate(self):
        if self.cur_epoch in self.args.schedule:
            i = self.args.schedule.index(self.cur_epoch)
            self.cur_lr *= self.args.lr_drops[i]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cur_lr

    @abstractmethod
    def _get_loss_and_backward(self, batch):
        return NotImplementedError

    @abstractmethod
    def _evaluate(self, batch):
        return NotImplementedError

    def train(self):
        """ Train Loop """
        self.model.train()  # Train mode
        best_top1 = 0.
        self.cur_lr = self.args.lr
        self.global_step = 0
        for epoch in range(self.args.n_epochs):
            self.cur_epoch = epoch
            if self.args.distributed or self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self._train_epoch()
            eval_result = self._eval_epoch()
            if best_top1 < eval_result['top1']:
                best_top1 = eval_result['top1']
                if self.writer is not None:
                    self.writer.add_text(f'top1: {best_top1}', self.cur_epoch)
                save_model(self.model, self._get_save_model_path(), self.logger, distributed=self.args.distributed)

    def eval(self):
        """ Evaluation Loop """
        self.model.eval()  # Evaluation mode
        self._eval_epoch()


