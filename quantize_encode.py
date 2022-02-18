import argparse
import json
import os
from pathlib import Path
import time

import comet_ml
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from helpers.utils import (
    check_dirs_exist,
    accuracy,
    print_nonzeros,
    set_seeds,
    load_model,
    Logger
)
from helpers import dataset
import models
from helpers.trainer import Trainer
from helpers.quantizer import PostQuantizer
from helpers.encoder import HuffmanEncoder


def get_args():
    parser = argparse.ArgumentParser(description="Quantize Process")
    parser.add_argument('--n-epochs', default=20, type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--model', type=str, default='resnet56')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--load-path', type=str, default='None')
    parser.add_argument('--quan-mode', type=str, default='all-quan')  # pattern: "(all|conv|fc)-quan"
    parser.add_argument('--quan-bits', type=int, default='None')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
    parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.1, 0.1, 0.1])
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--log-name', type=str, default='logs.txt')  # The name of the log file
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--distributed', action='store_true', help='distributed training')

    return parser.parse_args()
    
args = get_args()

if args.distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = f'saves/{args.log_name}'
args.quan_model_path = f'{args.save_dir}/model_best.pt'


class QuantizedModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_entropy = nn.CrossEntropyLoss()

        quantizer = PostQuantizer(self.args.quan_mode, device=self.device)
        quantizer.quantize(self.model, self.args.quan_bits)
        self.quan_dict = quantizer.get_quan_dict()

        self.mask = dict()

    def _set_quan_weight_grad(self):
        for name, module in self.model.named_modules():
            if name in self.quan_dict:
                if name not in self.mask:
                    self.mask[name] = dict()
                mask = self.mask[name]
                weight = module.weight.data
                grad = module.weight.grad.data

                # Mask gradients of pruend weights
                key = 'grad'
                if key not in mask:
                    mask[key] = torch.where(weight == 0, 0, 1)
                grad *= mask[key]

                # Set gradients of quantized weights
                quan_labels = self.quan_dict[name]
                quan_range = quan_labels.max().item() + 1
                key = 'ind'
                if key not in mask:
                    mask[key] = dict()
                for i in range(quan_range):
                    if i not in mask[key]:
                        mask[key][i] = torch.where(quan_labels == i)
                    group_ind = mask[key][i]
                    group_grad_sum = torch.sum(grad[group_ind])
                    grad[group_ind] = group_grad_sum
                module.weight.grad.data = grad

    def _get_loss_and_backward(self, batch):
        input, target = batch
        with autocast():
            logit = self.model(input)
            loss = self.cross_entropy(logit, target)
        self.scaler.scale(loss).backward()

        self._set_quan_weight_grad()
        
        top1, top5 = accuracy(logit, target, topk=(1, 5))
        if self.writer is not None:
            self.writer.add_scalars(
                'data/scalar_group', {
                    'total_loss': loss.item(),
                    'lr': self.cur_lr,
                    'top1': top1,
                    'top5': top5
                }, self.global_step
            )
        return loss, top1, top5

    def _evaluate(self, batch):
        input, target = batch
        logit = self.model(input)
        loss = self.cross_entropy(logit, target)
        top1, top5 = accuracy(logit, target, topk=(1, 5))
        return {'loss': loss.item(), 'top1': top1.item(), 'top5': top5.item()}


class Evaluator(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_entropy = nn.CrossEntropyLoss()

    def _get_loss_and_backward(self, _):
        pass

    def _evaluate(self, batch):
        input, target = batch
        logit = self.model(input)
        loss = self.cross_entropy(logit, target)
        top1, top5 = accuracy(logit, target, topk=(1, 5))
        return {'loss': loss.item(), 'top1': top1.item(), 'top5': top5.item()}


def main():
    set_seeds(args.seed)
    check_dirs_exist([args.save_dir])
    logger = Logger(args.log_path, distributed=args.distributed)
    if args.distributed:
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda')
        
    if args.dataset not in dataset.__dict__:
        raise NameError
    if args.model not in models.__dict__:
        raise NameError

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers+ngpus_per_node-1) / ngpus_per_node)

    logger.log_line()
    logger.log('\n'.join(map(str, vars(args).items())))
    train_loader, eval_loader, num_classes, sampler = dataset.__dict__[args.dataset](args.batch_size, distributed=args.distributed)

    # Quantize and quantize retrain
    model = models.__dict__[args.model](num_classes=num_classes)
    load_model(model, args.load_path, logger, device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )

    model = model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if not args.distributed or local_rank == 0:
        file = Path('./comet_config.json')
        if file.exists():
            comet_config = json.load(file.open())
            writer = SummaryWriter(log_dir=args.log_dir, comet_config=comet_config)
        else:
            writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    base_trainer_cfg = (args, model, train_loader, eval_loader, optimizer, args.save_dir, device, logger)
    trainer = QuantizedModelTrainer(*base_trainer_cfg, sampler=sampler, writer=writer)
    trainer.train()

    if not args.distributed or local_rank == 0:
        # Huffman encode and decode
        enc_model = models.__dict__[args.model](num_classes=num_classes)
        load_model(enc_model, args.quan_model_path, logger, device)
        enc_model.to(device)
        print('Before Huffman coding')
        base_cfg = (args, enc_model, None, eval_loader, None, args.save_dir, device, logger)
        evaluator = Evaluator(*base_cfg)
        evaluator.eval()

        encoder = HuffmanEncoder(logger)
        encoder.huffman_encode_model(enc_model)

        dec_model = models.__dict__[args.model](num_classes=num_classes)
        dec_model.to(device)
        encoder.huffman_decode_model(dec_model)

        print('After Huffman coding')
        base_cfg = (args, dec_model, None, eval_loader, None, args.save_dir, device, logger)
        evaluator = Evaluator(*base_cfg)
        evaluator.eval()

    logger.log(print_nonzeros(trainer.model))

if __name__ == '__main__':
    main()
