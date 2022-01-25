import argparse
import os
import time

from tensorboardX import SummaryWriter
import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

import models
from helpers import dataset
from helpers.trainer import Trainer
from helpers.utils import (
    check_dirs_exist,
    accuracy,
    set_seeds,
    Logger)

parser = argparse.ArgumentParser(description="Initial Train Process")
parser.add_argument('--n-epochs', default=200, type=int)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.1, 0.1, 0.1])
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--dev-idx', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--distributed', action='store_true', help='distributed training')

args = parser.parse_args()

if args.distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = f'saves/logs.txt'


class InitialModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_entropy = nn.CrossEntropyLoss()

    def _get_loss_and_backward(self, batch):
        input, target = batch
        with autocast():
            logit = self.model(input)
            loss = self.cross_entropy(logit, target)
        self.scaler.scale(loss).backward()

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
    train_loader, eval_loader, num_classes, sampler = dataset.__dict__[args.dataset](args.batch_size, num_workers=args.num_workers, distributed=args.distributed)
    if args.dataset == 'imagenet':
        model = models.__dict__[args.model](pretrained=True, num_classes=num_classes)
    else:
        model = models.__dict__[args.model](num_classes=num_classes)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    base_trainer_cfg = (args, model, train_loader, eval_loader, optimizer, args.save_dir, device, logger)
    
    if not args.distributed or local_rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
        
    trainer = InitialModelTrainer(*base_trainer_cfg, sampler=sampler, writer=writer)
    
    trainer.model = trainer.model.to(device)
    if args.distributed:
        # trainer.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)
        trainer.model = DDP(trainer.model, device_ids=[local_rank], output_device=local_rank)
    
    logger.log('\n'.join(map(str, vars(args).items())))
    trainer.train()
    print(f'Log Path : {args.log_path}')


if __name__ == '__main__':
    main()
