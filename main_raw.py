import argparse
from torchvision.models import resnet, ResNet50_Weights
import os
import time
import torch
from datetime import datetime
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence

import model
from data import build_vocab, get_flickr_data, get_iterator
from utils import setup_logging, adjust_optimizer, AverageMeter, select_optimizer
from model import CaptionModel
from torchvision.models import resnet

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Flickr8k caption genration training')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--cnn', '-a', metavar='CNN', default='resnet50',
                    choices=model_names,
                    help='cnn feature extraction architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--embedding_size', default=256, type=int,
                    help='size of word embedding used')
parser.add_argument('--rnn_size', default=256, type=int,
                    help='size of rnn hidden layer')
parser.add_argument('--num_layers', default=2, type=int,
                    help='number of rnn layers to use')
parser.add_argument('--max_length', default=30, type=int,
                    help='maximum time length to feed')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--finetune_epoch', default=1, type=int,
                    help='epoch to start cnn finetune')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-eb', '--eval_batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--grad_clip', default=5., type=float,
                    help='gradient max norm')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', '--learning_rate_decay', default=0.8, type=float,
                    metavar='LR', help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--share_weights', default=False, type=bool,
                    help='share embedder and classifier weights')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')


def main():
    global args
    args = parser.parse_args()
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    checkpoint_file = os.path.join(save_path, 'checkpoint_epoch_%s.pth.tar')

    logging.debug("run arguments: %s", args)
    logging.info("using pretrained cnn %s", args.cnn)
    if args.cnn == 'resnet50':
        cnn = resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    vocab = build_vocab()
    model = CaptionModel(cnn, vocab,
                         embedding_size=args.embedding_size,
                         rnn_size=args.rnn_size,
                         num_layers=args.num_layers,
                         share_embedding_weights=args.share_weights)

    train_data = get_iterator(get_flickr_data(vocab, train=True),
                              batch_size=args.batch_size,
                              max_length=args.max_length,
                              shuffle=True,
                              num_workers=args.workers)
    val_data = get_iterator(get_flickr_data(vocab, train=False),
                            batch_size=args.eval_batch_size,
                            max_length=args.max_length,
                            shuffle=False,
                            num_workers=args.workers)

    optimizer = select_optimizer(
        args.optimizer, params=model.parameters(), lr=args.lr)
    regime = lambda e: {'lr': args.lr * (args.lr_decay ** e),
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay}
    model.finetune_cnn(False)

    def forward(model, data, training=True, optimizer=None):
        import numpy as np  # 若需判断numpy数组，需导入

        use_cuda = False  # 禁用GPU，保持不变
        loss_fn = nn.CrossEntropyLoss()
        perplexity = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train() if training else model.eval()

        end = time.time()
        for i, (imgs, (captions, lengths)) in enumerate(data):
            data_time.update(time.time() - end)

            # 1. 处理图像张量（浮点型，训练时需梯度）
            if training:
                if isinstance(imgs, np.ndarray):
                    imgs = torch.from_numpy(imgs).float().requires_grad_(True)
                else:
                    imgs = imgs.clone().requires_grad_(True)
            else:
                with torch.no_grad():
                    if isinstance(imgs, np.ndarray):
                        imgs = torch.from_numpy(imgs).float().detach()
                    else:
                        imgs = imgs.clone().detach()

            # 2. 处理字幕张量（整数型，无需梯度）
            if training:
                if isinstance(captions, np.ndarray):
                    captions = torch.from_numpy(captions).long().clone()
                else:
                    captions = captions.clone()
            else:
                with torch.no_grad():
                    if isinstance(captions, np.ndarray):
                        captions = torch.from_numpy(captions).long().detach()
                    else:
                        captions = captions.clone().detach()

            # 3. 输入与目标处理（不变）
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths)[0]

            # 4. 模型前向传播与损失计算（不变）
            pred, _ = model(imgs, input_captions, lengths)
            err = loss_fn(pred, target_captions)

            # 5. 更新困惑度（已修复IndexError）
            perplexity.update(math.exp(err.item()))

            # 6. 训练阶段参数更新（不变）
            if training:
                optimizer.zero_grad()
                err.backward()
                clip_grad_norm(model.rnn.parameters(), args.grad_clip)
                optimizer.step()

            # 7. 时间更新与日志打印（不变）
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Perplexity {perp.val:.4f} ({perp.avg:.4f})'.format(
                    epoch, i, len(data),
                    phase='TRAINING' if training else 'EVALUATING',
                    batch_time=batch_time,
                    data_time=data_time, perp=perplexity))

        return perplexity.avg

    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= args.finetune_epoch:
            model.finetune_cnn(True)
        optimizer = adjust_optimizer(
            optimizer, epoch, regime)
        # Train
        train_perp = forward(
            model, train_data, training=True, optimizer=optimizer)
        # Evaluate
        val_perp = forward(model, val_data, training=False)

        logging.info('\n Epoch: {0}\t'
                     'Training Perplexity {train_perp:.4f} \t'
                     'Validation Perplexity {val_perp:.4f} \n'
                     .format(epoch + 1, train_perp=train_perp, val_perp=val_perp))
        model.save_checkpoint(checkpoint_file % (epoch + 1))


if __name__ == '__main__':
    main()
