#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net import net
from apex import amp
Act = nn.ReLU
import math
import random
import numpy as np

def structure_loss(pred, mask):

    wbce  = F.binary_cross_entropy_with_logits(pred, mask)
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()#

def dlr(epoch,totepoch):
    if epoch <= totepoch:
        return (1-abs((epoch+1)/(totepoch+1)*2-1))
    else:
        return 1.0/17

def train(Dataset, Network):
    ## Set random seeds
    seed = 4
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg    = Dataset.Config(datapath='../data/DUTS-TR',snapshot =None,savepath='./out',mode='train', batch=20, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)

    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name :
            print(name)
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = dlr(epoch,cfg.epoch)*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = dlr(epoch,cfg.epoch)*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            p= net(image)	
            loss = structure_loss(p, mask)

            optimizer.zero_grad() 
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            ## log

            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss':loss.item()}, global_step=global_step)

            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if epoch>=cfg.epoch-10:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))

if __name__=='__main__':
    torch.cuda.set_device(1)
    train(dataset, net)
