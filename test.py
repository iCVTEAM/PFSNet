#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.dont_write_bytecode = True
from torchstat import stat
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from net import net


class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                p = self.net(image, shape=shape)
                out   = torch.sigmoid(p[0,0])
                pred  = (out*255).cpu().numpy()
                head  = '../eval/FPN/'+self.model+'/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

if __name__=='__main__':
    torch.cuda.set_device(1)
    for path in ['../data/ECSSD','../data/DUTS-TE','../data/PASCAL-S','../data/DUT-OMRON','../data/HKU-IS']:
	    for model in ['model-50']:
        	t = Test(dataset,net, path,'./model/'+model)
       		t.save()
