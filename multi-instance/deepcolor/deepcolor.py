# created by Liqi Jiang

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from cvppp.train_cvppp import evaluate

import deepcoloring as dc

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net = dc.EUnet(3, 9, 4, 3, 1, depth=3, padding=1, init_xavier=True, use_bn=False, use_dropout=True).to(device)
net.load_state_dict(torch.load("model.t7", map_location=torch.device('cpu')))
net.eval()
print("Model loaded")

from skimage.io import imread
xo = imread("Plant/Ara2013-Canon/ara2013_plant001_rgb.png")[::2,::2]
#xo = imread("../images/plant050_rgb.png")[::2,::2]

x = dc.rgba2rgb()(xo, True)/255.
x = dc.normalize(0.5, 0.5, )(x, True)
x = x.transpose(2, 0, 1)[:, :248, :248]

vx = torch.from_numpy(np.expand_dims(x, 0)).to(device)
p = net(vx)
p_numpy = p.detach().cpu().numpy()[0]
f1, f2 = dc.visualize(xo[:,:,:3],p_numpy,65)
