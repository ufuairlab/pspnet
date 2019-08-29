#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 19:46:01 2018

@author: caiom
"""

import os.path as osp
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import vgg_unet
import psp_net
import glob
import torch
import cv2

img_folder = 'Test'
model_path = 'model_psp.pth'
save_dir = 'result_psp/'

#resolution = (640, 480)
resolution = (1280, 960)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Color in RGB
class_to_color = {'Ground': (127, 0, 0) , 'Healthy': (0, 127, 127), 'Pest': (0, 255, 0)}
class_to_id = {'Ground': 0, 'Healthy': 1, 'Pest': 2}
id_to_class = {v: k for k, v in class_to_id.items()}
nClasses = 3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


model = psp_net.PSPNet(nClasses)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

img_list = glob.glob(osp.join(img_folder, '*.png'))

for img_path in img_list:

        img_np = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
        img_np = cv2.resize(img_np, (resolution[0], resolution[1]))[..., ::-1]
        img_np = np.ascontiguousarray(img_np)
        
        img_pt = np.copy(img_np).astype(np.float32) / 255.0
        for i in range(3):
            img_pt[..., i] -= mean[i]
            img_pt[..., i] /= std[i]
            
        img_pt = img_pt.transpose(2,0,1)
            
        img_pt = torch.from_numpy(img_pt[None, ...]).to(device)
        
        label_out = model(img_pt)
        label_out = torch.nn.functional.softmax(label_out, dim = 1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)
        
        labels = np.argmax(label_out, axis=0)
        
        color_label = np.zeros((resolution[1], resolution[0], 3))
            
        for key, val in id_to_class.items():
            color_label[labels == key] = class_to_color[val]
            
        plt.figure()
        plt.imshow((img_np/255) * 0.5 + (color_label/255) * 0.5)
        plt.savefig(save_dir + "IMG" + ".png")
        plt.show()
        
        plt.figure()
        plt.imshow(color_label.astype(np.uint8))
        plt.savefig(save_dir + "GT" + ".png")
        plt.show()