#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:12:37 2018

@author: caiom
"""

import numpy as np
from torch.utils.data import Dataset
import json
import cv2
import torch
import os.path as osp
import glob

class SegmentationDataset(Dataset):
    """Segmentation dataset loader."""

    def __init__(self, json_folder, img_folder, is_train, class_to_id, resolution = (640, 480), augmentation = False, transform=None):
    #def __init__(self, json_folder, img_folder, is_train, class_to_id, resolution = (1280, 960), augmentation = False, transform=None):
        """
        Args:
            json_folder (str): Path to folder that contains the annotations.
            img_folder (str): Path to all images.
            is_train (bool): Is this a training dataset ?
            augmentation (bool): Do dataset augmentation (crete artificial variance) ?
        """

        self.gt_file_list = glob.glob(osp.join(json_folder, '*.json'))

        self.total_samples = len(self.gt_file_list)
        self.img_folder = img_folder
        self.is_train = is_train
        self.transform = transform
        self.augmentation = augmentation
        self.resolution = resolution
        self.class_to_id = class_to_id
        
        
        # Mean and std are needed because we start from a pre trained net
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        
        gt_file = self.gt_file_list[idx]
        img_number_str = gt_file.split('.')[0].split('/')[-1]
	# Abre Json
        gt_json = json.load(open(gt_file, 'r'))
	# Abre imagem
        img_np = cv2.imread(osp.join(self.img_folder, img_number_str + '.png'), cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
        original_shape = img_np.shape
        img_np = cv2.resize(img_np, (self.resolution[0], self.resolution[1]))[..., ::-1]
        img_np = np.ascontiguousarray(img_np)
	# Cria imagem zerada
        label_np = np.zeros((img_np.shape[0], img_np.shape[1]))
        label_np[...] = -1
        
	# Para todos poligonos
        for shape in gt_json['shapes']:
            # Transforma os pontos do poligono em array
            points_np = np.array(shape['points'], dtype = np.float64)

	    # Ajusta os pontos porque eu mudo o resolucao (pode ignorar)
            points_np[:, 0] *= self.resolution[0]/original_shape[1]
            points_np[:, 1] *= self.resolution[1]/original_shape[0]
	    # As coordenadas dos pontos que formam o poligono tem que ser inteiros
            points_np = np.round(points_np).astype(np.int64)
	    # Coloca os pontos no formato certo para o opencv
            points_np = points_np.reshape((-1,1,2))
	    # Pinta o poligono usando o opencv com o valor referente ao label
            label_np = cv2.fillPoly(label_np, [points_np], self.class_to_id[shape['label']])

        # Transforma o GT em inteiro    
        label_np = label_np.astype(np.int32)
        
        if self.is_train and self.augmentation:
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np)
                label_np = np.fliplr(label_np)
                img_np = np.ascontiguousarray(img_np)
                label_np = np.ascontiguousarray(label_np)
        
        img_pt = img_np.astype(np.float32) / 255.0
        for i in range(3):
            img_pt[..., i] -= self.mean[i]
            img_pt[..., i] /= self.std[i]
            
        img_pt = img_pt.transpose(2,0,1)
            
        img_pt = torch.from_numpy(img_pt)
        label_pt = torch.from_numpy(label_np).long()
        #print(img_number_str, img_pt.shape)

        sample = {'image': img_pt, 'gt': label_pt, 'image_original': img_np}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
