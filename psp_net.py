#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:08:41 2018

@author: Alex
"""
import torch
import torchvision
import numpy as np
import utils2


class PSPDec(torch.nn.Module):
	def __init__(self, in_dim, reduction_dim, setting):
		super(PSPDec, self).__init__()
		self.features = []
		for s in setting:
			self.features.append(torch.nn.Sequential(
				torch.nn.AdaptiveAvgPool2d(s),
				torch.nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
				#torch.nn.BatchNorm2d(reduction_dim, momentum=.95),
				#torch.nn.InstanceNorm2d(reduction_dim, momentum=.95),
				torch.nn.ReLU(inplace=True)
			))
		self.features = torch.nn.ModuleList(self.features)


	def forward(self, x):
		x_size = x.size()
		out = [x]
		for f in self.features:
			out.append(torch.nn.functional.upsample(f(x), x_size[2:], mode='bilinear'))
		out = torch.cat(out, 1)
		return out


class PSPNet(torch.nn.Module):

	def __init__(self, num_classes):
		super(PSPNet, self).__init__()

		resnet = torchvision.models.resnet101(pretrained=True)
		#print('resnet', resnet)

		self.layer0 = torch.nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
			resnet.maxpool
		)
		
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
		

		for n, m in self.layer3.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)
		for n, m in self.layer4.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)


		self.ppm = PSPDec(2048, 512, (1, 2, 3, 6))

		self.final = torch.nn.Sequential(
			torch.nn.Conv2d(4096, 512, 3, padding=1, bias=False),
			torch.nn.BatchNorm2d(512, momentum=.95),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(.1),
			torch.nn.Conv2d(512, num_classes, 1),
		)


	def forward(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x = self.ppm(x)
		x = self.final(x)
		return torch.nn.functional.upsample(x, (480, 640), mode='bilinear')


	@staticmethod
	def eval_net_with_loss(model, inp, gt, class_weights, device):

		weights = torch.from_numpy(np.array(class_weights, dtype=np.float32)).to(device)
		out = model(inp)

		softmax = torch.nn.functional.log_softmax(out, dim = 1)
		loss = torch.nn.functional.nll_loss(softmax, gt, ignore_index=-1, weight=weights)

		return (out, loss)
