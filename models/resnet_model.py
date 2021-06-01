from PIL import Image
import cv2, os
import comman
import torch
from torch import nn
from torch.autograd import Variable
from typing import Dict, Callable, Iterable, Optional, Type

from models.base_model import BaseModel
from models.networks import *
from comman import *


class ResNetModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.input_imgs: torch.Tensor = None
        self.output: torch.Tensor = None
        self.predicted: torch.Tensor = None
        self.model_names = ['G']
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                             opt.dropout_rate).to(self.device)

        self.criteria: Dict[str, Callable] = {'BCE': torch.nn.BCELoss(), 'cross': torch.nn.CrossEntropyLoss()}

        if opt.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        else:
            self.optimizer_G = None

        # self.netG = init_net(self.netG)

    def set_input(self, input):
        self.input_imgs = input['A'].float().to(self.device)
        if not self.opt.test:
            self.output = input['B'].float().to(self.device)

    def forward(self):
        self.predicted = self.netG(self.input_imgs)

    def get_outputs(self):
        predicted_label = torch.tensor(self.predicted > 0.5, dtype=float).to(self.device)
        return predicted_label

    def compute_losses(self):
        self.losses = {}
        self.losses['BCE'] = self.criteria['BCE'](self.predicted, self.output)
        # print(self.predicted, torch.softmax(self.predicted.data, 1))
        # _, predicted_label = torch.max(torch.softmax(self.predicted.data, 1), 1)
        predicted_label = torch.tensor(self.predicted > 0.5, dtype=float).to(self.device)
        self.losses['accuracy'] = (self.output == predicted_label).float().mean()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.compute_losses()
        self.losses['BCE'].backward()
        self.optimizer_G.step()
