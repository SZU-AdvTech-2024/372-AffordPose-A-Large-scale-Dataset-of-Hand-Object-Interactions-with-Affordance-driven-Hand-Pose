
# -*- coding: utf-8 -*-


import numpy as np

import torch
import torch.nn as nn
from mano.lbs import batch_rodrigues
from collections import namedtuple

model_output = namedtuple('output', ['vertices', 'global_orient', 'transl'])

class ObjectModel(nn.Module):

    def __init__(self,
                 v_template,
                 batch_size=1,
                 dtype=torch.float32):

        super(ObjectModel, self).__init__()


        self.dtype = dtype

        # Mean template vertices
        v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)
        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        transl = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))

        global_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

        self.batch_size = batch_size


    def forward(self, global_orient=None, transl=None, v_template=None, **kwargs):
        

        if global_orient is None:
            global_orient = self.global_orient
        if transl is None:
            transl = self.transl
        if v_template is None:
            v_template = self.v_template

        rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([self.batch_size, 3, 3])

        vertices = torch.matmul(v_template, rot_mats) + transl.unsqueeze(dim=1)

        output = model_output(vertices=vertices,
                              global_orient=global_orient,
                              transl=transl)

        return output

