# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
 
default_collate_func = dataloader.default_collate
 
 
def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)
 
setattr(dataloader, 'default_collate', default_collate_override)
 
for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]
 
####以下是train的原始代码
import sys
sys.path.append('.')
sys.path.append('..')
import os
import argparse
from grabnet.tools.cfg_parser import Config
from grabnet.train.trainer import Trainer


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    parser = argparse.ArgumentParser(description='AffordPoseNet-Training')

    parser.add_argument('--work-dir', required=True, type=str,
                        help='The path to the downloaded grab data')
    # parser.add_argument('--work-dir', required=False, type=str,
    #                     help='The path to the downloaded grab data')

    parser.add_argument('--data-path', required=True, type=str,
                        help='The path to the folder that contains GrabNet data')

    parser.add_argument('--rhm-path', required=True, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--expr-ID', default='V01', type=str,
                        help='Training ID')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='Training batch size')

    parser.add_argument('--n-workers', default=2, type=int,
                        help='Number of PyTorch dataloader workers')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Training learning rate')

    parser.add_argument('--kl-coef', default=5e-3, type=float,
                        help='KL divergence coefficent for Coarsenet training')

    parser.add_argument('--use-multigpu', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='If to use multiple GPUs for training')

    parser.add_argument('--load-on-ram', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='This will load all the data on the RAM memory for faster training.'
                             'If your RAM capacity is more than 40 Gb, consider using this.')


    args = parser.parse_args()

    work_dir = args.work_dir
    data_path = args.data_path
    rhm_path = args.rhm_path
    expr_ID = args.expr_ID
    batch_size = args.batch_size
    base_lr = args.lr
    n_workers = args.n_workers
    multi_gpu = args.use_multigpu
    kl_coef = args.kl_coef
    load_on_ram = args.load_on_ram


    cwd = os.getcwd()
    default_cfg_path = '/root/GRAB/affordance_Grabnet/grabnet/configs/grabnet_cfg.yaml'
    vpe_path = '/root/GRAB/affordance_Grabnet/grabnet/configs/verts_per_edge.npy'
    c_weights_path = '/root/GRAB/affordance_Grabnet/grabnet/configs/rhand_weight.npy'
    faces_path = '/root/GRAB/affordance_Grabnet/grabnet/configs/faces.npy'


    cfg = {
        'batch_size': batch_size,
        'n_workers': n_workers,

        'use_multigpu':multi_gpu,

        'kl_coef': kl_coef,

        'dataset_dir': data_path,
        'rhm_path': rhm_path,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,
        'faces_path': faces_path,

        'expr_ID': expr_ID,
        'work_dir': work_dir,

        'base_lr': base_lr,

        # 'best_cnet': 'grabnet/configs/snapshots/TR00_E000_cnet.pt',
        # 'best_rnet': 'grabnet/configs/snapshots/TR00_E006_rnet.pt',
        'best_cnet': None,
        'best_rnet': None,
        'load_on_ram': load_on_ram
        
    }

    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    
    torch.set_default_tensor_type(torch.FloatTensor)
    grabnet_trainer = Trainer(cfg=cfg)
    
    grabnet_trainer.fit()

    cfg = grabnet_trainer.cfg
    cfg.write_cfg(os.path.join(work_dir, 'TR%02d_%s' % (cfg.try_num, os.path.basename(default_cfg_path))))
