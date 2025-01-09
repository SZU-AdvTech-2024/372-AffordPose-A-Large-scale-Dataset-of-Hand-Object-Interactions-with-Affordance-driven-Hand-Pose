# -*- coding: utf-8 -*-
#

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os

import time
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False):

        super().__init__()
        self.ds_name = ds_name

        self.ds_name = "train"
        
        self.only_params = only_params

        self.ds_path = os.path.join(dataset_dir, ds_name)

        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))    
        
        
        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_dir, fname) for fname in frame_names])

        self.frame_objs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.cls_name = np.asarray([name.split('/')[-4] for name in self.frame_names])
        self.afford = np.asarray(['_'.join((name.split('/')[-2]).split('_')[1:]) for name in self.frame_names])

        bps_fname = os.path.join(dataset_dir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True

    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]) for k in data.files}
        return data_torch
    
    def load_disk(self,idx):

        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []
        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]


    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
        return data_out

if __name__=='__main__':

    # data_path = '/ps/scratch/grab/contact_results/omid_46/GrabNet/data'
    data_path = 'grab_unzip_data/data'
    ds = LoadData(data_path, ds_name='train', only_params=False)
    
    dataloader = data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    dl = iter(dataloader)
    for i in range(2):
        a = next(dl)
        print(a)
