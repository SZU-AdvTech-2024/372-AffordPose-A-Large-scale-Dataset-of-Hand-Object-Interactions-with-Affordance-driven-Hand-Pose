
# -*- coding: utf-8 -*-


import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from grabnet.tools.utils import rotmat2aa
from grabnet.tools.utils import CRot2rotmat
from grabnet.tools.train_tools import point2point_signed
from Hand_mano_graspit import getHandMesh
import transforms3d


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class CoarseNet(nn.Module):
    def __init__(self,
                 n_neurons = 512,
                 latentD = 10,
                 in_bps = 4096,
                 in_pose = 12,
                 in_label = 10,
                 **kwargs):

        super(CoarseNet, self).__init__()

        self.latentD = latentD

        self.enc_bn0 = nn.BatchNorm1d(in_bps + in_label)
        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_label + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_label + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps+ in_label + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps + in_label)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(latentD + in_bps + in_label, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + in_bps + in_label, n_neurons)

        self.dec_rot = nn.Linear(n_neurons, 6)
        self.dec_dof = nn.Linear(n_neurons, 16)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def encode(self, bps_object, afford_label , trans_rhand, global_orient_rhand_rotmat):
        
        bs = bps_object.shape[0]

        X = torch.cat([bps_object, afford_label, global_orient_rhand_rotmat.view(bs, -1), trans_rhand], dim=1)
        X0 = self.enc_bn1(X)
        X  = self.enc_rb1(X0, True)
        X  = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, bps_object, afford_label):

        bs = Zin.shape[0]
        afford_obj = torch.cat([bps_object, afford_label], dim=1)
        o_bps = self.dec_bn1(afford_obj)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        global_pose = self.dec_rot(X)
        pose = self.dec_dof(X)
        trans = self.dec_trans(X)
        results = parms_decode(global_pose, pose, trans)
        results['z'] = Zin

        return results

    def forward(self, bps_object, afford_label, trans_rhand, global_orient_rhand_rotmat, **kwargs):

        z = self.encode(bps_object, afford_label, trans_rhand, global_orient_rhand_rotmat)
        z_s = z.rsample()
        
        hand_parms = self.decode(z_s, bps_object, afford_label)
        results = {'mean': z.mean, 'std': z.scale}
        results.update(hand_parms)

        return results

    def sample_poses(self, bps_object, afford_label ,seed=None):
        bs = bps_object.shape[0]
        np.random.seed(seed)
        dtype = bps_object.dtype
        device = bps_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen,dtype=dtype).to(device)

        return self.decode(Zgen, bps_object, afford_label)


class RefineNet(nn.Module):
    def __init__(self,
                 in_size=946 + 16 + 6 + 3,
                 h_size=512,
                 n_iters=3):

        super(RefineNet, self).__init__()

        self.n_iters = n_iters
        self.bn1 = nn.BatchNorm1d(946)
        self.rb1 = ResBlock(in_size,  h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)
        # self.out_p = nn.Linear(h_size, 16 * 6)
        self.out_p = nn.Linear(h_size, 16)
        self.out_g = nn.Linear(h_size, 6)
        self.out_t = nn.Linear(h_size, 3)
        self.dout = nn.Dropout(0.3)
        self.actvf = nn.LeakyReLU(.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, h2o_dist, fpose_rhand_dof_f, trans_rhand_f, global_orient_rhand_rotmat_f, verts_object, **kwargs):

        bs = h2o_dist.shape[0]
        init_rpose = global_orient_rhand_rotmat_f[..., :2].reshape(bs, -1)
        init_pose = fpose_rhand_dof_f
        init_trans = trans_rhand_f

        for i in range(self.n_iters):

            if i != 0:
                hand_parms = parms_decode(init_rpose, init_pose, init_trans)
                verts_rhand = getHandMesh(self.cfg.rhm_path, self.rhm_train.palm.forward(**hand_parms)).to(self.device)
                
                _, h2o_dist, _ = point2point_signed(verts_rhand, verts_object)

            h2o_dist = self.bn1(h2o_dist)
            X0 = torch.cat([h2o_dist, init_rpose, init_pose, init_trans], dim=1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            dof = self.out_p(X)
            global_pose = self.out_g(X)
            trans = self.out_t(X)

            init_trans = init_trans + trans
            init_pose = init_pose + dof
            init_rpose = init_rpose + global_pose

        hand_parms = parms_decode(init_rpose, init_pose ,init_trans)
        return hand_parms

def parms_decode(global_pose,dof,trans):

    bs = trans.shape[0]

    pose_full = CRot2rotmat(global_pose)
    global_pose = pose_full.view([bs, 1, -1, 9])
    global_pose = rotmat2aa(global_pose).view(bs, -1)

    global_orient = pose_full
    hand_dof = dof
    pose_full = pose_full.view([bs, -1, 3, 3])
    


    root_transform = torch.cat((global_orient, trans.view([-1, 3, 1])), dim=2)

    hand_parms = {'root_trans': root_transform, 'dofs': hand_dof, "trans": trans, "global_orient": global_orient}

    return hand_parms