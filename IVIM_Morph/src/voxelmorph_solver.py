import torch
import numpy

import os
import argparse

import Dataset
import torch.utils.data as utils
import sys
sys.path.append('./src')

from Net import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
from losses import MutualInformation
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import datetime
import torchsummary
import matplotlib.pyplot as plt
# self.Writer will output to ./runs/ directory by default

def xavier_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class solver_voxelMorph:
    """Solver class for deformable registration of fetal lung DWI images."""
    def __init__(self, dataset, return_wraped_seg = False, args = None, device = 'cpu'):
        assert(args != None), 'Must provide arguments for the solver'
        #create directory for the plots: 
        self.dir_name = f'solver_run_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

        self.return_wraped_seg = return_wraped_seg
        self.args = args
        self.device = device
        # configure case dataset and dataloader
        self.dataset = dataset
        self.dataloader = utils.DataLoader(self.dataset,
                                           batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       drop_last=False)

        #configure model
        b_values = torch.tensor(args.b_values , dtype = torch.float).to(device)
        self.nb = b_values.size()[0]
        self.inshape = self.dataset[0][0].shape[1:]
        self.model = vxm.networks.VxmDense(
            inshape=self.inshape,
            nb_unet_features=[args.enc, args.dec],
            int_steps=args.int_steps,
            int_downsize = args.int_downsize,
            )  
        self.zero_phi = torch.zeros([self.nb-1, *self.inshape, len(self.inshape)])
        self.model.to(device)

        self.model.train()

        images = self.dataset[0][0].unsqueeze(0)
        self.B, self.nb, self.nx, self.ny = images.shape
        # tensorboard:
        if args.tensorboard:
            self.writer = SummaryWriter(comment=f'{args.comment}_{self.dataset[0][2]}', flush_secs=1)

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=False)

        self.dice_loss = vxm.losses.Dice().loss

        # prepare deformation loss
        self.losses_deform = vxm.losses.Grad('l2').loss
        self.weight_deform = 0.01

        #ncc loss:
        self.loss_ncc = vxm.losses.NCC().loss 
        self.weight_ncc = 1


        # transformers for dice loss
        self.seg_transformer = vxm.layers.SpatialTransformer(self.inshape, 'nearest').to(device)
        self.transformer = vxm.layers.SpatialTransformer(self.inshape, args.interpol_method).to(device)
        self.plot_transformer = vxm.layers.SpatialTransformer(self.inshape)


    def fit_case(self):
        batch = self.dataset[0]
        if self.return_wraped_seg:
            wrap_batch,  wrap_seg_batch = self._fit_one_batch(batch)
        else:
            wrap_batch = self._fit_one_batch(batch)
                
        if self.return_wraped_seg:
            return wrap_batch,  wrap_seg_batch
        
        return wrap_batch

    def _fit_one_batch(self, batch):
        # prepare case:
        FetalData, FetalSeg, FetalGA, DATA_ID = batch
        FetalData = FetalData.unsqueeze(0)
        img = FetalData[0,1:,...].to(self.device).to(torch.float32)
        seg = FetalSeg.to(self.device).to(torch.float32)
        if len(seg.shape) == 3:
            seg = seg.unsqueeze(0)
        seg = seg[:,1:,...]
        initial_seg = seg
        s0_batch = FetalData[0,0,...].unsqueeze(0).repeat_interleave(self.nb-1, dim=0).to(self.device)
        fixed = s0_batch.unsqueeze(1)
        best_Loss = 1e6
        iters_without_improvement = 0 
        for iter in range(self.args.iters):
            # run inputs through the model to produce a warped image and flow field
            moving = img.unsqueeze(1)

            wrap, phi_pre_integ, pos_flow = self.model(moving, fixed)

            if self.return_wraped_seg:
                warped_seg = self.transformer(seg.permute(1,0,2,3), pos_flow.squeeze(0)).permute(1,0,2,3)

            if self.args.wrap_costume:
                wrap =  self.transformer(moving, pos_flow.squeeze(0))

            # deformation field loss
            L_grad = self.losses_deform(self.zero_phi , phi_pre_integ.unsqueeze(0))

            # NCC loss
            L_NCC = self.loss_ncc(fixed, wrap) + 1

            # Total loss
            loss =L_grad * self.weight_deform + L_NCC * self.weight_ncc
            # backpropagate and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.lr_schedular:
                self.scheduler.step(loss)

            if iter > self.args.init_iter: # look for convergence from 'self.args.init_iter' iteration 
                if loss.item() < best_Loss:
                    best_Loss = loss.item()
                    iters_without_improvement = 0
                else:
                    if iter > self.args.min_iter: 
                        iters_without_improvement += 1
                        if iters_without_improvement > self.args.early_stopping:
                            print(f'Stopping training in iter {iter} because there is no improvement in the loss')
                            break
            if self.args.tensorboard:
                self.writer.add_scalar('Loss-reg-grad', L_grad.item(), iter)
                self.writer.add_scalar('Loss-reg-NCC', L_NCC.item(), iter)
                self.writer.add_scalar('Loss-total', loss.item(), iter)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], iter)
                self.writer.add_scalar('mean_value_of_deform_field', phi_pre_integ.mean().item(), iter)

        wrap = torch.hstack([s0_batch[0,...].unsqueeze(0).unsqueeze(0), wrap.view(-1,5,96,96).detach()])
       
        if self.args.tensorboard:
            self.writer.close()
        if self.return_wraped_seg:
            warped_seg  = torch.hstack([initial_seg[:,0,...].unsqueeze(0), warped_seg.detach()])
            return wrap, warped_seg
        
        return wrap
    

    




