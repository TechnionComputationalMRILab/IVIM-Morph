import torch
import numpy
import matplotlib.pyplot as plt
# import pynd
import os
import sys
sys.path.append('./src')
import argparse
import numpy as np
import Dataset
import torch.utils.data as utils
from losses import SER_score
from Net import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import torch
from torch.utils.tensorboard import SummaryWriter
import json
import datetime

class solver_IVIM_Morph:
    """Solver class for simultaneous optimization of IVIM model parameter estimation
       and deformable registration of fetal lung DWI images.
    """
    def __init__(self, args = None, device = 'cpu'):
        assert(args != None), 'Must provide arguments for the solver'
        #create directory for the plots: 
        self.dir_name = f'solver_run_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        self.args = args
        self.device = device
        # configure case dataset and dataloader
        enc_nf = args.enc
        dec_nf = args.dec


        self.nb, self.nx, self.ny = args.dimension 

        #configure model
        b_values = torch.tensor(args.b_values , dtype = torch.float).to(device)
        self.nb = b_values.size()[0]
        inshape = [self.nx, self.ny]
        self.model = qDIVIM(bvalues=b_values, 
                            inshape=inshape, 
                            nb_unet_features=[enc_nf,dec_nf], 
                            dropout=0,
                            nb_unet_features_reg=[enc_nf,dec_nf], 
                            int_steps = args.int_steps, 
                            s0_in_recon=False, 
                            reg_s0=args.reg_s0).to(device)

        self.model.to(device)
        self.model.train()

        # tensorboard:
        if args.tensorboard:
            self.writer = SummaryWriter(comment=f'{args.comment}_{self.dataset[0][2]}', flush_secs=1)

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=False)

        # prepare deformation loss
        self.losses_deform = vxm.losses.Grad('l2').loss
        self.weight_deform = args.weight_grad

        #ncc loss:
        self.loss_ncc = vxm.losses.NCC().loss
        self.weight_ncc = args.weight_ncc

        # prepare ser loss: 
        self.ser_loss = SER_score(6, self.args.n,norm = args.ser_loss_norm, device = device).loss
        self.weight_fit = args.weight_fit

        # transformers for dice loss
        self.seg_transformer = vxm.layers.SpatialTransformer(inshape, 'nearest').to(device)
        self.transformer = vxm.layers.SpatialTransformer(inshape, args.interpol_method).to(device)
        self.plot_transformer = vxm.layers.SpatialTransformer(inshape)

        
    def fit(self, data, return_wraped_seg = False):
        print(f'fitting case with {self.args.iters} iterations...')
        # prepare case:
        
        img = data[0].to(self.device).to(torch.float32).unsqueeze(0)
        if data[1] != None:
            seg = data[1].to(self.device).to(torch.float32).unsqueeze(0)

        s0_batch = img[0,0,...].unsqueeze(0).repeat_interleave(self.nb-1, dim=0).to(self.device)
        best_Loss = 1e6
        iters_without_improvement = 0 

        # training loop
        for iter in tqdm(range(self.args.iters), desc="Training Progress"):
            # run inputs through the model to produce a warped image and flow field
            wrap, phi_pre_integ, model_recon, s0, D, f, Dp, pos_flow = self.model(img)
                
            if self.args.wrap_costume:
                    wrap =  self.transformer(img.permute(1,0,2,3), pos_flow.squeeze(0)).permute(1,0,2,3)

            # deformation field loss
            L_grad = self.losses_deform(None, phi_pre_integ)

            # NCC loss
            L_NCC = self.loss_ncc(s0_batch.unsqueeze(1), wrap[0,1:,...].unsqueeze(1)) + 1

            # fit loss
            if self.args.roi_fit_loss and data[1] != None: # calculate loss fit just in roi
                s0_seg = seg[0,...]
                x_inds, y_inds = torch.where(s0_seg)
                L_ser = self.ser_loss(wrap[...,x_inds, y_inds], (s0*model_recon)[...,x_inds, y_inds])
            else:
                L_ser = self.ser_loss(wrap.view(1, self.nb, -1), (s0*model_recon).view(1, self.nb, -1))
            norm = model_recon.mean()
            L_ser = (L_ser/norm).mean()

            loss =L_grad * self.weight_deform + L_ser * self.weight_fit + L_NCC * self.weight_ncc

            if iter % self.args.print_every == 0:
                print(f'Iter {iter} loss = {"%.4f" % loss}')
            # backpropagate and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.args.lr_schedular:
                self.scheduler.step(loss)

            # look for convergence
            if loss.item() < best_Loss: 
                best_Loss = loss.item()
                iters_without_improvement = 0
            else:
                if iter > self.args.min_iter: # minimum iterations without early stopping
                    iters_without_improvement += 1
                    if iters_without_improvement > self.args.early_stopping:
                        print(f'Stopping training in iter {iter} because there is no improvement in the loss')
                        break

            if self.args.tensorboard:
                self.writer.add_scalar('Loss-grad', L_grad.item(), iter)
                self.writer.add_scalar('Loss-NCC', L_NCC.item(), iter)
                self.writer.add_scalar('Loss-SER', L_ser.item(), iter)
                self.writer.add_scalar('Loss-total', loss.item(), iter)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], iter)


        if 1 > 0: # if trying to optimize case with augmentations
            pos_flow = pos_flow.mean(0).unsqueeze(0)
            wrap = wrap.mean(0).unsqueeze(0)
            D = D.mean(0).unsqueeze(0)
            Dp = Dp.mean(0).unsqueeze(0)
            f = f.mean(0).unsqueeze(0)

        ivim_params = [D.detach(), f.detach(), Dp.detach()]
        print('Finished fitting...')
        if self.args.tensorboard:
            self.writer.close()      
        if return_wraped_seg:
            warped_seg = self.transformer(seg.permute(1,0,2,3), pos_flow.squeeze(0)).permute(1,0,2,3)
            return wrap.view(-1, self.nb, self.nx, self.ny).detach(), warped_seg.detach(), pos_flow.detach(), ivim_params

        return wrap.view(-1, self.nb, self.nx, self.ny).detach(),  pos_flow, ivim_params
