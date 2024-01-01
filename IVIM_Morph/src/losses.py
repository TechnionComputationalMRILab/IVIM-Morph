from torch import nn
import torch
from torch.autograd import Variable
import numpy as np

def WMSE(y_true, y_pred, W):
    return torch.mean(W * (y_true - y_pred) ** 2)

class comdined_loss(nn.Module):
    def __init__(self, fit_reg_loss_fn, deform_loss_fn,params_loss_fn,lambda_deform,lambda_params):
        """
        Combination loss function used in MIML
        """
        super().__init__()
        self.fit_reg_loss_fn = fit_reg_loss_fn
        self.deform_loss_fn = deform_loss_fn
        self.params_loss_fn = params_loss_fn
        self.lambda_deform = lambda_deform
        self.lambda_params = lambda_params

    def forward(self,img,recon,wrap,phi_true,phi_pred, D, f, Ds):
        # provide y_pred, y_true only for combined loss
        fit_loss = self.fit_reg_loss_fn(img, recon)
        reg_loss = self.fit_reg_loss_fn(wrap, recon)
        # fit_loss =  self.fit_reg_loss_fn(wrap, img)
        deform_reg = self.deform_loss_fn(phi_true, phi_pred)
        D_reg = self.params_loss_fn(D)
        f_reg = self.params_loss_fn(f)
        Ds_reg = self.params_loss_fn(Ds)

        loss = fit_loss +reg_loss+ self.lambda_deform * deform_reg + self.lambda_params * (D_reg+f_reg+Ds_reg)
        loss_list = [fit_loss.item()+reg_loss.item(), deform_reg.item(), D_reg.item(), f_reg.item(), Ds_reg.item()]
        # print(f'{fit_loss= }, {reg_loss= }, {self.lambda_deform*deform_reg=}, {self.lambda_params*D_reg=}, {self.lambda_params*f_reg=}, {self.lambda_params*Ds_reg=}')
        return loss , loss_list

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        # B, nb, 2, nx , ny
        dy = torch.abs(y_pred[:, :, :, 1:, : ] - y_pred[:, :, :, :-1, :  ])
        dx = torch.abs(y_pred[:, :, :, : , 1:] - y_pred[:, :, :, :  , :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class GradParam:
    """
    based on voxelmorph implementation
    N-D gradient loss.
    """

    def __init__(self, penalty='l1',  loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, param):
        dy = torch.abs(param[:, 1:, :] - param[:,:-1, :])
        dx = torch.abs(param[:, :, 1:] - param[:,:, :-1])
        # dz = torch.abs(param[:, :, :, :, 1:] - param[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            # dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy)# + torch.mean(dz)
        grad = d / 2.0 #3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def r2_score(img, model):
    # [batch, nb, nx, ny]
    model_mean = torch.mean(model, dim=1)
    ss_tot = torch.sum((model - model_mean) ** 2, dim=1)
    ss_res = torch.sum((model - img) ** 2, dim=1)
    r2 = 1 - ss_res / ss_tot

    # if ss_tot = 0 -> return NaN and ignore when average/median
    r2 = torch.where(ss_tot <= 1e-4 , float('nan'), r2)
    # mse = (model - img).square().mean(dim=1)
    # r2 = mse
    return r2


class SER_score():
    def __init__(self,n,p,norm,device):
        self.n = n
        self.p = p
        b_vector = torch.tensor([0, 50, 100, 200, 400, 600]).to(device)
        self.w = torch.log(b_vector+1)+1
        self.norm = norm

    def loss(self, img, model):
        '''
        [batch, nb, nx * ny]
        calc SER for each pixel per batch
        '''
        # mse = (model - img).square().sum()
        # SER = ((model - img).square().sum() / (n-p-1)).sqrt() #stansart error of regression
        # mse =  torch.sum(w*((img - model)** 2)/model, dim=1)
        if self.norm == 'l2':
            wmse = (((img - model) ** 2) * self.w.unsqueeze(1)).sum(dim=1) / self.w.sum()
            # SER = torch.sqrt(SER +1e-8)
            mse_ad = wmse/(self.n-self.p-1)
            SER = torch.sqrt(mse_ad+1e-10)
        elif self.norm == 'l1':
            wl1 = (torch.abs(img - model) * self.w.unsqueeze(1)).sum(dim=1) / self.w.sum()
            l1_ad = wl1/(self.n-self.p-1)
            SER = torch.sqrt(l1_ad+1e-10)
        return SER

class SER_score_np():
    def __init__(self,n,p):
        self.n = n
        self.p = p
        b_vector = np.array([0, 50, 100, 200, 400, 600])
        self.w = np.log(b_vector+1)+1

    def loss(self, img, model):
        '''
        [batch, nb, nx * ny]
        calc SER for each pixel per batch
        '''

        wmse =(((img - model) ** 2) * np.expand_dims(self.w,1)).sum(axis=1) / self.w.sum()
        mse_ad = wmse/(self.n-self.p-1)
        SER = np.sqrt(mse_ad+1e-10)
        return SER

def dice_score_np(y_true, y_pred):
    ndims = len(list(y_pred.shape)) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(axis=tuple(vol_axes))
    bottom = np.clip((y_true + y_pred).sum(axis=tuple(vol_axes)), a_min=1e-5, a_max= None)
    dice = np.mean(top / bottom)
    return dice


class SER_loss():
    def __init__(self,n,p):
        self.n = n
        self.p = p
    def loss(self, img, model):
        '''
        [batch, nb, nx * ny]
        calc SER for each pixel per batch
        '''
        # mse = (model - img).square().sum()
        SER = ((model - img).square().sum(dim=1) / (self.n-self.p-1)) #stansart error of regression
        SER = torch.sqrt(SER +1e-10)
        return SER

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/5ebef416a7bfefda80b4bd8f58e6850eccebca2e/ViT-V-Net/losses.py#L288
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        # print(sigma)

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean() #average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)