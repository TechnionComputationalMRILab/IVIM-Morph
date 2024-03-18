import os
import sys
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import torch.nn as nn
import torch

class IVIM_DWI(nn.Module):
    """
    IVIM parameters prediction
    """

    def __init__(self,
                 inshape,
                 bvalues,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 dropout = 0.1,
                 s0_in_recon = True):
        """
        Based on voxelmorph unet
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
        """
        super().__init__()
        self.bvalues = bvalues
        self.s0_in_recon = s0_in_recon

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = vxm.networks.Unet(
            inshape=inshape,
            infeats=(bvalues.size(0)),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=False,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.D_layer= Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.f_layer= Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.Dp_layer= Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.S0_layer= Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        bounds = torch.tensor([[0, 0.0003, 0.07, 0.006], [1.0, 0.0032, 0.5, 0.15]]) #[[s0_min, D_min, F_min, Ds_min],[s0_max, D_max, F_max, Ds_max]]
        boundsrange = 0.15 * (bounds[1] - bounds[0])  # ensure that we are on the most lineair bit of the sigmoid function
        self.bounds = torch.clamp(torch.stack((bounds[0] - boundsrange, bounds[1] + boundsrange)),0)

    def IVIM_model(self, D, Fp, Dp):

        B, nb, nx, ny = D.shape
        D = D.view(-1)
        Fp = Fp.view(-1)
        Dp = Dp.view(-1)

        recon = Fp*torch.exp(-self.bvalues.unsqueeze(1)*Dp) + (1-Fp)*torch.exp(-self.bvalues.unsqueeze(1)*D)
        recon = recon.view(B, self.bvalues.shape[0], nx, ny)
        return recon

    def forward(self, img):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        x = self.unet_model(img)

        D = self.D_layer(x)
        f = self.f_layer(x)
        Dp = self.Dp_layer(x)


        D = self.bounds[0,1]+ torch.sigmoid(D) *(self.bounds[1,1]- self.bounds[0,1])
        f = self.bounds[0,2]+ torch.sigmoid(f) *(self.bounds[1,2]-self.bounds[0,2])
        Dp = self.bounds[0,3]+ torch.sigmoid(Dp) *(self.bounds[1,3]-self.bounds[0,3])

        if Dp.clone().mean() < D.clone().mean():
            tmp = D
            D = Dp
            Dp = tmp
            f = 1 - f

        s0 = img[:,0,:,:].unsqueeze(0)
        if self.s0_in_recon:
            recon = self.IVIM_model(D, f, Dp)*s0
        else:
            recon = self.IVIM_model(D, f, Dp)
        return recon, s0, D, f, Dp

class Registration_SubNet(nn.Module):
    """registration subnetwork - 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, inshape, bvalues, nb_unet_features = None, int_steps = 0, int_downsize = 1 ):
        super().__init__()
        self.inshape = inshape
        self.nb = bvalues.size(0)
        self.model_vxm = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=nb_unet_features,
            int_steps=int_steps,
            int_downsize = int_downsize,
        )

    def forward(self, fixed, moving):
        moving = moving.float().permute(0,-1,1,2)
        fixed = fixed.to(moving.device).float().permute(0,-1,1,2)
        inputs =[moving, fixed]

        pred = self.model_vxm(*inputs)
        return pred

class qDIVIM(nn.Module):
    """
    IVIM parameters prediction
    """

    def __init__(self,
                 inshape,
                 bvalues,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 dropout = 0.1,
                 nb_unet_features_reg = None,
                 int_steps = 5,
                 int_downsize = 2,
                 s0_in_recon = True,
                 reg_s0 = True
    ):
        """
        """
        super().__init__()
        self.IVIM_net = IVIM_DWI(inshape,
                 bvalues,
                 nb_unet_features,
                 nb_unet_levels,
                 unet_feat_mult,
                 nb_unet_conv_per_level,
                 dropout,
                 s0_in_recon)
        
        self.bvalues = bvalues
        self.reg_s0 = reg_s0
        self.Reg_Net = Registration_SubNet(
            inshape=inshape,
            bvalues = bvalues,
            nb_unet_features=nb_unet_features_reg,
            int_steps = int_steps,
            int_downsize = int_downsize
        )
    def forward(self, img):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        B, nb, nx, ny = img.shape
        recon,s0, D, f, Dp = self.IVIM_net(img)
        s0 = s0.permute(1,0,2,3)
        if not self.reg_s0: # NOT registering the S0 image
            recon_img = (s0*recon[:,1:,...]).view(-1,nx,ny).unsqueeze(-1)
            img = img[:,1:,...].reshape(-1,nx,ny).unsqueeze(-1)

            # Forward pass
            wrap , phi_pre_integrated ,pos_flow = self.Reg_Net(recon_img, img)
            wrap = torch.cat((s0, wrap.view(B, nb-1, nx, nx)), dim=1)

            # arrange the deformation fields before integration layer
            phi_pre_integrated = phi_pre_integrated.view(B, nb-1,2,phi_pre_integrated.shape[-2],phi_pre_integrated.shape[-1])
            phi_pre_integrated = torch.cat((torch.zeros(1,1,2,phi_pre_integrated.shape[-1],phi_pre_integrated.shape[-1]).to(img.device).expand(B, 1, 2, phi_pre_integrated.shape[-1], phi_pre_integrated.shape[-1]), phi_pre_integrated), dim=1)
            
            # arrange the deformation fields after integration layer
            pos_flow =  pos_flow.view(B, nb-1,2,pos_flow.shape[-2],pos_flow.shape[-1])
            pos_flow =  torch.cat((torch.zeros(1,1,2,nx, ny).to(img.device).expand(B, 1, 2, nx, ny), pos_flow), dim=1)
            output =  [wrap,  phi_pre_integrated, recon,  s0,  D, f,  Dp, pos_flow]
        else:
            recon_img = (s0.unsqueeze(1)*recon).view(B*nb,nx,ny).unsqueeze(-1)
            img = img.view(B*nb,nx,ny).unsqueeze(-1)
            wrap , phi_pre_integrated ,pos_flow = self.Reg_Net(recon_img , img)
            output =  [wrap.permute(1,0,2,3),  phi_pre_integrated.unsqueeze(0), recon,  s0,  D, f,  Dp, pos_flow.unsqueeze(0)]

        return output



