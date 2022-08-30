import os
import pandas as pd

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import torch.nn as nn
import torchio as tio
import json
from config import config
from ExpFitGA_ADC import ADC_GA_corr
from plots import *
from torch.utils.tensorboard import SummaryWriter

class Registration_SubNet(nn.Module):
    def __init__(self, config, inshape):
        super().__init__()
        self.inshape = inshape
        self.nb = config['b_vector'].size(0)
        self.model_vxm = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[config['enc_nf'], config['dec_nf']],
            int_steps=config['int_steps'], int_downsize=config['int_downsize']
        )
        self.zero_phi = torch.zeros([self.nb, *self.inshape, len(self.inshape)])
        self.device = config['device']

    def forward(self, fixed, moving, seg):
        inputs = [moving, fixed, seg]
        inputs = [d.to(self.device).float().permute(0, 4, 1, 2, 3) for d in inputs]

        pred = self.model_vxm(*inputs, registration=True)

        return pred


class qDWI_SubNet(nn.Module):
    def __init__(self, config, inshape, eps=1e-4):
        super().__init__()
        self.b_vector = config['b_vector']
        self.eps = eps
        self.nb = config['b_vector'].size(0)
        self.nx, self.ny, self.nz = inshape

    def LS_Layer(self, s):
        '''

        :param s: (n_b_values , n_pixels)
        :param b_vector:
        :param eps:
        :return:
        '''
        A = torch.stack((torch.ones_like(self.b_vector), -self.b_vector), 1)
        s[s <= 0] = self.eps
        b = torch.log(s)
        x = torch.pinverse(A) @ b
        s0 = torch.exp(x[0])
        ADC = x[1]
        Ax = torch.matmul(A, x)
        return s0, ADC, Ax, b

    def Recon_Layer(self, s0, ADC):
        model_recon = s0 * torch.exp(-self.b_vector[:, None] * ADC)
        return model_recon

    def forward(self, img):
        S = torch.reshape(img, (self.nb, self.nx * self.ny * self.nz))
        s0, ADC, Ax, b = self.LS_Layer(S)
        model_recon = self.Recon_Layer(s0, ADC)
        model_recon = torch.reshape(model_recon, (self.nb, self.nx, self.ny, self.nz))
        ADC = torch.reshape(ADC, (self.nx, self.ny, self.nz))
        Ax = torch.reshape(Ax, (self.nb, self.nx, self.ny, self.nz))
        b = torch.reshape(b, (self.nb, self.nx, self.ny, self.nz))

        return model_recon, ADC, Ax, b


def LoadCase(DF, case, datebase_path):
    GA = DF[DF['Case number'] == case]['GA'].values[0]
    filename = case.lower() + '.nii'
    path = os.path.join(datebase_path, filename)
    img = tio.ScalarImage(path).data
    seg = tio.ScalarImage(path.replace("case", "seg")).data
    return img, seg, GA


def PreProcessCase(img, seg, config):
    nb, nx_original, ny_original, nz_original = img.shape

    nx = config['crop_x']
    ny = config['crop_y']
    nz = config['crop_z']
    inshape = (nx, ny, nz)

    seg_roi_slice = torch.unique(torch.argwhere(seg)[:, 3]).item()
    if seg_roi_slice > (nz_original - nz) // 2 and seg_roi_slice < (nz_original - (nz_original - nz) // 2):
        transform = tio.Crop(
            ((nx_original - nx) // 2, (nx_original - nx) // 2, (ny_original - ny) // 2, (ny_original - ny) // 2,
             (nz_original - nz) // 2, (nz_original - nz) // 2 + (nz_original - nz) % 2))
    elif seg_roi_slice <= (nz_original - nz) // 2:
        transform = tio.Crop(
            ((nx_original - nx) // 2, (nx_original - nx) // 2, (ny_original - ny) // 2, (ny_original - ny) // 2,
             seg_roi_slice, nz_original - nz - seg_roi_slice))
    else:
        transform = tio.Crop(
            ((nx_original - nx) // 2, (nx_original - nx) // 2, (ny_original - ny) // 2, (ny_original - ny) // 2,
             -nz + seg_roi_slice + 1, nz_original - seg_roi_slice - 1))

    img = transform(img).to(config['device'])
    seg = transform(seg).squeeze().to(config['device']).type(torch.float32)

    norm_factor = torch.max(img[0, :, :, :])
    return img / norm_factor, seg, inshape, norm_factor


def Loss(outputs, pred, config, fit_mask=None, AX=None, b=None):
    loss = 0
    loss_list = []

    # sim_loss:
    curr_loss = config['losses'][0](outputs[0], pred[0]) * config['weights'][0]
    loss_list.append(curr_loss.item())
    loss += curr_loss

    # Deformation Regularization:
    curr_loss = config['losses'][1](outputs[1], pred[1]) * config['weights'][1]
    loss_list.append(curr_loss.item())
    loss += curr_loss

    if AX is not None:
        # curr_loss = config['losses'][2](AX * fit_mask, b * fit_mask) * config['weights'][2]
        curr_loss = config['losses'][2](b, AX, fit_mask) * config['weights'][2]
        loss_list.append(curr_loss.item())
        loss += curr_loss

    return loss, loss_list


def fitMonoExpModelIRLS(s, b_vector, maxiter=10, tolarance=0.001, delta=0.0001, eps=1e-4):
    # pixel wise (for only one pixel)
    A = torch.stack((torch.ones_like(b_vector), -b_vector), 1)
    s[s <= 0] = eps
    b = torch.log(s)
    W = torch.diag(torch.ones_like(b))
    x = (torch.linalg.inv(A.T @ W @ A)) @ (A.T @ W @ b)
    E = []
    for _ in range(maxiter):
        e = A @ x - b
        w = 1.0 / torch.maximum(torch.abs(e), delta * torch.ones_like(e))
        W = torch.diag(w)
        x = (torch.linalg.inv(A.T @ W @ A)) @ (A.T @ W @ b)

        ee = torch.linalg.norm(A @ x - b, 1)
        E.append(ee)
        if ee < tolarance:
            break

    s0 = torch.exp(x[0])
    ADC = x[1]
    resid = torch.norm(torch.matmul(A, x) - b) ** 2
    R_2 = 1 - resid / (b.shape[0] * b.var())

    return s0, ADC, R_2, E


def TrainCase(DF, case, datebase_path, config, plots_path, Reg_model_weights=None):
    moving_img_history = []
    ROI_history = []
    s0_IRLS_history = []
    ADC_IRLS_history = []
    ADC_history = []
    model_recon_history = []

    # Load and crop&pad data:
    img, seg, GA = LoadCase(DF, case, datebase_path)
    img, seg, inshape, norm_factor = PreProcessCase(img, seg, config)
    seg = seg.unsqueeze(0).repeat(6, 1, 1, 1)
    seg_slice = torch.unique(torch.argwhere(seg)[:, 3])

    # plt.figure()
    # plt.imshow(seg[:, :, seg_slice[0]].detach().cpu())
    # plt.show()

    if config['plot']:
        plotSlice(img.detach().cpu(), seg_slice[0], case, config, seg_vol=seg.detach().cpu(),
                  name=f'Original img', path=plots_path)

    # define registration sub-network
    reg_model = Registration_SubNet(config, inshape)
    if Reg_model_weights is not None:
        reg_model.load_state_dict(Reg_model_weights)
        print('load previous weights')

    reg_model.to(config['device'])
    reg_model.train()

    # define quantitative DWI sub-network
    model_fit = qDWI_SubNet(config, inshape)

    # define optimizer
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=config['lr'])

    epoch_loss = []
    epoch_total_loss = []
    fit_r2 = []
    best_r_2 = 0

    mask_ROI = seg > 0.3
    model_recon, ADC, Ax, b = model_fit(img)

    if config['plot']:
        plotParamMap(ADC.detach().cpu(), seg_slice[0], case, seg_vol=seg.detach().cpu(), path=plots_path)
    ROI = torch.masked_select(img, mask_ROI.to(config['device']))
    ROI = torch.reshape(ROI, (config['b_vector'].size(0), -1))
    s0_IRLS, ADC_IRLS, R_2_IRLS, ROI_error = fitMonoExpModelIRLS(torch.mean(ROI, 1), config['b_vector'], maxiter=50)

    ADC_count = 1
    ROI_ADC = ADC_IRLS
    moving_img_history.append(img.detach().cpu())
    ROI_history.append(ROI.detach().cpu())
    s0_IRLS_history.append(s0_IRLS.detach().cpu())
    ADC_IRLS_history.append(ADC_IRLS.detach().cpu())
    ADC_history.append(ADC.detach().cpu())
    model_recon_history.append(model_recon.detach().cpu())

    for j in range(1, config['max_iter']):
        # Normalization:

        norm_factor = torch.max(img[0, :, :, :])
        img /= norm_factor

        # Registration sub network:
        moving = img[..., np.newaxis]
        fixed = model_recon[..., np.newaxis]
        seg = seg[..., np.newaxis]
        pred = reg_model(fixed, moving, seg)
        img = pred[0].squeeze()
        seg = pred[2].squeeze()

        # fig = plt.figure()
        # plt.imshow(seg[:, :, seg_slice[0]].detach().cpu())
        # plt.show()

        # fit sub network:
        model_recon, ADC, Ax, b = model_fit(img)

        # Loss:
        outputs = [fixed, reg_model.zero_phi]
        outputs = [d.to(config['device']).float().permute(0, 4, 1, 2, 3) for d in outputs]

        fit_mask = seg > 0.3

        loss, loss_list = Loss(outputs, pred, config, fit_mask, Ax, b)
        print(
            f'loss: {loss.item()} --- sim_loss: {loss_list[0]} --- smooth_loss: {loss_list[1]} --- fit_regularize: '
            f'{loss_list[2]}')

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask_ROI = seg > 0.3

        ROI = torch.masked_select(pred[0], mask_ROI.to(config['device']))
        ROI = torch.reshape(ROI, (config['b_vector'].size(0), -1))
        s0_IRLS, ADC_IRLS, R_2_IRLS, ROI_error = fitMonoExpModelIRLS(torch.mean(ROI, 1), config['b_vector'], maxiter=50)

        fit_r2.append(R_2_IRLS.item())
        img = img.detach()
        model_recon = model_recon.detach()
        seg = seg.detach()

        moving_img_history.append(img.detach().cpu())
        ROI_history.append(ROI.detach().cpu())
        s0_IRLS_history.append(s0_IRLS.detach().cpu())
        ADC_IRLS_history.append(ADC_IRLS.detach().cpu())
        ADC_history.append(ADC.detach().cpu())
        model_recon_history.append(model_recon.detach().cpu())

        if R_2_IRLS > best_r_2:
            best_r_2 = R_2_IRLS
            best_ADC = ADC_IRLS
            RegImg = img
            ConvIter = j

        # registration stop condition
        if j == 1 or epoch_total_loss[-1] < epoch_total_loss[-2]:
            continue
        else:
            config['lr'] /= 10
            optimizer = torch.optim.Adam(reg_model.parameters(), lr=config['lr'])


        # stop condition
        if torch.round(ADC_IRLS, decimals=4) != torch.round(ROI_ADC, decimals=4):
            ROI_ADC = ADC_IRLS
            ADC_count = 1
        else:
            ADC_count += 1
            if ADC_count == config['ADC_Convergence']:
                break

    if config['plot']:
        if ConvIter >= 4:
            PlotMovingFixedADC(moving_img_history, model_recon_history, ADC_history, ConvIter, seg_slice[0], case,
                               path=plots_path)
            PlotRegIter(moving_img_history, ConvIter, seg_slice[0], case, config, name='Reg', path=plots_path)
            plotSlice(RegImg.detach().cpu(), seg_slice[0], case, config, seg_vol=seg.detach().cpu(),
                      name=f'wrapped img epoch = {ConvIter}', path=plots_path)
            PlotROI(s0_IRLS_history, ADC_IRLS_history, ROI_history, ConvIter, config, case, path=plots_path)
            PlotLossR_2(case, epoch_total_loss, fit_r2, plots_path)

    Reg_model_weights = reg_model.state_dict()
    return GA, best_ADC, best_r_2, RegImg, Reg_model_weights, ConvIter


def TrainDataSet(config):
    table_path = config['table_path']
    datebase_path = config['datebase_path']
    DF = pd.read_excel(table_path, index_col=None, header=0)
    DATA_ID = DF['Case number'].values

    path = os.path.join(init['plots_path'], f'seed_{init["seed"]}')
    if not os.path.exists(path):
        os.makedirs(path)

    Reg_model_weights = None
    R_2_list = []
    for _ in range(2):
        R_2_IRLS_list = []
        fit = dict()
        for case in DATA_ID:
            print(f'case: {case}')
            if case == 'Case749':
                continue

            path_case = os.path.join(path, case)
            if not os.path.exists(path_case):
                os.makedirs(path_case)
            GA, ADC_IRLS, R_2_IRLS, RegImg, Reg_model_weights, ConvIter = TrainCase(DF, case, datebase_path, config,
                                                                                    plots_path=path_case,
                                                                                    Reg_model_weights=Reg_model_weights)

            fit[case] = {'GA': GA, 'ADC_IRLS_AVG': ADC_IRLS.item(), 'R2_AVG': R_2_IRLS.item(), 'RegImg': RegImg,
                         'ConvIter': ConvIter}

            R_2_IRLS_list.append(R_2_IRLS.item())
        with open(init['Save_dict_path'], 'w') as fp:
            json.dump(fit, fp, default=str, indent=4, sort_keys=True)

        R_2 = ADC_GA_corr(path, fit)
        R_2_list.append(R_2)

        R_2_IRLS_mean = np.mean(R_2_IRLS_list)
        R_2_IRLS_var = np.var(R_2_IRLS_list)
        print(f'{R_2_IRLS_mean = :.3f}')
        print(f'{R_2_IRLS_var = :.3f}')

    plt.figure()
    plt.plot(R_2_list)
    plt.xlabel('epoch')
    plt.ylabel('R_2')
    plt.title(f'LS R^2 as function of epoch - shared weights - for each case calculate model once per epoch')
    plt.savefig(f'R_2_vs_epoch')
    return fit, R_2


if __name__ == '__main__':
    # config = config()
    # random_seed = torch.randint(100, (1,)).item()

    # seed = torch.seed()
    init = config()
    torch.manual_seed(init['seed'])
    fit, R_2 = TrainDataSet(init)
