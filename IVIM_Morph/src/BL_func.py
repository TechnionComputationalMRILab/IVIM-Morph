import numpy as np
import sys
sys.path.append('../')
from  ClassicIVIMfit import *
import time
import ants
import losses
import scipy 
""" Base line methods implementations
    """
def SLS_TRF(FetalData, bounds, bvalues):
    start = time.time()
    n_b, sx , sy = FetalData.shape
    datatot = np.reshape(FetalData, (n_b, sx * sy)).T
    bounds[0][3] = datatot[:, 0].min().item() * 0.9
    bounds[1][3] = datatot[:, 0].max().item() * 1.1
    fit = IVIM_LS(bvalues, bounds)
    D, f, DStar, s0 = fit.IVIM_fit_sls_trf(datatot)
    Recon = s0 * (f * np.exp(-bvalues[:, None] * (D + DStar)) + (1 - f) * np.exp(-bvalues[:, None] * D))
    model_recon = Recon.reshape(6, sx, sy)
    D = D.reshape(sx,sy)
    f = f.reshape(sx,sy)
    DStar = DStar.reshape(sx,sy)
    s0 = s0.reshape(sx,sy)
    elapsed = (time.time() - start)

    return D, f, DStar, s0, model_recon, elapsed

def SyN_TRF(FetalData , FetalSeg,bounds, bvalues, reg_to = 'b0'):
    #reg:
    start = time.time()

    new_data = []
    new_seg = []
    fi = ants.from_numpy(FetalData[0])
    new_data.append(np.expand_dims(fi.numpy(), 0))
    new_seg.append(np.expand_dims(FetalSeg[0], 0))
    for j in range(FetalData.shape[0] - 1):
        mi = ants.from_numpy(FetalData[j + 1])
        reg = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
        new_data.append(np.expand_dims(reg['warpedmovout'].numpy(), 0))
        new_seg.append(np.expand_dims(ants.apply_transforms(fixed=fi, moving=ants.from_numpy(FetalSeg[j + 1]),
                                              transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor' ).numpy(),0))
        if reg_to == 'next_b':
            fi = reg['warpedmovout']

    warpdata = np.vstack(new_data)
    warpseg = np.vstack(new_seg)
    D, f, DStar, s0, model_recon, _ = SLS_TRF(warpdata, bounds, bvalues)
    elapsed = (time.time() - start)
    return D, f, DStar, s0, model_recon, warpdata, warpseg, elapsed

def Affine_TRF(FetalData , FetalSeg,bounds, bvalues, reg_to = 'b0', return_affine_mat = False):
    #reg:
    start = time.time()

    new_data = []
    new_seg = []
    affine_mats = []
    fi = ants.from_numpy(FetalData[0])
    new_data.append(np.expand_dims(fi.numpy(), 0))
    new_seg.append(np.expand_dims(FetalSeg[0], 0))
    for j in range(FetalData.shape[0] - 1):
        mi = ants.from_numpy(FetalData[j + 1])
        reg = ants.registration(fixed=fi, moving=mi, type_of_transform='AffineFast')
        new_data.append(np.expand_dims(reg['warpedmovout'].numpy(), 0))
        new_seg.append(np.expand_dims(ants.apply_transforms(fixed=fi, moving=ants.from_numpy(FetalSeg[j + 1]),
                                              transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor' ).numpy(),0))
        if return_affine_mat:
            affine_mats.append(scipy.io.loadmat(reg['fwdtransforms'][0])['AffineTransform_float_2_2'].T.tolist())
        if reg_to == 'next_b':
            fi = reg['warpedmovout']

    warpdata = np.vstack(new_data)
    warpseg = np.vstack(new_seg)
    D, f, DStar, s0, model_recon, _ = SLS_TRF(warpdata, bounds, bvalues)
    elapsed = (time.time() - start)
    if return_affine_mat:
        return D, f, DStar, s0, model_recon,warpdata, warpseg, elapsed, affine_mats
    return D, f, DStar, s0, model_recon,warpdata, warpseg, elapsed

def Iter_SyN_TRF(FetalData , FetalSeg, bounds, bvalues, num_iter = 1000):
    start = time.time()
    SER_max = 1e6
    OriginalData = FetalData
    for i in range(num_iter):
        D, f, DStar, s0, model_recon, elapsed = SLS_TRF(FetalData, bounds, bvalues)

        new_data = []
        new_seg = []
        for j in range(FetalData.shape[0]):
            fi = ants.from_numpy(model_recon[j])
            mi = ants.from_numpy(OriginalData[j])
            reg = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
            new_data.append(np.expand_dims(reg['warpedmovout'].numpy(), 0))
            new_seg.append(np.expand_dims(ants.apply_transforms(fixed=fi, moving=ants.from_numpy(FetalSeg[j]),
                                              transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor' ).numpy(),0))
        FetalData = np.vstack(new_data)
        warpseg = np.vstack(new_seg)

        SER = losses.SER_score_np(6, 4).loss(np.expand_dims(FetalData, 0).reshape(1, 6, -1),
                                             np.expand_dims(model_recon, 0).reshape(1, 6, -1))
        if SER.mean() >= SER_max or i == (num_iter-1):
            elapsed = (time.time() - start)
            return D_final, f_final, DStar_final, s0_final, model_recon_final, warpdata_final, warpseg_final, elapsed
        else:
            SER_max = SER.mean()
            D_final =D
            f_final = f
            DStar_final = DStar
            s0_final = s0
            model_recon_final = model_recon
            warpdata_final = FetalData
            warpseg_final = warpseg

