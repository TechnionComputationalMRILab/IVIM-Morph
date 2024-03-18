import os
import sys
sys.path.append('./src')
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from BL_func import *
import torch 
import Dataset
import torch.utils.data as utils
from tqdm import tqdm
import os
from Net import *
import pandas as pd
from solver import *
from voxelmorph_solver import *
# from lung_seg_analysis import *
# import random
from config import *

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

def calc_bl_all_dataset(b_values, bounds, FetalData, FetalSeg, DATA_ID, FetalGA):
    """calculate all baseline methods IVIM parameters estimation

    Args:
        b_values (np.array): array of the b-values (include b-value=0)
        bounds (list): list consist of 2 sublists, each represent the lower/upper bound of the 
                       IVIM parameters and S0, for example: [[0.0005, 0.05, 0.01, 1], [0.003, 0.55, 0.1, 1]]
        FetalData (torch.tensor): The diffusion weights images + the base line image, shape = (N, b, nx, ny); N = number of cases;
                                  b = number of b-values; nx & ny = image dimensionsץ 
        FetalSeg (torch.tensor): 2D mask in the fetal lung, shape = (N, nx, ny)
        DATA_ID (list): list of strings with cases names in the length of N
        FetalGA (list): list of floats with GAs.
    """
    assert isinstance(b_values, np.ndarray), 'b-value vector must be of np.ndarray'
    assert isinstance(bounds, list) and len(bounds) == 2, 'bounds must be a list with 2 components'
    assert isinstance(FetalData, torch.Tensor) and len(FetalData.shape) == 4, 'FetalData must be a tensor and have 4 dimensions'
    assert isinstance(FetalSeg, torch.Tensor) and len(FetalSeg.shape) == 3, 'FetalData must be a tensor and have 3 dimensions'
    assert isinstance(DATA_ID, list), 'DATA_ID must be a list'
    assert isinstance(FetalGA, list), 'FetalGA must be a list'


    loss_dice = vxm.losses.Dice().loss
    bl_results = {'SLS_TRF':{'D':[],'D*':[],'f':[], 'dice':[]},
                  'Affine_TRF':{'D':[],'D*':[],'f':[], 'dice':[]},
                  'SyN_TRF_b0':{'D':[],'D*':[],'f':[], 'dice':[]},
                  'SyN_TRF_next_b':{'D':[],'D*':[],'f':[], 'dice':[]},
                  'Iter_SyN_TRF':{'D':[],'D*':[],'f':[], 'dice':[]}}
    bl_results['ga'] = FetalGA
    bl_results['cases'] = DATA_ID
    affine_trans = {}
    for i in range(FetalData.shape[0]):
        ROI_mask: object = FetalSeg[i,...].unsqueeze(0).repeat_interleave(6, dim=0).cpu().numpy()
        fetalData = FetalData[i,...].squeeze(0).cpu().numpy()

        # SLS_TRF without registration:
        D, f, DStar, s0, model_recon, _ = SLS_TRF(fetalData, bounds, b_values)
        bl_results['SLS_TRF']['D'].append(D[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SLS_TRF']['D*'].append(DStar[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SLS_TRF']['f'].append(f[ROI_mask[0,...].astype(bool)].mean())

        # Affine_TRF register to b0:
        # * save all affine matrix 
        D, f, DStar, s0, model_recon, warpdata, warpseg, elapsed, affine_mats = Affine_TRF(fetalData , ROI_mask, bounds, b_values, reg_to = 'b0', return_affine_mat=True)
        bl_results['Affine_TRF']['D'].append(D[ROI_mask[0,...].astype(bool)].mean())
        bl_results['Affine_TRF']['D*'].append(DStar[ROI_mask[0,...].astype(bool)].mean())
        bl_results['Affine_TRF']['f'].append(f[ROI_mask[0,...].astype(bool)].mean())
        bl_results['Affine_TRF']['dice'].append(-loss_dice(torch.tensor(ROI_mask),  torch.tensor(warpseg).unsqueeze(1)).item())
        affine_trans[DATA_ID[i]] = affine_mats

        # SyN_TRF register to b0:
        D, f, DStar, s0, model_recon, warpdata, warpseg, elapsed = SyN_TRF(fetalData , ROI_mask, bounds, b_values, reg_to = 'b0')
        bl_results['SyN_TRF_b0']['D'].append(D[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SyN_TRF_b0']['D*'].append(DStar[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SyN_TRF_b0']['f'].append(f[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SyN_TRF_b0']['dice'].append(-loss_dice(torch.tensor(ROI_mask),  torch.tensor(warpseg).unsqueeze(1)).item())

        # SyN_TRF register to next b value:
        D, f, DStar, s0, model_recon, warpdata, warpseg, elapsed = SyN_TRF(fetalData , ROI_mask, bounds, b_values, reg_to = 'next_b')
        bl_results['SyN_TRF_next_b']['D'].append(D[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SyN_TRF_next_b']['D*'].append(DStar[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SyN_TRF_next_b']['f'].append(f[ROI_mask[0,...].astype(bool)].mean())
        bl_results['SyN_TRF_next_b']['dice'].append(-loss_dice(torch.tensor(ROI_mask),  torch.tensor(warpseg).unsqueeze(1)).item())

        # Iterative SyN-TRF
        D_final, f_final, DStar_final, s0_final, model_recon_final, warpdata_final, warpseg_final, elapsed = Iter_SyN_TRF(fetalData , ROI_mask, bounds, b_values, num_iter = 1000)
        bl_results['Iter_SyN_TRF']['D'].append(D_final[ROI_mask[0,...].astype(bool)].mean())
        bl_results['Iter_SyN_TRF']['D*'].append(DStar_final[ROI_mask[0,...].astype(bool)].mean())
        bl_results['Iter_SyN_TRF']['f'].append(f_final[ROI_mask[0,...].astype(bool)].mean())
        bl_results['Iter_SyN_TRF']['dice'].append(-loss_dice(torch.tensor(ROI_mask),  torch.tensor(warpdata_final).unsqueeze(1)).item())


    with open('bl_methods_gaCorr_results_all_data.json', "w") as json_file:
        json.dump(bl_results, json_file)
    with open('affine_trans_all_data.json', "w") as json_file:
        json.dump(affine_trans, json_file)
        

def calc_solver_results(args, FetalData, FetalSeg, DATA_ID, FetalGA, b_values, bounds, only_reg = False, cases = None, use_sls_trf = False):
    """run solver (IVIM-Morph or VoxelMorph) and calculate the f-GA correlations and save the results in json file.

    Args:
        args (argparser.args): the arguments for the solver
        FetalData (torch.tensor): The diffusion weights images + the base line image, shape = (N, b, nx, ny); N = number of cases;
                                  b = number of b-values; nx & ny = image dimensionsץ 
        FetalSeg (torch.tensor): 2D mask in the fetal lung, shape = (N, nx, ny)
        DATA_ID (list): list of strings with cases names in the length of N
        FetalGA (list): list of floats with GAs._
        only_reg (bool, optional): if True, use VoxelMorph solver. Defaults to False.
        cases (list, optional): list of cases within the dataset to run the analysis for. Defaults to None.
        use_sls_trf (bool, optional): if True, uses SLS-TRF to estimate the IVIM parameters. Defaults to False.

    Returns:
        Analysis: results dictionary
    """
    assert isinstance(b_values, np.ndarray), 'b-value vector must be of np.ndarray'
    assert isinstance(bounds, list) and len(bounds) == 2, 'bounds must be a list with 2 components'
    assert isinstance(FetalData, torch.Tensor) and len(FetalData.shape) == 4, 'FetalData must be a tensor and have 4 dimensions'
    assert isinstance(FetalSeg, torch.Tensor) and len(FetalSeg.shape) == 3, 'FetalData must be a tensor and have 3 dimensions'
    assert isinstance(DATA_ID, list), 'DATA_ID must be a list'
    assert isinstance(FetalGA, list), 'FetalGA must be a list'
    if only_reg: use_sls_trf = True # if using VoxleMorph solver, SLS-TRF must be used.

    if only_reg:
        solver_cls = solver_voxelMorph
    else:
        solver_cls = solver_IVIM_Morph

    # if only subset of the cases are provided to solve
    if isinstance(cases, list):
        cases_inds = [DATA_ID.index(case) for case in cases]
        FetalData = FetalData[cases_inds,...]
        FetalSeg = FetalSeg[cases_inds,...]
        DATA_ID = [DATA_ID[i] for i in cases_inds]
        FetalGA = [FetalGA[i] for i in cases_inds]

    results = {'dice' : [],
                'D' : [],
                'D*': [],
                'f' : [],
                'ga' : [],
                'cases' : []}
    
    # iterate over the dataset and run solver for each case
    for i in range(FetalData.shape[0]):
        ds = Dataset.FetalDataset(FetalData[i,...].unsqueeze(0),
                                  FetalSeg[i,...].unsqueeze(0),
                                  [DATA_ID[i]], 
                                  [FetalGA[i]])
        original_seg = ds[0][1]
        print(f'fitting case : {ds[0][2]}')
        # Initial the solver for each case :
        solver = solver_cls(ds, args = args, device='cuda')
        if only_reg:
            warpdata = solver.fit_case()

        else:
            warpdata, _, ivim_params = solver.fit_dataset()
            D, f, Dp = ivim_params[0]
            D = D.squeeze(0).squeeze(0).cpu().numpy()
            f = f.squeeze(0).squeeze(0).cpu().numpy()
            Dp = Dp.squeeze(0).squeeze(0).cpu().numpy()

        if use_sls_trf:
            D, f, Dp, s0, model_recon, _ = SLS_TRF(warpdata.squeeze(0).cpu().numpy(), bounds, b_values)

        results['D'].append(float(D[original_seg.to(bool).cpu()].mean()))
        results['D*'].append(float(Dp[original_seg.to(bool).cpu()].mean()))
        results['f'].append(float(f[original_seg.to(bool).cpu()].mean()))
        results['ga'].append(float(ds[0][-1]))
        results['cases'].append(DATA_ID[i])

    path = f'{solver_cls.__name__}_ivim_params_{args.comment}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.json'
    print(f'Saving solver results in path {path}')
    with open(path, "w") as json_file:
        json.dump(results, json_file)
    return results

def calc_corr(results, ga = None):
    """calculate the f-GA corelation using the results dict output from the baseline method function/ solver function.

    Args:
        results (dict): results of the baseline methods/voxelmorph/IVIM-Morph
        ga (list, optional): list of the GA if they do not appear in results. Defaults to None.

    Returns:
        _type_: _description_
    """
    if ga == None:
        ga = results['ga']
        
    ind = np.argsort(ga)
    GA_list = np.array([ga[i] for i in ind])
    D_list = np.array([results['D'][i] for i in ind])
    Dp_list = np.array([results['D*'][i] for i in ind])
    F_list = np.array([results['f'][i] for i in ind])
    GA_list_Can = GA_list[GA_list < 26]
    GA_list_Sac = GA_list[GA_list >= 26]
    Analysis = {}
    for param, name, range in zip([D_list, Dp_list, F_list], ['Dt', 'Dp', 'Fp'],
                                    [(0.001, 0.0028), (0.01, 0.105), (0.05, 0.45)]):
        # plt.scatter(GA_list, param)
        param= np.array(param).squeeze()
        parm_corr = np.square(np.corrcoef(GA_list, param))
        param_Can = param[GA_list < 26]
        param_Sac = param[GA_list >= 26]
        Can_fit = stats.linregress(GA_list_Can, param_Can)
        Sac_fit = stats.linregress(GA_list_Sac, param_Sac)

        Analysis[f'{name}_can'] = Can_fit.rvalue ** 2
        Analysis[f'{name}_sac'] = Sac_fit.rvalue ** 2
    return Analysis

def separate_results(results, group1_cases, cases = None, ga = None):
    """split the results dictionary to two groups based on group1 cases names

    Args:
        results (dict): results dictionary
        group1_cases (list): list of cases in one group
        cases (list, optional): cases names of all the cases in results if the cases in results doesn't appear in results. Defaults to None.
        ga (list, optional): list of the GA if they do not appear in results. Defaults to None.

    Returns:
        _type_: _description_
    """
    if cases == None:
        cases = results['cases']
    if ga == None:
        ga = results['ga']
    group1 = {'dice' : [], 'D' : [], 'D*': [], 'f' : [], 'ga' : [],'cases' : []}
    group2 = {'dice' : [], 'D' : [], 'D*': [], 'f' : [], 'ga' : [],'cases' : []}
    for i, case in enumerate(cases):
        if case in group1_cases:
            group1['D'].append(results['D'][i])
            group1['D*'].append(results['D*'][i])
            group1['f'].append(results['f'][i])
            group1['ga'].append(ga[i])
            group1['cases'].append(cases[i])
        else:
            group2['D'].append(results['D'][i])
            group2['D*'].append(results['D*'][i])
            group2['f'].append(results['f'][i])
            group2['ga'].append(ga[i])
            group2['cases'].append(cases[i])     

    return group2, group1       

def plot_fit(GA_list, param, axs, title):
    font_properties = {
        'fontsize': 18,
        'fontweight': 'bold',
        'fontfamily': 'serif'
         }
    param_Can = param[GA_list < 26]
    param_Sac = param[GA_list >= 26]
    GA_list_Can = GA_list[GA_list < 26]
    GA_list_Sac = GA_list[GA_list >= 26]
    # CAN 
    slope_can, intercept_can, r_value_can, p_value_can, std_err_can = stats.linregress(GA_list_Can, param_Can)
    predicted_y = slope_can * GA_list_Can + intercept_can
    r_squared_can = r_value_can ** 2
    axs.scatter(GA_list_Can, param_Can,color='blue')
    axs.plot(GA_list_Can, predicted_y, label=f'Canalcular Stage - R²(Can) = {r_squared_can:.4f}', color='green')

    # SAC
    slope_sac, intercept_sac, r_value_sac, p_value_sac, std_err_sac = stats.linregress(GA_list_Sac, param_Sac)
    predicted_y = slope_sac * GA_list_Sac + intercept_sac
    r_squared_sac = r_value_sac ** 2
    axs.scatter(GA_list_Sac, param_Sac, color = 'blue')
    axs.plot(GA_list_Sac, predicted_y, label=f'Saccular Stage - R²(Sac) = {r_squared_sac:.4f}', color='red')
    axs.set_xlabel('GA in weeks', fontsize=15)
    axs.set_ylabel('f', fontsize=15)
    axs.legend()
    axs.set_title(title, fontsize=18, fontdict=font_properties)


if __name__ == '__main__':
    torch.manual_seed(123)
    args = configure_solver_args()
    b_values = args.b_values
    bounds = args.bounds
    FetalData = args.FetalData
    FetalSeg = args.FetalSeg
    DATA_ID = args.DATA_ID
    FetalGA = args.FetalGA
    args  = configure_ga_corrs_analysis_args(args)
    if not isinstance(args.baseline_methods_results, str): args.baseline_methods_results = 'no_path'
    if os.path.exists(args.baseline_methods_results):
        with open(args.baseline_methods_results, "r") as json_file:
            bl_methods_results_all_data = json.load(json_file)

    else:

        bl_methods_results_all_data = calc_bl_all_dataset(b_values, bounds, FetalData, FetalSeg, DATA_ID, FetalGA)
        
    calc_only_test_cases = True # calculate only the cases that are not in hp_tune_cases
    calc_just_all_cases = True # calculate the corelation for all cases together without splitting them to major and minor motion
    args.comment = f'group{args.group}_test_cases_final'
    test_cases = [case for case in bl_methods_results_all_data['cases'] if case not in args.hp_tune_cases]

    if not calc_only_test_cases:
        test_cases = bl_methods_results_all_data['cases']
        hp_tune_cases = []

    test_ga = [val for i, val in enumerate(bl_methods_results_all_data['ga']) if bl_methods_results_all_data['cases'][i] in test_cases]

    for method in  ['SLS_TRF','Affine_TRF', 'SyN_TRF_b0', 'SyN_TRF_next_b', 'Iter_SyN_TRF']:
        _ , bl_methods_results_all_data[method] = separate_results(bl_methods_results_all_data[method], test_cases, cases = bl_methods_results_all_data['cases'], ga = bl_methods_results_all_data['ga'])
        bl_methods_results_all_data['cases'] = test_cases
        bl_methods_results_all_data['ga'] = test_ga

    if args.ivim_solver_results_path != None:
        with open(args.ivim_solver_results_path, "r") as json_file:
            ivimnet_results_all_cases = json.load(json_file)
            ivimnet_results_all_cases, _ = separate_results(ivimnet_results_all_cases, args.hp_tune_cases)
    else:
        ivimnet_results_all_cases = calc_solver_results(args, FetalData, FetalSeg, DATA_ID, FetalGA, b_values, bounds, cases = test_cases, use_sls_trf=False)
            
        
    if args.voxelmorph_solver_results_path != None:
        with open(args.voxelmorph_solver_results_path, "r") as json_file:
            voxel_morph_results_all_cases = json.load(json_file)
            voxel_morph_results_all_cases, _ = separate_results(voxel_morph_results_all_cases, args.hp_tune_cases)

    else:
        voxel_morph_results_all_cases = calc_solver_results(args, FetalData, FetalSeg, DATA_ID, FetalGA, b_values, bounds, only_reg=True, use_sls_trf = True)


    # calculate correlation for baseline methods:
    non_motion_methods_corrs = {}
    motion_methods_corrs = {}
    all_cases_corrs = {}
    for method in ['SLS_TRF','Affine_TRF', 'SyN_TRF_b0', 'SyN_TRF_next_b', 'Iter_SyN_TRF']:
        if calc_just_all_cases:
            # all group cases:
            ga = bl_methods_results_all_data['ga']
            all_cases_corrs[method] = calc_corr(bl_methods_results_all_data[method], ga)  
        else:
            bl_motion_results, bl_non_motion_results = separate_results(bl_methods_results_all_data[method], args.non_motion_cases, cases = bl_methods_results_all_data['cases'], ga = bl_methods_results_all_data['ga'])
            # minor motion cases:
            ga = bl_non_motion_results['ga']
            non_motion_methods_corrs[method] = calc_corr(bl_non_motion_results, ga)
            # major motion cases:
            ga = bl_motion_results['ga']
            motion_methods_corrs[method] = calc_corr(bl_motion_results, ga) 
            # all group cases:
            ga = bl_methods_results_all_data['ga']
            all_cases_corrs[method] = calc_corr(bl_methods_results_all_data[method], ga)  
               
    titles = {'SLS_TRF':'SLS-TRF', 
            'Affine_TRF': 'Affine - Reg to b0', 
            'SyN_TRF_b0' : 'SyN - Reg to b0', 
            'SyN_TRF_next_b' : 'SyN - Reg to next b' , 
            'Iter_SyN_TRF' : 'Iterative SyN - TRF', 
            'voxelMorph_solver\n+SLS_TRF' : 'VoxelMorph',
                        'IVIM_Morph_solver' : 'IVIM-Morph', 
            }
    if calc_just_all_cases == False:
        # calculate correlations for ivim morph solver
        ivim_morph_solver_motion_results, ivim_morph_solver_non_motion_results = separate_results(ivimnet_results_all_cases, args.non_motion_cases)

        non_motion_methods_corrs['ivimnet'] = calc_corr(ivim_morph_solver_non_motion_results)
        motion_methods_corrs['ivimnet'] = calc_corr(ivim_morph_solver_motion_results)
        all_cases_corrs['ivimnet'] = calc_corr(ivimnet_results_all_cases)

        # calculate correlations for voxelmorph solver
        voxelmorph_solver_motion_results, voxelmorph_solver_non_motion_results = separate_results(voxel_morph_results_all_cases, args.non_motion_cases)

        non_motion_methods_corrs['voxelmorph'] = calc_corr(voxelmorph_solver_non_motion_results)
        motion_methods_corrs['voxelmorph'] = calc_corr(voxelmorph_solver_motion_results)
        all_cases_corrs['voxelmorph'] = calc_corr(voxel_morph_results_all_cases)

        motion_df = pd.DataFrame(motion_methods_corrs).T
        non_motion_df = pd.DataFrame(non_motion_methods_corrs).T
        all_cases_df = pd.DataFrame(all_cases_corrs).T
        print('Major motion cases correlations')
        print(motion_df)
        print('Minor motion cases correlations')
        print(non_motion_df)
        print('All cases correlations')
        print(all_cases_df)
        non_motion_df.to_csv(f'non_motion_{args.comment}.csv')
        motion_df.to_csv(f'motion_{args.comment}.csv')
        all_cases_df.to_csv(f'all_cases_{args.comment}.csv')


        # plot fit lines for non motion cases
        figs, axs = plt.subplots(3,3, figsize = (15,15))
        figs.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.suptitle('Minor Motion Cases', fontsize=30)
        ind = np.argsort(ivim_morph_solver_non_motion_results['ga'])
        GA_list = np.array([ivim_morph_solver_non_motion_results['ga'][i] for i in ind])
        param = np.array([ivim_morph_solver_non_motion_results['f'][i] for i in ind])
        plot_fit(GA_list, param, axs[0,0], 'IVIM-Morph')

        ind = np.argsort(voxelmorph_solver_non_motion_results['ga'])
        GA_list = np.array([voxelmorph_solver_non_motion_results['ga'][i] for i in ind])
        param = np.array([voxelmorph_solver_non_motion_results['f'][i] for i in ind])
        plot_fit(GA_list, param, axs[0,1], 'VoxelMorph')

        inds = [[0,2], [1,0],[1,1],[1,2], [2,0]]
        for i, method in enumerate(['SLS_TRF','Affine_TRF', 'SyN_TRF_b0', 'SyN_TRF_next_b', 'Iter_SyN_TRF']):
            _, bl_non_motion_results = separate_results(bl_methods_results_all_data[method], args.non_motion_cases, cases = bl_methods_results_all_data['cases'], ga = bl_methods_results_all_data['ga'])
            ga = bl_non_motion_results['ga']
            ind = np.argsort(ga)
            GA_list = np.array([ga[i] for i in ind])
            param = np.array([bl_non_motion_results['f'][i] for i in ind])
            plot_fit(GA_list, param, axs[inds[i][0],inds[i][1]], titles[method])
        plt.savefig(f'all_cases_corrs_non_motion_{args.comment}.png')
        plt.close()

        # plot fit lines for motion cases
        figs, axs = plt.subplots(3,3, figsize = (15,15))
        figs.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.suptitle('Major Motion Cases', fontsize=30)
        ind = np.argsort(ivim_morph_solver_motion_results['ga'])
        GA_list = np.array([ivim_morph_solver_motion_results['ga'][i] for i in ind])
        param = np.array([ivim_morph_solver_motion_results['f'][i] for i in ind])
        plot_fit(GA_list, param, axs[0,0], 'IVIM-Morph')

        ind = np.argsort(voxel_morph_results_all_cases['ga'])
        GA_list = np.array([voxel_morph_results_all_cases['ga'][i] for i in ind])
        param = np.array([voxel_morph_results_all_cases['f'][i] for i in ind])
        plot_fit(GA_list, param, axs[0,1], 'VoxelMorph')

        inds = [[0,2], [1,0],[1,1],[1,2], [2,0]]
        for i, method in enumerate(['SLS_TRF','Affine_TRF', 'SyN_TRF_b0', 'SyN_TRF_next_b', 'Iter_SyN_TRF']):
            bl_motion_results, _ = separate_results(bl_methods_results_all_data[method], args.non_motion_cases, cases = bl_methods_results_all_data['cases'], ga = bl_methods_results_all_data['ga'])
            ga = bl_motion_results['ga']
            ind = np.argsort(ga)
            GA_list = np.array([ga[i] for i in ind])
            param = np.array([bl_motion_results['f'][i] for i in ind])
            plot_fit(GA_list, param, axs[inds[i][0],inds[i][1]],titles[method])
        plt.savefig(f'all_cases_corrs_motion_{args.comment}.png')
        plt.close()


    else:
        # calculate correlations for ivim-morph solver
        all_cases_corrs['ivimnet'] = calc_corr(ivimnet_results_all_cases)
        # calculate correlations for voxelmorph solver
        all_cases_corrs['voxelmorph'] = calc_corr(voxel_morph_results_all_cases)
        all_cases_df = pd.DataFrame(all_cases_corrs).T

        print('All cases correlations')
        print(all_cases_df)
        all_cases_df.to_csv(f'all_cases_{args.comment}.csv')
        # plot fit lines for all cases 
        font_properties = {
        'size': 30,
        'fontweight': 'bold',
        'fontfamily': 'serif'
         }
        figs, axs = plt.subplots(3,3, figsize = (17,17))
        figs.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.suptitle(f'Group {args.group} Test Cases',  fontweight= 'bold', fontsize=30,   fontfamily= 'serif')
        ind = np.argsort(ivimnet_results_all_cases['ga'])
        GA_list = np.array([ivimnet_results_all_cases['ga'][i] for i in ind])
        param = np.array([ivimnet_results_all_cases['f'][i] for i in ind])
        plot_fit(GA_list, param, axs[0,0], 'IVIM-Morph')

        ind = np.argsort(voxel_morph_results_all_cases['ga'])
        GA_list = np.array([voxel_morph_results_all_cases['ga'][i] for i in ind])
        param = np.array([voxel_morph_results_all_cases['f'][i] for i in ind])
        plot_fit(GA_list, param, axs[0,1], 'VoxelMorph')

        inds = [[0,2], [1,0],[1,1],[1,2], [2,0]]
        for i, method in enumerate(['SLS_TRF','Affine_TRF', 'SyN_TRF_b0', 'SyN_TRF_next_b', 'Iter_SyN_TRF']):
            ga = bl_methods_results_all_data[method]['ga']
            ind = np.argsort(ga)
            GA_list = np.array([ga[i] for i in ind])
            param = np.array([bl_methods_results_all_data[method]['f'][i] for i in ind])
            plot_fit(GA_list, param, axs[inds[i][0],inds[i][1]], titles[method])
        axs[2,1].axis('off')
        axs[2,2].axis('off')
        plt.savefig(f'all_cases_corrs_{args.comment}.png',  bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
