import argparse
import torch
import numpy as np
def configure_ga_corrs_analysis_args(args):
    args.group = 2
    args.baseline_methods_results = None
    if args.group == 1:
        args.weight_grad = 0.015
        args.weight_ncc = 0.1
        args.weight_fit = 10
        args.ivim_solver_results_path = '/tcmldrive/NogaK/IVIM-Morph/results/solver_ivim_params/new_groups/solver_IVIM_Morph_ivim_params_weight_grad_loss0.015_w_ncc_0.1_w_fit_10_new_group1_23_cases_2023_12_27_15_53_02.json'
        args.voxelmorph_solver_results_path ='/tcmldrive/NogaK/IVIM-Morph/results/solver_ivim_params/new_groups/solver_voxelMorph_ivim_params_weight_grad_loss0.015_w_ncc_0.1_w_fit_10_new_group1_23_cases_2023_12_27_16_41_22.json'

        args.hp_tune_cases = ['Case844','Case800','Case816','Case867',
                          'Case801','Case756','Case828', 'Case603',
                          'Case612','Case758','Case777','Case779',
                          'Case818','Case819', 'Case820','Case900']
    elif args.group == 2:
        args.weight_grad = 0.015
        args.weight_ncc = 0.8
        args.weight_fit = 0.5
        args.ivim_solver_results_path = '/tcmldrive/NogaK/IVIM-Morph/results/solver_ivim_params/new_groups/solver_IVIM_Morph_ivim_params_weight_grad_loss0.015_w_ncc_0.8_w_fit_0.5_new_group2_23_cases_2023_12_27_14_28_46.json'
        args.voxelmorph_solver_results_path ='/tcmldrive/NogaK/IVIM-Morph/results/solver_ivim_params/new_groups/solver_voxelMorph_ivim_params_weight_grad_loss0.015_w_ncc_0.8_w_fit_0.5_new_group2_23_cases_2023_12_27_15_17_19.json'

        args.hp_tune_cases = ['Case877','Case856','Case860', 'Case879',
                           'Case681','Case817','Case708', 'Case710',
                           'Case709','Case794','Case798','Case735',
                           'Case737','Case770','Case807','Case887']
        # separate bl methods results to motion and non_motion
    args.non_motion_cases = ['Case603','Case612','Case681','Case708','Case710',
                        'Case758','Case798','Case820','Case844','Case856', 
                        'Case650','Case693', 'Case777' ,'Case817']

    return args

def configure_solver_args():
        # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    # parser.add_argument('--std', type=float, default=0.3, help='std for syntentic deformation field (default: 0)')
    parser.add_argument('--num-repeats', type=int, default=10,
                        help='number of synthetic deformation field to genarate (default: 10)')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    # parser.add_argument('--epochs', type=int, default=1500,
    #                     help='number of training epochs (default: 1500)')
    parser.add_argument('--epoch-per-std', type=int, default=10,
                        help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=10,
                        help='frequency of model saves (default: 100)')
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--b-values', type=int, nargs='+',
                        help='list of unet encoder filters (default: 0 50 100 200 400 600)')
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=5,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--n', type=int, default=3,
                        help='if fit s0 n is 4 else n is 3')
    # loss hyperparameters
    parser.add_argument('--image-loss', default='mse',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--weight-grad', type=float, default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--weight-dice', type=float, default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--weight-fit', type=float , default = 1,
                        help='weight of deformation loss (default: 0.01)')

    parser.add_argument('--checkpoints', type=str , default = 'checkpoints',
                        help='checkpoints')
    parser.add_argument('--patience', type=int , default = 10,
                        help='number of bad epochs before stopping training')

    args, unknown  = parser.parse_known_args()

    args.weight_grad = 0.015
    args.weight_ncc = 0.1
    args.weight_fit = 1
    args.weight_ncc2 = 0 #keep this!!!!
    args.tensorboard = False
    args.comment = 'lung_seg_exp_grad_0.03_ncc0.1_fit1'
    args.iters = 25
    args.early_stopping = 25
    args.lr = 0.001 # 0.0005 dont change!!!
    args.grad_clip_norm = None
    args.batch_size = 1
    args.lr_schedular = True
    args.lr_patience = 10
    args.lr_factor = 0.9
    args.ser_loss_norm = 'l2'
    args.big_net = False
    args.reg_s0 = False
    args.enc = [16, 32, 32, 32]#
    args.dec = [32, 32, 32, 32, 32, 16, 16]#
    args.wrap_costume = False #keep this!!!!
    args.interpol_method = 'nearest'
    args.img_factor = 1
    args.int_steps = 5
    args.min_iter = 500
    args.init_iter = 1
    args.int_downsize = 2
    args.roi_fit_loss = False
    args.b_values = np.array([0, 50, 100, 200, 400, 600])
    args.bounds =  [[0.0005, 0.05, 0.01, 1], [0.003, 0.55, 0.1, 1]]
    args.FetalData = torch.load('/tcmldrive/NogaK/IVIM-Morph/data/Data_2D.pt')
    args.FetalSeg = torch.load('/tcmldrive/NogaK/IVIM-Morph/data/Data_seg_2D.pt')
    args.DATA_ID = torch.load('/tcmldrive/NogaK/IVIM-Morph/data/Data_case_2D.pt')
    args.FetalGA = torch.load('/tcmldrive/NogaK/IVIM-Morph/data/Data_GA_2D.pt')

    args.group = 1
    if args.group == 1:
        args.weight_grad = 0.015
        args.weight_ncc = 0.1
        args.weight_fit = 10
    elif args.group == 2:
        args.weight_grad = 0.015
        args.weight_ncc = 0.8
        args.weight_fit = 0.5
    return args
