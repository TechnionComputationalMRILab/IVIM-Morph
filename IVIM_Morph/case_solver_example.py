import os
import sys
sys.path.append('./src')
import numpy as np
from BL_func import *
import torch 
import os
from Net import *
from IVIM_morph import solver_IVIM_Morph
from config import *
# os.environ['NEURITE_BACKEND'] = 'pytorch'
# os.environ['VXM_BACKEND'] = 'pytorch'
# sys.path.append('./voxelmorph')
# import voxelmorph as vxm  # nopep8

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
    args.comment = 'case_example'
    args.print_every = 100
    args.iters = 2500
    args.early_stopping = 25
    args.lr = 0.001 # 0.0005 dont change!!!
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
    args.interpol_method = 'linear'
    args.img_factor = 1
    args.int_steps = 7
    args.min_iter = 500
    args.init_iter = 1
    args.int_downsize = 2
    args.roi_fit_loss = False
    args.dimension = [6,96,96]
    args.b_values = np.array([0, 50, 100, 200, 400, 600])
    args.bounds =  [[0.0005, 0.05, 0.01, 1], [0.003, 0.55, 0.1, 1]]
    args.path_to_case = '//tcmldrive/NogaK/fetal_lung_case'
    return args


def load_case(path_to_case):
    data = None
    seg = None
    for file in os.listdir(path_to_case):
        if file.endswith('data.pt'):
            data = torch.load(os.path.join(path_to_case, file))
        if file.endswith('seg.pt'):
            seg = torch.load(os.path.join(path_to_case, file))

    assert data != None, 'No data was found in this path'
    return data, seg



    
if __name__ == '__main__':

    args = configure_solver_args()
    solver = solver_IVIM_Morph(args = args, device='cuda')
    dataset = load_case(args.path_to_case)
    warpdata, deformation_field, ivim_params = solver.fit(dataset)