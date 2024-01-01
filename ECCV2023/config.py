import os

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import torch
import ModelFitLoss

def config():
    config = dict()
    # general:
    config['seed'] = 37
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['b_vector'] = torch.tensor([0, 50, 100, 200, 400, 600], dtype=torch.float).to(config['device'])

    # data:
    config['table_path'] = 'DataMotionGA.xlsx'
    config['datebase_path'] = r'/tcmldrive/databases/Private/fetal_data'
    # train init:
    config['lr'] = 1e-4
    config['max_iter'] = 50
    config['early_stopping'] = 5
    config['crop_x'] = 96
    config['crop_y'] = 96
    config['crop_z'] = 16
    # reg model:
    config['enc_nf'] = [16, 32, 32, 32]
    config['dec_nf'] = [32, 32, 32, 32, 32, 16, 16]
    config['int_steps'] = 7  # 0
    # loss init
    config['int_downsize'] = 2
    sim_loss = torch.nn.L1Loss()

    deformation_loss = vxm.losses.Grad('l2').loss
    fit_loss = ModelFitLoss.ModelFit().loss

    alpha1 = 0.01
    alpha2 = 10

    config['losses'] = [sim_loss, deformation_loss, fit_loss]
    config['weights'] = [1, alpha1, alpha2]
    #
    config['Results_folder'] = './Results'
    config['Save_dict_path'] = os.path.join(config['Results_folder'], 'fit_per_case_VoxelMorph.json')
    config['plots_path'] = './plots'
    config['plot'] = True
    config['ADC_Convergence'] = 5
    config['BGthreshold'] = 100
    # for ADC_GA fit:
    config['case_list_valid'] = ['Case612', 'Case645', 'Case647', 'Case650', 'Case681', 'Case691',
                                 'Case693', 'Case697', 'Case708', 'Case709', 'Case710', 'Case735',
                                 'Case737', 'Case739', 'Case756', 'Case758', 'Case770', 'Case777',
                                 'Case779', 'Case794', 'Case798', 'Case800', 'Case801', 'Case807',
                                 'Case816', 'Case817', 'Case818', 'Case819', 'Case820', 'Case828',
                                 'Case844', 'Case856', 'Case860', 'Case867', 'Case877', 'Case879',
                                 'Case887', 'Case900']

    #
    return config
