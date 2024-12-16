import yaml
import argparse
import matplotlib.pyplot as plt

import numpy as np
import torch

from utils.load import loadModel
from utils.interpolate import nonlinearFnc
from activation_dataset import loadActivationCsv

# python activation_valid.py plotResult_withX --test_name result --exps Activation_learning 
# python activation_valid.py plotResult_withX --test_name result --exps AFMM_N8000_LS 
# python activation_valid.py plotResult_withX --test_name result --exps AFMM_N8000
# python activation_valid.py plotResult_withX --test_name result2 --exps AFMM_lin2 --data_root nonlinear2.csv
def plotResult_withX(test_name, exps, root, **kwargs):
    model = loadModel(exps[0])
    with open(f"activation_logs/{exps[0]}/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
        
    x, y = loadActivationCsv(root=root,
                             N=config_dict['norm'],
                             logscale=config_dict['logscale'],
                             log_N=config_dict['log_N'],
                             moveX=config_dict['moveX'])
    
    Ys = []
    for val in x:
        val = torch.Tensor([val]).unsqueeze(-1).cuda()
        y_temp = model(val)
        # print(y_temp.shape)
        Ys.append(y_temp[0][0][0].item())
        
    SE = []
    for y_gt, y_nn in zip(y, Ys):
        SE.append((y_gt - y_nn)**2)
    MSE = np.mean(SE)
    
    plt.plot(x, y, label='Desired')
    plt.plot(x, Ys, label='NN generated')
    plt.title(f'Result plot of desired act. fnc. and NN generated\nMSE = {MSE:.2f}')
    plt.legend()
    plt.savefig(f'images/{test_name}.png', dpi=300)
    
# python activation_valid.py plotResult_withIntp --test_name interpolate --exps AFMM_N8000
# python activation_valid.py plotResult_withIntp --test_name interpolate --exps AFMM_N8000_LS
def plotResult_withIntp(test_name, exps, root, **kwargs):
    with open(f"activation_logs/{exps[0]}/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    model = loadModel(exps[0])
    
    data = loadActivationCsv(root=root,
                             N=config_dict['norm'],
                             logscale=config_dict['logscale'],
                             log_N=config_dict['log_N'])
    
    x = np.arange(-30, 30, 0.01)
    len_x = len(x)
    Y_gt = []
    for x_val in x:
        Y_gt.append(nonlinearFnc(x_val, data))
    Y_gt = np.array(Y_gt)
        
    X_nn = torch.Tensor(x).cuda().reshape([len_x, 1, 1])
    Y_nn = model(X_nn).squeeze().cpu().detach().numpy()
    
    RMSE = np.mean((Y_gt - Y_nn)**2)**0.5
    
    plt.plot(x, Y_gt, label='Desired')
    plt.plot(x, Y_nn, label='NN generated')
    plt.title(f'Result plot of desired act. fnc. and NN generated\nRMSE = {RMSE:.5f}')
    plt.legend()
    plt.savefig(f'images/{test_name}.png', dpi=300)
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('func_type', type=str)
    parser.add_argument('--test_name', type=str, default='')
    parser.add_argument('--data_root', type=str, default='nonlinear.csv')
    parser.add_argument('--norm', type=float)
    parser.add_argument('--logscale', action="store_true")
    parser.add_argument('--log_N', type=float)
    parser.add_argument('--exps', action='append', default=None, help='experiment names')
    parser.add_argument('--caps', action='append', default=None, help='experiment captions')
    args = parser.parse_args()
    
   
    func = globals()[args.func_type]
    func(
        test_name = args.test_name,
        exps = args.exps,
        caps = args.caps,
        root = args.data_root
    )



