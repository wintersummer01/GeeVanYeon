import random
import sys
import os
import configargparse
from pynvml.smi import nvidia_smi
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

from activation_model import *
from activation_dataset import loadData, loadActivationCsv
from utils.yaml import dumpyaml
from utils.progress import progressMeter, logger_configuration
from utils.checkpoint import load_ckpt, save_ckpt

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


def train_one_epoch(model, epoch, dataloader, criterion, optimizer, writer, logger, 
                    clip_max_norm, gpu_id):
    model.train()
    total_len = len(dataloader.dataset)
    progress = progressMeter('train', writer, logger, total_len, epoch, use_pbar=True)
    
    for x, y in dataloader:
        x = x.to(gpu_id)
        y = y.to(gpu_id)
        optimizer.zero_grad()

        y_hat = model(x)

        loss = criterion(y, y_hat)
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        out_dict = {}
        out_dict['batch_size'] = len(x)
        out_dict['loss'] = loss
        out_dict['PSNR'] = 10 * torch.log10(1/loss)
        progress.update(out_dict)
    progress.verbose_states()
    avg_loss = progress.write_summary()
    return avg_loss


def valid_with_plot(model, MSE, raw_data, save_path):
    x, y = raw_data
    plt.clf()
    
    if model is not None:
        with torch.no_grad():
            x_range = np.arange(-30, 30, 0.01)
            X_nn = torch.Tensor(x_range).cuda().reshape([len(x_range), 1, 1])
            Y_nn = model(X_nn).squeeze().cpu().detach().numpy()
        plt.plot(x_range, Y_nn, label='NN generated')
    
    plt.plot(x, y, label='Desired')
    plt.title(f'Result plot of desired act. fnc. and NN generated\nRMSE = {MSE**0.5:.5f}')
    plt.legend()
    plt.savefig(f'{save_path}/result.png', dpi=300)
    


def parse_args(argv):
    parser = configargparse.ArgumentParser()
    parser.add_argument('config', is_config_file=True, help='Path to config file to replace defaults.')
    parser.add_argument('--experiment_name', type=str, default='experiment names')
    
    # Model
    parser.add_argument('--model_class', type=str, help='Model class')
    parser.add_argument('--hidden_dim', type=int, help='calculation dimension of coder')
    parser.add_argument('--num_layers', type=int, help='output dimension of coder')
    parser.add_argument('--initializer', type=float, help='initial value of the model')
    
    # Training
    parser.add_argument('--loss_type', type=str, default='mse', help='Type of loss')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Starting epoch idx. Only for when pretrain_ckpt does not exist')
    parser.add_argument('--epochs', type=int, default=300, help='End epoch idx.')
    parser.add_argument('--learning_rate', type=float, default=1.0e-4, help='Symbols per pixel')
    parser.add_argument('--lr_milestone_epoch', action='append', default=[])
    parser.add_argument('--clip_max_norm', type=float, default=0, help='Gradient clipping for stable training.')
    parser.add_argument('--shutup', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    
    # Dataset
    parser.add_argument('--norm', type=float)
    parser.add_argument('--logscale', action="store_true")
    parser.add_argument('--log_N', type=float)
    parser.add_argument('--moveX', type=float)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # logging
    parser.add_argument('--log_dir', type=str, help='logs/')
    parser.add_argument('--save_model', action="store_true", help="Save the models")
    parser.add_argument('--save_every', type=int, default=50, help='Frequency (epoch) of saving the model.')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='Path to pretrained model weights.')

    # Environments
    parser.add_argument('--seed', type=int, default=1030, help='For reproduction')
    parser.add_argument('--gpu_ids', action='append', default=[], help='GPU id to use.')

    args = parser.parse_args(argv)
    return args



def main(argv):
    # Argument parsing
    args = parse_args(argv)
    if args.shutup:
        argv.remove('--shutup')
        argument = ' '.join(argv)
        os.system(f"nohup python train.py {argument} > /dev/null 2>&1 &")
        return 
    
    # Single GPU Learning
    local_gpu_id = None
    nvsmi = nvidia_smi.getInstance()
    res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.used")["gpu"]
    for i in args.gpu_ids:
        i = int(i)
        mem_used = res_util_list[i]['fb_memory_usage']['used']
        gpu_util = res_util_list[i]['utilization']['gpu_util']
        if mem_used < 1000 and gpu_util == 0:
            local_gpu_id = i
            args.gpu = i
            torch.cuda.set_device(local_gpu_id)
            print(f"Using GPU No. {local_gpu_id}")
            break
    if local_gpu_id is None:
        print("No GPUs Available!")
        return
        
    # Reproduction
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # Dataset
    dataloader = loadData(
        data_root=  args.dataset_root,
        batch_size= args.batch_size,
        num_cpus=   args.num_workers,
        norm=       args.norm,
        logscale=   args.logscale,
        log_N=      args.log_N,
        moveX=      args.moveX,
        shuffle=    True,
    )
    raw_data = loadActivationCsv(root=   args.dataset_root,
                                 N=       args.norm,
                                 logscale=   args.logscale,
                                 log_N=      args.log_N,
                                 moveX=      args.moveX,
    )
    
    # Model
    args.model_params = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'initializer': args.initializer,
    }
    model = globals()[args.model_class](**args.model_params)
    model = model.cuda(local_gpu_id)
    print(summary(model, torch.zeros((1, 1), device='cuda')))
    
    # Training utils (criterion, optimizer, scheduler)
    if args.loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    milestone = []
    for num in args.lr_milestone_epoch:
        milestone.append(int(num))
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=0.5)

    # logger configuration
    save_path, logger = logger_configuration(args.log_dir, args.experiment_name, args.save_model)
    logger.info(args)
    dumpyaml(args, save_path)
    valid_with_plot(None, 0, raw_data, save_path)
        
    # Load Checkpoints
    if args.pretrain_ckpt == 'null':
        args.pretrain_ckpt = None
    if args.pretrain_ckpt:  # load from previous checkpoint
        model, optimizer, lr_scheduler, start_epoch = load_ckpt(
            args.pretrain_ckpt,
            local_gpu_id,
            model, 
            optimizer,
            lr_scheduler, 
        )
        logger.info(f"Loaded {args.pretrain_ckpt}")
    else:
        start_epoch = args.initial_epoch
        
    # Summary writer      
    writer = SummaryWriter(os.path.join(save_path, "tsb_logs"))
    
    best_loss = float("inf")
    old_lr = 'None'
    # Start training!
    for epoch in range(start_epoch, args.epochs):
        # Print learning rate update
        now_lr = optimizer.param_groups[0]['lr']
        if old_lr != now_lr:
            print(f"Learning rate change: {old_lr} -> {now_lr}")
            old_lr = now_lr
        
    
        loss = train_one_epoch(
            model=         model,
            epoch=         epoch,
            dataloader=    dataloader,
            criterion=     criterion,
            optimizer=     optimizer,
            writer=        writer,
            logger=        logger,
            clip_max_norm= args.clip_max_norm,
            gpu_id=        local_gpu_id
        )
        if args.plot:
            valid_with_plot(model, loss, raw_data, save_path)
        lr_scheduler.step()
        
        # Check best
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
    
        # Save CKPT
        if args.save_model:
            state_dict = model.state_dict()
            save_ckpt(save_path, 
                        epoch, 
                        state_dict, 
                        loss, 
                        optimizer, 
                        lr_scheduler, 
                        args.save_every, 
                        is_best
            )

if __name__ == "__main__":
    main(sys.argv[1:])