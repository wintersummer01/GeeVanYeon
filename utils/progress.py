import sys
import os
import logging
from tqdm import tqdm

import torch
import torch.distributed as dist

class progressMeter:
    def __init__(self, desc, writer, logger, total_data_len, epoch, use_pbar):
        self.states = torch.zeros(1, device='cuda', dtype=torch.float32)
        self.states_name_idx = {'batch_size':0}
        self.num_states = 0
        self.states_master = torch.zeros(1, device='cuda', dtype=torch.float32)
        
        self.writer = writer
        self.logger = logger
        self.use_pbar = use_pbar
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}: '
        self.total_data_len = total_data_len
        if use_pbar:
            self.pbar = tqdm(total=total_data_len, ncols=70, file=sys.stdout, desc=self.desc)
        
    def update(self, states):
        assert 'batch_size' in states.keys(); "Batch size should be provided in states with key: 'batch_size'"
        self.states[0] = states.pop('batch_size')
        for name, val in states.items():
            if name not in self.states_name_idx.keys():
                self.num_states += 1
                self.states_name_idx[name] = self.num_states
                self.states = torch.concat((self.states, torch.zeros((1), device='cuda')))
            idx = self.states_name_idx[name]
            self.states[idx] = val.data * self.states[0]
        del states
        
        more = len(self.states) - len(self.states_master)
        if more > 0:
            self.states_master = torch.concat((self.states_master, torch.zeros((more), device='cuda')))
        self.states_master += self.states
    
    def verbose_states(self):
        avg_states = self.states_master / self.states_master[0]
        if self.use_pbar:
            print('\r', end='')
            self.pbar.update(int(self.states[0]))
        else:
            print(self.desc, end='')
        out = ''
        for name, idx in self.states_name_idx.items():
            if name != 'batch_size':
                out += f" | {name}: {avg_states[idx]:.3f}"
        print(out, end='')
        epoch_prog = f"({self.states_master[0]:4.0f}/{self.total_data_len:4.0f})"
        self.logger.info(self.desc+epoch_prog+out)
        if not self.use_pbar:
            print("")
    
    def write_summary(self):
        avg_states = self.states_master / self.states_master[0]
        for name, idx in self.states_name_idx.items():
            if name != 'batch_size':
                self.writer.add_scalar(name, avg_states[idx], self.epoch)
        if self.use_pbar:
            self.pbar.close()
        return avg_states[1] # Average Loss
    
            
def logger_configuration(log_dir, exp_name, save_ckpt):
    logger = logging.getLogger("DeepJSCC")
    save_path = os.path.join(log_dir, exp_name)
    log_path = save_path + '/logging.log'
        
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if save_ckpt:
        weight_path = os.path.join(save_path, "weights")
        if not os.path.exists(weight_path):
            os.makedirs(weight_path, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    filehandler = logging.FileHandler(log_path)
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    
    return save_path, logger