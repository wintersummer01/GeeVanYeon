import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def loadActivationCsv(root, N, logscale, log_N, moveX=0):
    with open(root, 'r') as f:
        lines = f.readlines()
    
    # Load x, y
    x = []
    y = []
    for line in lines[1:]:
        elements = line.strip().split(',')
        if elements[0] == '':
            break
        x.append(float(elements[0]))
        y.append(float(elements[1]))
    
    # increment y
    num_y = len(y)
    mini = float('inf')
    for i in list(reversed(range(num_y))):
        if y[i] > mini:
            y[i] = mini
        mini = y[i]
        
    # value correction
    rmv_lst = []
    for i in range(num_y):
        y[i] *= N
        if logscale:
            y[i] = float(np.log10(y[i]))
            y[i] *= log_N
        
        # xmove
        x[i] += moveX
        
        if x[i] > 4 or x[i] < -6:
            rmv_lst.append(i)
    
    rmv_lst.reverse()
    for i in rmv_lst:
        del x[i]
        del y[i]
    
    
    return x, y



class activationFncData(Dataset):
    def __init__(self, root, N=15500, logscale=False, log_N=0, moveX=0):
        self.x, self.y = loadActivationCsv(root, N, logscale, log_N, moveX)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.Tensor([self.x[idx]]).unsqueeze(-1)
        y = torch.Tensor([self.y[idx]]).unsqueeze(-1)
        
        return x, y

    
def loadData(data_root='nonlinear.csv', batch_size=8, num_cpus=16, shuffle=True, norm=1, logscale=False, log_N=0, moveX=0):
    dataset = activationFncData(data_root, N=norm, logscale=logscale, log_N=log_N, moveX=moveX)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_cpus,
                            persistent_workers=True
                            )
    return dataloader

if __name__ == '__main__':
    dataloader = loadData()
    for d in dataloader:
        print(d[0].shape)
        break