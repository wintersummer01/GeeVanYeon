import os
import yaml
import torch
import activation_model

def loadModel(exp, log_dir='activation_logs/', device='cuda'):
    work_dir = log_dir + exp
    
    with open(f"{work_dir}/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_name = config_dict["model_class"]
    model_params = config_dict["model_params"]
    for fname in os.listdir(work_dir):
        if "best" in fname:
            break
    model_ckpt = f"{work_dir}/{fname}"
    
    model = getattr(activation_model, model_name)(**model_params).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    
    model.load_state_dict(ckpt["state_dict"], strict=False)
    
    return model 