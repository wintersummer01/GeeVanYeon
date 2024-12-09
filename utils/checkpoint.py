import os
import torch

def load_ckpt(checkpoint, local_gpu_id, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint, 
                            map_location=torch.device('cuda:{}'.format(local_gpu_id)))

    model.load_state_dict(checkpoint["state_dict"])  
    start_epoch = checkpoint["epoch"] + 1
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    print("Loaded", checkpoint)
    
    return model, optimizer, lr_scheduler, start_epoch
    
def save_ckpt(save_path, epoch, state_dict, loss, optimizer, lr_scheduler, save_every, is_best):
    state = {
            "epoch": epoch,
            "state_dict": state_dict,
            "loss": loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
    }
    # Save latest
    torch.save(state, os.path.join(save_path, f"latest_ckpt_ep{epoch}.pth.tar"))
    for fname in os.listdir(save_path):
        if 'latest_ckpt' in fname and fname != f"latest_ckpt_ep{epoch}.pth.tar":
            os.remove(os.path.join(save_path, fname))
    # Save every
    if (epoch+1) % save_every == 0:
        torch.save(state, os.path.join(save_path, 'weights', f"Epoch_{epoch}.pth.tar"))
    # Save best
    if is_best:
        torch.save(state, os.path.join(save_path, f"best_ckpt_ep{epoch}.pth.tar"))
        for fname in os.listdir(save_path):
            if 'best_ckpt' in fname and fname != f"best_ckpt_ep{epoch}.pth.tar":
                os.remove(os.path.join(save_path, fname))

    
    