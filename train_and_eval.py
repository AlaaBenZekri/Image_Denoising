import torch
import numpy as np
from utils import PSNR, MSE


def eval_model(model, val_dataloader, noise_type, noise_parameter):
    model.eval()
    psnr = []
    mse = []
    with torch.no_grad():
        for images, _ in val_dataloader:
            if (noise_type == "normal"):
                noisy_images = (images + torch.normal(0,noise_parameter,images.shape)).clip(0,1)
            elif (noise_type == "bernoulli"):
                a = noise_parameter*torch.ones(images.shape)
                noisy_images = images*torch.bernoulli(a)
            elif (noise_type == "poisson"):
                a = noise_parameter*torch.ones(images.shape)
                p = torch.poisson(a)
                p_norm = p/p.max()
                noisy_images = (images + p_norm).clip(0,1)
            images = images.to(device)
            noisy_images = noisy_images.to(device)
            preds = model(images)
            psnr.extend(PSNR(images.cpu().detach(), preds.cpu().detach()))
            mse.extend(MSE(images.cpu().detach(), preds.cpu().detach()))
        print(f"Peak Signal to Noise Ratio:   Mean: {np.array(psnr).mean()} || Std: {np.array(psnr).std()}")
        print(f"Mean Squared Error:   Mean: {np.array(mse).mean()} || Std: {np.array(mse).std()}")
        return np.array(psnr).mean(), np.array(mse).mean()

def train_model(model, noise_type, noise_parameter, optimizer, train_dataloader, val_dataloader, loss_module, target_type="clean", num_epochs=30):
    model.train()
    epoch_num = []
    mse_train = []
    mse_val = []
    psnr_train = []
    psnr_val = []
    mse = 0.0
    psnr = 0.0
    for epoch in range(num_epochs):
        for images, _ in train_dataloader:
            targets = torch.clone(images)
            if (noise_type == "normal"):
                images = (images + torch.normal(0,noise_parameter,images.shape)).clip(0,1)
            elif (noise_type == "bernoulli"):
                a = noise_parameter*torch.ones(images.shape)
                images = images*torch.bernoulli(a)
            elif (noise_type == "poisson"):
                a = noise_parameter*torch.ones(images.shape)
                p = torch.poisson(a)
                p_norm = p/p.max()
                images = (images + p_norm).clip(0,1)
            if (target_type == "noisy"):
                if (noise_type == "normal"):
                    targets = (targets + torch.normal(0,noise_parameter,targets.shape)).clip(0,1)
                elif (noise_type == "bernoulli"):
                    a = noise_parameter*torch.ones(targets.shape)
                    targets = targets*torch.bernoulli(a)
                elif (noise_type == "poisson"):
                    a = noise_parameter*torch.ones(targets.shape)
                    p = torch.poisson(a)
                    p_norm = p/p.max()
                    targets = (targets + p_norm).clip(0,1)           
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = loss_module(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch%3 == 0):
            print(f"********EPOCH {epoch+1}:********")
            epoch_num.append(epoch+1)
            print("Train set:")
            psnr, mse = eval_model(model, train_dataloader, noise_type, noise_parameter)
            psnr_train.append(psnr)
            mse_train.append(mse)
            print("Validation set:")
            psnr, mse = eval_model(model, val_dataloader, noise_type, noise_parameter)
            
            psnr_val.append(psnr)
            mse_val.append(mse)

            
