import torch
from torch.nn import MSELoss
from torch.optim import Adam
from data import prepare_data
from model import AutoEncoders
from utils import MSE, PSNR


PATH = os.getcwd() + "\\Data\\thumbnails128x128"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

train_dataloader, val_dataloader = prepare_data(PATH, split_size=0.8, batch_size=16)

loss_module = MSELoss()

normal_ae_mse = AutoEncoders()
normal_optimizer_mse = Adam(normal_ae_mse.parameters(), lr=1e-3)
normal_ae_mse = normal_ae_mse.to(device)

train_model(normal_ae_mse, "normal", 0.2, normal_optimizer_mse, train_dataloader, val_dataloader, loss_module)


normal_ae_mse.eval()

images, _ = next(iter(val_dataloader))
images = images.float().to(device)
output = normal_ae_mse(images)

imshow(images[1].cpu().detach(), normalize=False)

noisy_images = (images + torch.normal(0,0.2,images.shape)).clip(0,1)

imshow(images[1], normalize=False)

output = normal_ae_mse(noisy_images.to(device))

imshow(output[1].cpu().detach(), normalize=False)

print(f"The Mean Squared Error for the reconstruction (for an example batch) is:{MSE(images, output).mean()}")

print(f"The Peak Signal to Noise Ratio for the reconstruction (for an example batch) is:{PSNR(images, output).mean()}")
