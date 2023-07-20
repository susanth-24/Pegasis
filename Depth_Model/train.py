from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from .Depth_model.model import Depth_Model
from .Depth_model.dataloader import NYUV2Dataset
from .Depth_model.loss import ssim
from utils import DepthNorm

#these are example file paths
csv_file = '/kaggle/input/nyu-depth-v2/nyu_data/data/nyu2_train.csv'
base_dir = '/kaggle/input/nyu-depth-v2/nyu_data/'

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])
batch_size = 8
output_shape = (batch_size, 1, 240, 320)
nyu_dataset = NYUV2Dataset(csv_file, base_dir=base_dir, output_shape=output_shape, transform=transform)

data_loader = torch.utils.data.DataLoader(nyu_dataset, batch_size=batch_size, shuffle=True)

model = Depth_Model()
model.cuda()

epochs = 3
l1_criterion = nn.L1Loss()
learning_rate = 0.0001
optimizer = torch.optim.Adam( model.parameters(), learning_rate )

def train():
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for batch in pbar:
            optimizer.zero_grad()
            inputs, targets = batch
            image = inputs.cuda()
            depth = targets.cuda(non_blocking=True)

            depth_n = DepthNorm(depth)

            output = model(image)
            depth_n.shape

            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, 11)) * 0.5, 0, 1)
            loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Loss': loss.item()})

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")

train()
#save the model
torch.save(model.state_dict(), '/kaggle/working/pegasisv5_epoch_4.pth')
