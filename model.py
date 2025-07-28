import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.io import decode_image
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import os
from PIL import Image
import zipfile

zip_path = 'Data/Images.zip'
extract_to = 'Data'

os.makedirs(extract_to, exist_ok=True)
os.makedirs('weights', exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

df = pd.read_parquet('Data/data.parquet', engine="fastparquet")
train_df, test_df = train_test_split(df, test_size=0.2)

image_size = 256
image_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    v2.GaussianNoise(sigma = 0.2)
])

class frog_dataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        self.pulse = df['Absolute Pulse']
        self.phase = df['Phase']
        self.codes = df['Trace Name']

    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.codes.iloc[idx])
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        image = torch.tensor(image, dtype = torch.float).unsqueeze(0)
        if self.transforms:
            image = self.transforms(image)
        pulse = self.pulse.iloc[idx]
        pulse = (pulse - np.min(pulse))/(np.max(pulse) - np.min(pulse))
        phase = self.phase.iloc[idx]
        y = np.concatenate([pulse, phase])
        y = torch.tensor(y, dtype=torch.float)

        return image, y
    
image_dir = 'Data/Images'
best_weights = 'weights/best_weight.pth'
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True
num_epochs = 100

train_ds = frog_dataset(image_dir, train_df, transforms=image_transforms)
test_ds = frog_dataset(image_dir, test_df, transforms=image_transforms)

train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True)
test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = False)

class FROG_NET(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 64,  output_size = 1024, image_size = 256):
        super(FROG_NET, self).__init__()
        self.in_channels = in_channels
        self.output_size = output_size
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.convT2 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels * 4, out_channels * 2, kernel_size=2, stride = 2, padding=0)
        )

        self.convT3 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=2, stride = 2, padding=0)
        )

        self.convT4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride = 2, padding=0)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, output_size)
        )


    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)

        d2 = self.convT2(e3)
        d3 = self.convT3(d2)
        d4 = self.convT4(d3)
        
        output = self.linear(d4)
        
        return output

class Trainer:
    def __init__(self, model, train_loader, test_loader, device):
        super(Trainer, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.least_loss = float('inf')

    def train(self, epochs):
        train_loss_list = np.zeros(epochs)
        test_loss_list = np.zeros(epochs)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            train_loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
            for x, y in train_loop:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                self.optimizer.zero_grad()
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())

            avg_train_loss = running_loss / len(self.train_loader)
            train_loss_list[epoch] = avg_train_loss

            test_loss = self.evaluate(epoch)
            test_loss_list[epoch] = test_loss

            if test_loss < self.least_loss:
                torch.save(self.model.state_dict(), 'best_weights.pth')
                self.least_loss = test_loss

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f'model_weights_{epoch}.pth')

    def evaluate(self, epoch=None):
        self.model.eval()
        total_loss = 0.0
        test_loop = tqdm(self.test_loader, desc=f"Epoch {epoch+1 if epoch is not None else '?'} [Evaluating]", leave=False)
        with torch.no_grad():
            for x, y in test_loop:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                total_loss += loss.item()
                test_loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FROG_NET()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    t = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, device=DEVICE)
    t.train(num_epochs)


