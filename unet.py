'''
if you are using google colab, use the following lines mentioned below

from google.colab import drive
drive.mount('/content/gdrive') your google drive

create a folder 'data' in your drive
!unzip gdrive/My\ Drive/data/data.zip > /dev/null
 '''

'''Import the required packages before proceeding and if you are using colab add % before each pip install

pip install segmentation-models-pytorch 
pip install segmentation-models-pytorch torchsummary
pip install segmentation-models

'''

#Imported the required modules
import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torchsummary as summary
import tensorflow as tf
import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import seaborn as sns


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat):
        super(ResidualBlock, self).__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.repeat = repeat

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels//4, kernel_size=(1, 1),
                      stride=(2, 2), padding=0),
            nn.BatchNorm2d(num_features=self.output_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_channels//4, out_channels=self.output_channels//4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=self.output_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_channels//4, out_channels=self.output_channels, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=self.output_channels),
        )
        self.conv_other = nn.Sequential(
            nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels//4, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=self.output_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_channels//4, out_channels=self.output_channels//4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=self.output_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_channels//4, out_channels=self.output_channels, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=self.output_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_first(x)
        for i in range(self.repeat - 1):
            x_2 = self.conv_other(x)
            x = x + x_2
            x = self.relu(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet152, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.conv2 = ResidualBlock(in_channels=64, out_channels=256, repeat=3)
        self.conv3 = ResidualBlock(in_channels=256, out_channels=512, repeat=8)
        self.conv4 = ResidualBlock(in_channels=512, out_channels=1024, repeat=36)
        self.conv5 = ResidualBlock(in_channels=1024, out_channels=2048, repeat=3)

    def forward(self, x):
        x_512 = x
        x = self.conv1(x)
        x_256 = x
        x = self.conv2(x)
        x_128 = x
        x = self.conv3(x)
        x_64 = x
        x = self.conv4(x)
        x_32 = x
        x_16 = self.conv5(x)
        return x_16, x_32, x_64, x_128, x_256, x_512

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(num_features = out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)
        x = self.convblock(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ResNet152 = ResNet152(in_channels=in_channels)
        self.Up_5 = Up(2048 + 1024, 256)
        self.Up_4 = Up(256 + 512, 128)
        self.Up_3 = Up(128 + 256, 64)
        self.Up_2 = Up(64 + 64, 32)
        self.Up_1 = Up(32 + 3, 16)
        self.Conv_1 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x_16, x_32, x_64, x_128, x_256, x_512 = self.ResNet152(x)
        x = self.Up_5(x_16, x_32)
        x = self.Up_4(x, x_64)
        x = self.Up_3(x, x_128)
        x = self.Up_2(x, x_256)
        x = self.Up_1(x, x_512)
        x = self.Conv_1(x)
        return torch.sigmoid(x)

summary.summary(UNet(in_channels=3, out_channels=1),input_size=(3, 512, 512),device='cpu')

class BuildingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.img_list[index])
        mask_path = os.path.join(self.mask_dir, self.img_list[index])

        img_np = np.array(Image.open(image_path).convert('RGB'))
        mask_np = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)  

        mask_np[mask_np == 255.0] = 1.0 

        if self.transforms is not None:
            augumented_pair = self.transforms(image=img_np, mask=mask_np) 
            img = augumented_pair['image']  
            mask = augumented_pair['mask']
        else:
            img = img_np
            mask = mask_np

        return img, mask

dice_list = []
train_loss_list = []
val_acc_list = []

# Hyperparameters and dataset path directories
LEARNING_RATE = 0.00008
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_BATCH_SIZE = 18
VAL_BATCH_SIZE = 4
NUM_EPOCHS = 10
TRAIN_NUM_WORKERS = 0 # If you run in CPU, keep this value as 0. Increase the values only when you use GPU
VAL_NUM_WORKERS = 0 # If you run in CPU, keep this value as 0. Increase the values only when you use GPU
IMAGE_HEIGHT = 512
IMAGE_WIDTH = IMAGE_HEIGHT
VAL_IMAGE_HEIGHT = 1472
VAL_IMAGE_WIDTH = 1472
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train"
TRAIN_MASK_DIR = "data/train_labels"
VAL_IMG_DIR = "data/val"
VAL_MASK_DIR = "data/val_labels"

# for training the model for each epoch.
def train_one_epoch(loader, model, optimizer, loss_fn, scaler, scheduler):
    loop = tqdm(loader)
    cumulative_loss = 0
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE)
        
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            cumulative_loss += loss
            
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    scheduler.step()
    return cumulative_loss.item() / len(loop)

def val_once(loader, model, device='cpu'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device).unsqueeze(1)

            predictions = (model(data))
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == targets).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * targets).sum()) / (
                    (predictions + targets).sum() + 1e-8
            )


    acc = num_correct / num_pixels
    print("Got {}/{} pixels correct with acc {}%, dice {}".format(
        num_correct, num_pixels, acc * 100, dice_score / len(loader)
    ))
    
    model.train()
    return dice_score.item(), acc.item()


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(obj=state, f=filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    

def save_predictions_as_images(loader, model, epoch, device=DEVICE):
    model.eval()
    
    for index, (data, target) in enumerate(loader):
        data = data.to(device)

        with torch.no_grad():
            prediction = (model(data))
            prediction = (prediction > 0.5).float()
        
        
        torchvision.utils.save_image(
            prediction, fp='./pred_epoch {}.data'.format(epoch)
        )

    model.train()

    
def main():
    train_transform = A.Compose([
        A.Rotate(limit=40),
        A.RandomCrop(int(IMAGE_HEIGHT), int(IMAGE_WIDTH)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.CenterCrop(VAL_IMAGE_HEIGHT, VAL_IMAGE_WIDTH),
        A.Normalize(),
        ToTensorV2()
    ])

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode='binary')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    lmbda = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda, verbose=True)

    train_dataset = BuildingDataset(img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transforms=train_transform)
    valid_dataset = BuildingDataset(img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transforms=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True,
                              pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_dataset, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False,
                              pin_memory=PIN_MEMORY)

    scaler = torch.cuda.amp.GradScaler()
    
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print("Epoch {}".format(epoch))
        loss = train_one_epoch(loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler, scheduler=scheduler)
        train_loss_list.append(loss)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        
        # check accuracy
        dice, acc = val_once(loader=valid_loader, model=model, device=DEVICE)
        dice_list.append(dice)
        val_acc_list.append(acc)
        
        if dice_list[-1] == max(dice_list):
            save_checkpoint(checkpoint)
        
        save_predictions_as_images(loader=valid_loader, model=model, device=DEVICE, epoch=epoch)
        
    if LOAD_MODEL:
        load_checkpoint(checkpoint=torch.load(f="my_checkpoint.pth.tar",map_location ='cpu'), model=model)


main()

sns.lineplot(data=dice_list)

sns.lineplot(data=val_acc_list)

sns.lineplot(data=train_loss_list)

plt.legend(labels=["dice_loss","validation_accuracy","training_loss"], title = "Legend", loc = 2)

plt.savefig('Result/plot.png')
          
TEST_IMG_DIR = "data/test"
TEST_MASK_DIR = "data/test_labels"

valid_transform = A.Compose([
            A.CenterCrop(VAL_IMAGE_HEIGHT, VAL_IMAGE_WIDTH),
            A.Normalize(),
            ToTensorV2()
        ])

test_dataset = BuildingDataset(img_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transforms=valid_transform)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=VAL_NUM_WORKERS, shuffle=False, pin_memory=PIN_MEMORY)
    
model = UNet(in_channels=3, out_channels=1).to(DEVICE)

load_checkpoint(checkpoint=torch.load(f="my_checkpoint.pth.tar"), model=model)

mask_list = []
for index, (img, mask) in enumerate(test_loader):    
    img = img.to(DEVICE)
    output = model(img)
    
    output = output.detach().squeeze().cpu().numpy()
    mask_list.append(output)

fig, ax = plt.subplots(10, 3, figsize=(15,50))
fig.tight_layout()

img_names = os.listdir(TEST_IMG_DIR)

for j in range(len(img_names)):
    img = Image.open("data/test/{}".format(img_names[j]))
    img = np.array(img)
    mask = Image.open("data/test_labels/{}".format(img_names[j]))
    mask = np.array(mask)
    ax[j, 0].set_title('Input Image')
    ax[j, 0].imshow(img)
    ax[j, 1].set_title('Predicted Mask')
    ax[j, 1].imshow(mask_list[j]>0.65)
    ax[j, 2].set_title('Created Mask')
    ax[j, 2].imshow(mask)
    plt.savefig('Result/finaloutput.pdf')
