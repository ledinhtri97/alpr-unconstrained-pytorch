import torch
import torch.nn as nn
import torch.nn.functional as F

    
class OCR(nn.Module):
    def __init__(self):
        super(OCR, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU()
        )
        
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU()
        )
        
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )
        
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU()
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU()
        )
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU()
        )
        
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU()
        )
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU()
        )
        
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=80, kernel_size=1, padding=0, stride=1, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        
        return x
    
if __name__ == '__main__':
    ocr_net = OCR()
    x = torch.rand(1, 3, 80, 240)
    y = ocr_net(x)
    print(y.size())