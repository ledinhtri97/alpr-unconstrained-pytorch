import torch
import torch.nn as nn
import torch.nn.functional as F

class WPOD(nn.Module):
    def __init__(self):
        super(WPOD, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=16, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=16, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=32, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=32, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=32, eps=1e-3, momentum=0.99)
        )
        
        # add block5-bn and block3 and pass to relu
        
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99)
        )
        
        # add block8-bn and block6 and pass to relu
        
        self.block9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99)
        )
        
        # add block10-bn and block8 and pass to relu
        
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.block11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99)
        )
        
        # add block13-bn and block11 and pass to relu
        
        self.block14 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.99)
        )
        
        # add block15-bn and block13 and pass to relu
        
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.block16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block17 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block18 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99)
        )
        
        # add block18-bn and block16 and pass to relu
        
        self.block19 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block20 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99)
        )
        
        # add block20-bn and block18 and pass to relu
        
        self.block21 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block22 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99)
        )
        
        # add block22-bn and block18 and pass to relu
        
        self.block23 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )
        
        self.block24 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99)
        )
        
        # add block24-bn and block22 and pass to relu
        self.softmax = nn.Softmax(1)
        self.conf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1, stride=1)
        
        self.loc = nn.Conv2d(in_channels=128, out_channels=6, kernel_size=3, padding=1, stride=1)
        
        # concate conf and loc
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.pool1(x)
        
        x3 = self.block3(x)
        x = self.block4(x3)
        x = self.block5(x)
        # add block5-bn and block3 and pass to relu
        x = F.relu(x + x3)
        
        x = self.pool2(x)
        
        x6 = self.block6(x)
        x = self.block7(x6)
        x = self.block8(x)
        # add block8-bn and block6 and pass to relu
        x8 = F.relu(x + x6)
        
        x = self.block9(x8)
        x = self.block10(x)
        # add block10-bn and block8 and pass to relu
        x = F.relu(x + x8)
        
        x = self.pool3(x)
        
        x11 = self.block11(x)
        x = self.block12(x11)
        x = self.block13(x)
        # add block13-bn and block11 and pass to relu
        x13 = F.relu(x + x11)
        
        x = self.block14(x13)
        x = self.block15(x)
        # add block15-bn and block13 and pass to relu
        x = F.relu(x + x13)
        
        x = self.pool4(x)
        
        x16 = self.block16(x)
        x = self.block17(x16)
        x = self.block18(x)
        # add block18-bn and block16 and pass to relu
        x18 = F.relu(x + x16)
        
        x = self.block19(x18)
        x = self.block20(x)
        # add block20-bn and block18 and pass to relu
        x20 = F.relu(x + x18)
        
        x = self.block21(x20)
        x = self.block22(x)
        # add block22-bn and block20 and pass to relu
        x22 = F.relu(x + x20)
        
        x = self.block23(x22)
        x = self.block24(x)
        # add block24-bn and block22 and pass to relu
        x = F.relu(x + x22)
        
        conf = self.softmax(self.conf(x))
        loc  = self.loc(x)
        
        print(loc.size())
        
        return conf, loc
        
        
if __name__ == '__main__':
    lp_net = WPOD()
    x = torch.rand(1, 3, 208, 208)
    conf, loc = lp_net(x)
    
    print(conf.size(), loc.size())