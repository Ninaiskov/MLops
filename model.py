from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1), 
                                   nn.ReLU()) # --> torch.Size([1, 3, 24, 24])
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1), 
                                   nn.ReLU()) # --> torch.Size([1, 6, 20, 20])
        #self.conv3 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1), 
        #                           nn.ReLU()) # --> torch.Size([1, 12, 16, 16])
        self.fc1 = nn.Linear(6*20*20, 10) # --> torch.Size([1, 10])
        
    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        #x = self.conv3(x)
        #print(x.shape)
        x = x.view(x.shape[0], -1) # --> torch.Size([1, 3072])
        #print(x.shape)
        x = self.fc1(x)
        return x