import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Cloud_extractor(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(Cloud_extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))  
        x1 = self.pool(x1)          
        x2 = F.relu(self.conv2(x1)) 
        x2 = self.upconv1(x2)       
        x2 = torch.cat([x2, x1], dim=1)   
        x2 = F.relu(self.conv3(x2))    
        x_out = self.upconv2(x2)      
        x_out = self.conv4(x_out)     
        return torch.sigmoid(x_out)   

class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        neg_x = -x
        pooled_neg_x = self.max_pool(neg_x)
        return -pooled_neg_x

def compute_median(tensor):
    values = tensor.view(tensor.shape[0], -1)
    median_values = []
    for batch in range(values.shape[0]):
        median_values.append(torch.median(values[batch]))
    median_tensor = torch.tensor(median_values).view(-1, 1, 1, 1)   
    return median_tensor

class Mist_Extractor(nn.Module):
    def __init__(self, window_size=15,sample_num=1000):
        super(Mist_Extractor, self).__init__()
        self.minpool = MinPool2d(window_size)
        self.pool = nn.MaxPool2d(4,4)
        self.sample_num = sample_num

    def forward(self, image):
        assert image.shape[1] == 3
        dark_channel = self._compute_dark_channel(image)
        mist = self._estimate_mist(dark_channel)
        return mist,dark_channel


    def _compute_dark_channel(self, image):
        min_channels, _ = torch.min(image, dim=1, keepdim=True) 
        dark_channel = self.minpool(min_channels)
        return dark_channel

    def _estimate_mist(self, dark_channel):
        mean_value = dark_channel.mean(dim=(2, 3), keepdim=True)
        median_value = self._compute_median(dark_channel)
        median_value = median_value.to(mean_value.device)
        threshold = torch.min(mean_value, median_value)
        flattened_dark_channel = dark_channel.view(dark_channel.shape[0], -1)   
        R = []
        for batch in range(dark_channel.shape[0]):
            flat_dc = flattened_dark_channel[batch]
            probs = torch.ones_like(flat_dc) / len(flat_dc)   
            dist = Categorical(probs)
            sampled_indices = dist.sample((self.sample_num,))
            sampled_values = flat_dc[sampled_indices]
            valid_mask = sampled_values <= threshold[batch, 0, 0, 0]

            if valid_mask.any():
                valid_values = sampled_values[valid_mask]
                R.extend(valid_values.tolist())

         
        if len(R) > 0:
            R_tensor = torch.tensor(R, dtype=torch.long).unsqueeze(0)   
        else:
            R_tensor = torch.zeros((1, 1), dtype=torch.float32)
        R_tensor_float = R_tensor.float()
        mist = torch.mean(R_tensor_float, dim=1)
        mist=mist.half()
        return mist

    def _compute_median(self, dark_channel):
         
        flattened = dark_channel.view(dark_channel.shape[0], -1)
        median_values = []
        for batch in range(flattened.shape[0]):
            median_values.append(torch.median(flattened[batch]))
        median_tensor = torch.tensor(median_values).view(-1, 1, 1, 1)   
        return median_tensor






class CDBS(nn.Module):
    def __init__(self,c1=None, kernel_size=7):
        super().__init__()
        self.cloud= Cloud_extractor(in_channels=3, out_channels=3)
        self.light = Mist_Extractor(window_size=15)

    def forward(self, x): 
        mist_all, dark_channel=self.light(x)  
        mist =  mist_all.unsqueeze(-1).unsqueeze(-1)
        mist = mist.to((x.device))
        cloud = self.cloud(x)

        J = (x-mist)/cloud+mist
        return J


