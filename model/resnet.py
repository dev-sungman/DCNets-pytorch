import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=1):
        super(Conv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel = self._get_conv_filter(out_ch, in_ch, k_size)
    
    def _get_conv_filter(self, out_ch, in_ch, k_size):
        kernel = nn.Parameter(torch.Tensor(out_ch, in_ch, k_size, k_size))
        
        # weight intialization
        kernel = nn.init.kaiming_normal_(kernel)
        return kernel
    
    def _get_filter_norm(self, kernel):
        eps = 1e-4
        return torch.norm(kernel.data, 2, 1, True)

    def forward(self, x):
        w_norm = self._get_filter_norm(self.kernel)
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)
        return x


class DCNet(nn.Module):
    def __init__(self):
        super(DCNet, self).__init__()
        self.conv1 = Conv2d(3, 64, 3) 

    '''
    def _get_input_norm(self, feat, kszie, stride, pad):
        eps = 1e-4
        shape = [ksize, ksize, feat.get_shape()[3], 1]
        f = torch.ones(shape)
        input_norm = torch.sqrt(nn.Conv2d(feat*feat, f, stride=stride, padding=pad)+eps)
        return input_norm
    
    def _get_filter_norm(self, filt):
        eps = 1e-4
        return torch.norm(filt.weight.data, 2, 1, True)
    

    def _get_conv_filter(self, shape):
        filt = nn.Parameter(torch.Tensor(shape))
        print('filter shape: ', filt.shape)


    def _conv2d(self, feat, k_size, n_out, is_training, stride=1, pad=1, orth=False):
        n_in = feat.shape[1]
        print('n_in: ', n_in)
        filt = self._get_conv_filter([n_out, n_in, k_size, k_size])
    '''
        
    
    def forward(self, x):
        x = self.conv1(x) 

        return x


        
            
if __name__ == '__main__':
    net = DCNet()

    x = torch.randn([1,3,112,112])
    y = net(x)



