import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=1):
        super(Conv2d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.padding = padding
        self.k_size = k_size
        self.kernel = self._get_conv_filter(out_ch, in_ch, k_size)
        self.eps = 1e-4
    
    def _get_conv_filter(self, out_ch, in_ch, k_size):
        kernel = nn.Parameter(torch.Tensor(out_ch, in_ch, k_size, k_size))
        
        # weight intialization
        kernel = nn.init.kaiming_normal_(kernel)
        return kernel
    
    def _get_filter_norm(self, kernel):
        return torch.norm(kernel.data, 2, 1, True)
    
    def _get_input_norm(self, feat):
        f = torch.ones(1, self.in_ch, self.k_size, self.k_size)
        #print(feat, feat*feat)
        input_norm = torch.sqrt(F.conv2d(feat*feat, f, stride=self.stride, padding=self.padding)+self.eps)
        return input_norm

    def forward(self, x):
        print('x : ', x)
        print('x shape: ', x.shape)
        x_norm = self._get_input_norm(x)
        print('x_norm : ', x_norm)
        print('x_norm shape: ', x_norm.shape)
        x = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)
        w_norm = self._get_filter_norm(self.kernel)
        print('w_norm : ', w_norm)
        print('w_norm shape: ', w_norm.shape) 
        return x


class DCNet(nn.Module):
    def __init__(self):
        super(DCNet, self).__init__()
        self.conv1 = Conv2d(3, 3, 3) 

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

    x = torch.randn([1,3,3,3])
    y = net(x)



