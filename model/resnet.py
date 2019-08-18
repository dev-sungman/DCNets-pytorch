import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=1, magnitude=None, angular=None):
        super(Conv2d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.padding = padding
        self.k_size = k_size
        self.kernel = self._get_conv_filter(out_ch, in_ch, k_size)
        self.eps = 1e-4

        self.magnitude = magnitude
        self.angular = angular
    
    def _get_conv_filter(self, out_ch, in_ch, k_size):
        kernel = nn.Parameter(torch.Tensor(out_ch, in_ch, k_size, k_size))
        
        # weight intialization
        kernel = nn.init.kaiming_normal_(kernel)
        return kernel
    
    def _get_filter_norm(self, kernel):
        return torch.norm(kernel.data, 2, 1, True)
    
    def _get_input_norm(self, feat):
        f = torch.ones(1, self.in_ch, self.k_size, self.k_size)
        input_norm = torch.sqrt(F.conv2d(feat*feat, f, stride=self.stride, padding=self.padding)+self.eps)
        return input_norm
    
    #TODO: add orthogonal constraint
    '''
    def _add_orthogonal_constraint(self):
    '''

    def forward(self, x):
        
        if self.magnitude is None:
            print('magnitude function: default')
            out = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)

        elif self.magnitude is "ball":
            print('magnitude function: ball convolution')
            
            x_norm = self._get_input_norm(x) 
            w_norm = self._get_filter_norm(self.kernel)
            
            #self.kernel = torch.div(self.kernel, w_norm)
            kernel_tensor = nn.utils.parameters_to_vector(self.kernel)
            kernel_tensor = torch.reshape(kernel_tensor,self.kernel.shape)
            kernel_tensor = kernel_tensor / w_norm
            out = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)

            radius = nn.Parameter(torch.Tensor(1, 1, 1, out.shape[0]))
            radius = nn.init.constant_(radius, 1.0) ** 2 + self.eps
        
        return out


class DCNet(nn.Module):
    def __init__(self):
        super(DCNet, self).__init__()
        self.conv1 = Conv2d(in_ch=3, out_ch=3, k_size=3) 
        #self.conv1 = Conv2d(in_ch=3, out_ch=3, k_size=3, magnitude="ball") 
        self.bn1 = nn.BatchNorm2d(3)
        
    
    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)
        
        print(x)

        return x


        
            
if __name__ == '__main__':
    net = DCNet()

    x = torch.randn([1,3,3,3])
    y = net(x)


