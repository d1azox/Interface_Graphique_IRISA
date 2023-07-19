#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:56:07 2021

@author: courtrai
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



class ResidualBlock2D(nn.Module):
    def __init__(self, channels , k=3, p=1):
        super(ResidualBlock2D, self).__init__()
        self.net = nn.Sequential(
			nn.Conv2d( channels, channels, kernel_size=k, padding=p),
			nn.BatchNorm2d( channels),
			nn.PReLU(),
        	
        	    nn.Conv2d( channels,  channels, kernel_size=k, padding=p),
        	    nn.BatchNorm2d( channels)
        )
    def forward(self, x):
        return x + self.net(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BottleNetResidualBlock2D_OLD(nn.Module):
    def __init__(self, channels  , k=3, p=1):
        super(BottleNetResidualBlock2D, self).__init__()
        place=channels //4
        self.c1=    conv1x1( channels,place   )
        self.b1=	nn.BatchNorm2d( place)
        self.r1=	nn.PReLU()    
        
        self.c2=	conv3x3( place, place )
        self.b2=	nn.BatchNorm2d( place)
        self.r2=	nn.PReLU()
        
        self.c3=    conv1x1( place,  channels )
        self.b3=    nn.BatchNorm2d( channels)
     
    def forward(self, x):
        
        print(x.size())
        out=self.c1(x)
        out=self.b1(out)
        out=self.r1(out)
        out=self.c2(out)
        out=self.b2(out)
        out=self.r2(out)

        out=self.c3(out)
        out=self.b3(out)
#        out=self.net(x)
        print(out.size())
        return x + out
    
class BottleNetResidualBlock2D(nn.Module):
    def __init__(self, channels  , k=3, p=1):
        super(BottleNetResidualBlock2D, self).__init__()
        place=channels //4
        self.net = nn.Sequential(
                
            conv1x1( channels,place   ),
			nn.BatchNorm2d( place),
			nn.PReLU() ,   
                
			conv3x3( place, place ),
			nn.BatchNorm2d( place),
			nn.PReLU(),
        	
    	    conv1x1( place,  channels ),
    	    nn.BatchNorm2d( channels)
    )
    def forward(self, x):
#        print(x.size())
#        out=self.net(x)
#        print(out.size())
        return x + self.net(x)
    
    
    
class RedNet2DV3L_3x3_Residual_2B(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64),init=True):
        super(RedNet2DV3L_3x3_Residual_2B, self).__init__()
        
        
        
        self.dims=dims
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        self._res2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.conv_Res3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self._res1=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
                 
             
                 
                 
        if init :         
            self._initialize_weights()
    
        
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)
    # def _initialize_weights(self):
       
    #     init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.conv_Res3ToUnPool[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self._res2.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self._res1.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self._res2.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self._res1.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
      
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out = self.res1(out)
        x_size=out.size()
       
        out,idx1 = self.maxPool1(out)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        out = self._res2(out)
        out=self.conv_Res3ToUnPool(out)
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        out = self._res1(out)
        out =self.last(out)
    
        return out    



class RedNet2DV3L_3x3_Residual_3B(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
                 
        if init :
            self._initialize_weights()
            
            
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)          
        
    def _initialize_weights_OLF(self):
       
        init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        
        init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        
     
        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out = self.res1(out)
        x_size=out.size()
       
        out,idx1 = self.maxPool1(out)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        x_size1=out.size()
        out,idx2 = self.maxPool2(out)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 



class RedNet2DV3L_3x3_BottleNetResidual_3B(nn.Module ):
 
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=False):
        
        super(RedNet2DV3L_3x3_BottleNetResidual_3B, self).__init__()
        
        self.dims=dims
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  BottleNetResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  BottleNetResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  BottleNetResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  BottleNetResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        
        self.res_2=  BottleNetResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res_1=  BottleNetResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :         
            self._initialize_weights()

    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)        
    def _initialize_weights_OLD(self):
       
        init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[6].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[6].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[6].weight, init.calculate_gain('relu'))
         
        init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_1.net[6].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[6].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[6].weight, init.calculate_gain('relu'))
           
     
        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out = self.res1(out)
        x_size=out.size()
       
        out,idx1 = self.maxPool1(out)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        x_size1=out.size()
        out,idx2 = self.maxPool2(out)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 



class RedNet2DV3L_3x3_Residual_5B(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128,256,512),init=False):
        super(RedNet2DV3L_3x3_Residual_5B, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_5B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        self.maxPool3= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes4 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[3]),
            nn.LeakyReLU(0.2, True))
        
        self.res4=  ResidualBlock2D(dims[3])
         

        self.maxPool4= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes5 = nn.Sequential(
            nn.Conv2d(dims[3], dims[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[4]),
            nn.LeakyReLU(0.2, True))
        
        self.res5=  ResidualBlock2D(dims[4])
         
        
        self.res_5=  ResidualBlock2D(dims[4])
        
        
        self.convRes_5ToUnPool = nn.Sequential(
            nn.Conv2d(dims[4], dims[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[3]),
            nn.LeakyReLU(0.2, True))        
      
        self.res_4=  ResidualBlock2D(dims[3])
        
        
        self.convRes_4ToUnPool = nn.Sequential(
            nn.Conv2d(dims[3], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :         
            self._initialize_weights()
            print("fin init model")


    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)
        
    def _initialize_weights_old(self):
       
        print("_initialize_weights")
        print("_ init  conv MaxToRes")
        init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes4[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes5[0].weight, init.calculate_gain('relu'))
        print("_ init  conv UnPoll")

#        init.orthogonal_(self.convRes_5ToUnPool[0].weight, init.calculate_gain('relu'))
#        init.orthogonal_(self.convRes_4ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
        print("_ init  conv res")
        init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res4.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res4.net[3].weight, init.calculate_gain('relu'))
 
#        init.orthogonal_(self.res5.net[0].weight, init.calculate_gain('relu'))
#        init.orthogonal_(self.res5.net[3].weight, init.calculate_gain('relu'))
        print("_ init conv _res")
        init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_4.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_4.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_5.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_5.net[3].weight, init.calculate_gain('relu'))
        
     
        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out = self.res1(out)
        x_size=out.size()
       
        out,idx1 = self.maxPool1(out)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        x_size1=out.size()
        out,idx2 = self.maxPool2(out)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        x_size2=out.size()
        out,idx3 = self.maxPool3(out)
        
        
        out=self.convMaxToRes4(out)
        out = self.res4(out)    
         
        x_size3=out.size()
        out,idx4 = self.maxPool4(out)
        
        
        out=self.convMaxToRes5(out)
        out = self.res5(out)    
        
        
        out = self.res_5(out)
        out=self.convRes_5ToUnPool(out)
              
        out = F.max_unpool2d(out, idx4, kernel_size=2, stride=2, output_size=x_size3)
         
 
       
        out = self.res_4(out)
        out=self.convRes_4ToUnPool(out)
              
        out = F.max_unpool2d(out, idx3, kernel_size=2, stride=2, output_size=x_size2)
        
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 
    
    
class RedNet2DV3L_3x3_Residual_4B(nn.Module ):
     
     def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128,256),init=False):
         super(RedNet2DV3L_3x3_Residual_4B, self).__init__()
         
         
         
         self.dims=dims
         
         
         print("init  SargassumNet2DV3L_3x3_Residual_5B DIMS ",dims)
    
         self.layer1 = nn.Sequential(
             nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[0]),
             nn.LeakyReLU(0.2, True))
             
         self.res1=  ResidualBlock2D(dims[0])      
         self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
         self.convMaxToRes2 = nn.Sequential(
             nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[1]),
             nn.LeakyReLU(0.2, True))
         self.res2=  ResidualBlock2D(dims[1])
         
         
         self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
         self.convMaxToRes3 = nn.Sequential(
             nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[2]),
             nn.LeakyReLU(0.2, True))
         
         self.res3=  ResidualBlock2D(dims[2])
         
         
         self.maxPool3= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
         self.convMaxToRes4 = nn.Sequential(
             nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[3]),
             nn.LeakyReLU(0.2, True))
         
         self.res4=  ResidualBlock2D(dims[3])
          

             
       
         self.res_4=  ResidualBlock2D(dims[3])
         
         
         self.convRes_4ToUnPool = nn.Sequential(
             nn.Conv2d(dims[3], dims[2], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[2]),
             nn.LeakyReLU(0.2, True))        
         
         
         
         self.res_3=  ResidualBlock2D(dims[2])
         
         
         self.convRes_3ToUnPool = nn.Sequential(
             nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[1]),
             nn.LeakyReLU(0.2, True))        
         
         
         self.res_2=  ResidualBlock2D(dims[1])
         #  unpool2d
         self.convRes_2ToUnPool = nn.Sequential(
             nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
             nn.BatchNorm2d(dims[0]),
             nn.LeakyReLU(0.2, True))
         
         self.res_1=  ResidualBlock2D(dims[0]) 
         self.last=nn.Sequential(
                  nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                            padding=1),
                  nn.BatchNorm2d(num_classes)
                 )
         if init :         
             self._initialize_weights()
             print("fin init model")
     def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)        
     def _initialize_weights_old(self):
        
         print("_initialize_weights")
         print("_ init  conv MaxToRes")
         init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.convMaxToRes4[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.convMaxToRes5[0].weight, init.calculate_gain('relu'))
         print("_ init  conv UnPoll")

 #        init.orthogonal_(self.convRes_5ToUnPool[0].weight, init.calculate_gain('relu'))
 #        init.orthogonal_(self.convRes_4ToUnPool[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
         print("_ init  conv res")
         init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res4.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res4.net[3].weight, init.calculate_gain('relu'))
  
 #        init.orthogonal_(self.res5.net[0].weight, init.calculate_gain('relu'))
 #        init.orthogonal_(self.res5.net[3].weight, init.calculate_gain('relu'))
         print("_ init conv _res")
         init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_4.net[0].weight, init.calculate_gain('relu'))
         init.orthogonal_(self.res_4.net[3].weight, init.calculate_gain('relu'))
       
      
         init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
         
         
     def forward(self, x):
      
         out = self.layer1(x)     
         out = self.res1(out)
         x_size=out.size()
        
         out,idx1 = self.maxPool1(out)
        
         out=self.convMaxToRes2(out)
         out = self.res2(out)
         
         x_size1=out.size()
         out,idx2 = self.maxPool2(out)
         
         
         out=self.convMaxToRes3(out)
         out = self.res3(out)    
         
         
         x_size2=out.size()
         out,idx3 = self.maxPool3(out)
         
         
         out=self.convMaxToRes4(out)
         out = self.res4(out)    
          
         
        
         out = self.res_4(out)
         out=self.convRes_4ToUnPool(out)
               
         out = F.max_unpool2d(out, idx3, kernel_size=2, stride=2, output_size=x_size2)
         
         
         
         out = self.res_3(out)
         out=self.convRes_3ToUnPool(out)
         
         
         out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
         
         
         out = self.res_2(out)
         out=self.convRes_2ToUnPool(out)
         
         
         out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
         
         
         out = self.res_1(out)
         out =self.last(out)
     
         return out 
     
        

class RedNet2DV3L_3x3_Residual_3B_FullResiduel(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_FullResiduel, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        
        self.res_2=  ResidualBlock2D(2*dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(2*dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res_1=  ResidualBlock2D(2*dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(2*dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :
            self._initialize_weights()
 
        
    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)
    def _initialize_weights_OLD(self):
       
        init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        
        init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        
     
      
        
        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out1 = self.res1(out)
        x_size=out1.size()
#        print("out1",out1.size())
        
        out,idx1 = self.maxPool1(out1)
       
        out=self.convMaxToRes2(out)
        out2 = self.res2(out)
        
        x_size1=out2.size()
        
        out,idx2 = self.maxPool2(out2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
         
#        print("out2",out2.size())
#        print("out apres max_unpool2d",out.size())
        out = torch.cat((out, out2), dim=1)
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        out = torch.cat((out, out1), dim=1)
        
#        print("Out apres unpool2d",out.size())
        out = self.res_1(out)
        out =self.last(out)
    
        return out 


class RedNet2DV3L_3x3_ConcatResidual(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_ConcatResidual, self).__init__()
        
        
        
        self.dims=dims
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        self.res3=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res4=  ResidualBlock2D(2*dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(2*dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :
            self._initialize_weights()
 
         
    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)
    def _initialize_weights_old(self):
       
        init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convRes3ToUnPool[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res4.net[0].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.res4.net[3].weight, init.calculate_gain('relu'))
        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out1 = self.res1(out)
        x_size=out1.size()
       
        out,idx1 = self.maxPool1(out1)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        out = self.res3(out)
        out= self.convRes3ToUnPool(out)
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        out = torch.cat((out, out1), dim=1)
        
        out = self.res4(out)
        out =self.last(out)
    
        return out    
    
    
    
    
class RedNet2DV3L_3x3_FullResidual(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_FullResidual, self).__init__()
        
        
        
        self.dims=dims
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        self.res3=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.res4=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :
            self._initialize_weights()
 
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)       
 
    # def _initialize_weights(self):
       
    #     init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.convRes3ToUnPool[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res4.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res4.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out1 = self.res1(out)
        x_size=out1.size()
       
        out,idx1 = self.maxPool1(out1)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
        out = self.res3(out)
        out=self.convRes3ToUnPool(out)
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        out += out1
        out = self.res4(out)
        out =self.last(out)
    
        return out       

 
 
 
    
class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

     
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        
     
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

 
class RedNet2DV3L_3x3_Residual_2B_Att(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64),init=True):
        super(RedNet2DV3L_3x3_Residual_2B_Att, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        
        
       
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        self.att_1 = Attention_block(F_g=dims[0], F_l=dims[0], F_int=1)
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :
            self._initialize_weights()
        
        
    def _initialize_weights(self):
          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.constant_(m.bias, 0)    
                  
                  
    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)                  
                  
    # def _initialize_weightsOLD(self):
       
    #     init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
       
        
    #     init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
         
    #     init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
          
     
    #     init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
         
    #     init.orthogonal(self.att_1.W_g[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal(self.att_1.W_x[0].weight, init.calculate_gain('relu'))
    #     init.orthogonal(self.att_1.psi[0].weight, init.calculate_gain('relu'))
        
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out_r1 = self.res1(out)
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out = self.res2(out)
        
   
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
         
        out = self.att_1(g=out, x=out_r1)
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 
 
 

class RedNet2DV3L_3x3_Residual_3B_Att(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_Att, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        self.att_2 = Attention_block(F_g=dims[1], F_l=dims[1], F_int=dims[0])
        
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        self.att_1 = Attention_block(F_g=dims[0], F_l=dims[0], F_int=1)
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :
            self._initialize_weights()
        
        
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)              
    
        
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out_r1 = self.res1(out)
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
    
        out = self.att_2(g=out, x=out_r2)
       
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        out = self.att_1(g=out, x=out_r1)
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 
 
    
 
    
 
 
 
    
     
class RedNet2DV3L_3x3_Residual_3B_MultiAtt(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_MultiAtt, self).__init__()
        
        
        
        self.dims=dims
        self.num_classes=num_classes
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.nb_firstRes=3
        firstListe=[]
        for b in range(self.nb_firstRes) :
            firstListe.append(ResidualBlock2D(dims[0]))
        self.firstRes = nn.Sequential(*firstListe) 
        
        
        self.convFirstToForAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.convFinalToAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        
        
        self.res3=  ResidualBlock2D(dims[2])
#        
    
#        
        
       
        self.att = Attention_block(F_g=num_classes , F_l=num_classes   , F_int=10) 
#        
#        self.convAttTo_Res3 = nn.Sequential(
#            nn.Conv2d(self.num_classes,dims[0] , kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(dims[2]),
#            nn.LeakyReLU(0.2, True))
#               
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
   
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        
        
       
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        
        
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0] , num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init :
            self._initialize_weights()
        
       
    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)        
        
#     def _initialize_weights(self):
       
#         init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.convFirstToForAtt[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convFinalToAtt[0].weight, init.calculate_gain('relu'))
# #        init.orthogonal_(self.convAttTo_Res3[0].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        
        
        
#         for c in range(self.nb_firstRes):
#             init.orthogonal(self.firstRes[c].net[0].weight, init.calculate_gain('relu'))
#             init.orthogonal(self.firstRes[c].net[3].weight, init.calculate_gain('relu'))
        
     
            
#         init.orthogonal(self.att.W_g[0].weight, init.calculate_gain('relu'))
#         init.orthogonal(self.att.W_x[0].weight, init.calculate_gain('relu'))
#         init.orthogonal(self.att.psi[0].weight, init.calculate_gain('relu'))
            
# #        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
            
        
        
    def forward(self, x):
     
        out = self.layer1(x)   
        
        
        
        outFirst =self.firstRes(out)
        
        for_att=self.convFirstToForAtt(outFirst)
        
        out_r1 = self.res1(outFirst) # debut encoder
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
       
       
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        
        out = self.res_1(out)
        
        finalAtt=self.convFinalToAtt(out)
        
        
        res = self.att(finalAtt,for_att)
        
        
         
        
        outFirst=self.last(outFirst )
        return outFirst,res      
    
 
    
class RedNet2DV3L_3x3_Residual_3B_LS_Att(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_LS_Att, self).__init__()
        
        
        
        self.dims=dims
        self.num_classes=num_classes
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        self.convFinalToAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        
        self.upsample=nn.Upsample(scale_factor=4, mode='nearest')
        self.res3=  ResidualBlock2D(dims[2])
#        
        self.unpool_Att = nn.MaxUnpool2d(4, stride=2)
#        
        
      
        self.convLSToForAtt = nn.Sequential(
            nn.Conv2d(dims[2], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
         
        self.att = Attention_block(F_g=num_classes , F_l=num_classes   , F_int=10) 
#        
#        self.convAttTo_Res3 = nn.Sequential(
#            nn.Conv2d(self.num_classes,dims[0] , kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(dims[2]),
#            nn.LeakyReLU(0.2, True))
#               
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
   
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        
        
       
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        
        
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0] , num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
                 
        self.lastShort4=nn.Sequential(
                 nn.Conv2d(dims[2] , num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )         
        if init :
            self._initialize_weights()
        
       
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)       
        
#     def _initialize_weights(self):
       
#         init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.lastShort4[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        
        
#         init.orthogonal_(self.convLSToForAtt[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convFinalToAtt[0].weight, init.calculate_gain('relu'))
# #        init.orthogonal_(self.convAttTo_Res3[0].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        
        
#        
#        for c in range(self.nb_firstRes):
#            init.orthogonal(self.firstRes[c].net[0].weight, init.calculate_gain('relu'))
#            init.orthogonal(self.firstRes[c].net[3].weight, init.calculate_gain('relu'))
#        
     
            
        # init.orthogonal_(self.att.W_g[0].weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.att.W_x[0].weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.att.psi[0].weight, init.calculate_gain('relu'))
            
#        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
            
        
        
    def forward(self, x,short=False):
     
        out = self.layer1(x)   
        
        
     
        
   
        
        out_r1 = self.res1(out) # debut encoder
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        if  short :
            out_=self.lastShort4(out)
            return out_
       
        for_att=self.convLSToForAtt(out)
          
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
 
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
         
        
        out = self.res_1(out)
        
        finalAtt=self.convFinalToAtt(out)
        
        for_att=self.upsample(for_att)
        res = self.att(finalAtt,for_att)
         
        
        
        return  res      
    
    


     
        
        
         
            
    
class RedNet2DV3L_3x3_Residual_3B_MS_Att(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True,train=True):
        super(RedNet2DV3L_3x3_Residual_3B_MS_Att, self).__init__()
        
        
        
        self.dims=dims
        self.num_classes=num_classes
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        self.convFinalToAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        
        self.upsample=nn.Upsample(scale_factor=2, mode='nearest')
        self.res3=  ResidualBlock2D(dims[2])
#        
        self.unpool_Att = nn.MaxUnpool2d(2, stride=2)
#        
        
      
        self.convLSToForAtt = nn.Sequential(
            nn.Conv2d(dims[1], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
         
        self.att = Attention_block(F_g=num_classes , F_l=num_classes   , F_int=10) 
#        
#        self.convAttTo_Res3 = nn.Sequential(
#            nn.Conv2d(self.num_classes,dims[0] , kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(dims[2]),
#            nn.LeakyReLU(0.2, True))
#               
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
   
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        
        
       
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        
        
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0] , num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
                 
        self.lastShort2=nn.Sequential(
                 nn.Conv2d(dims[1] , num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )  
        self.train_=train    
        if init :
            self._initialize_weights()
        
       
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)       
        
#     def _initialize_weights(self):
       
#         init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.lastShort2[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        
        
#         init.orthogonal_(self.convLSToForAtt[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convFinalToAtt[0].weight, init.calculate_gain('relu'))
# #        init.orthogonal_(self.convAttTo_Res3[0].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        
        
# #        
# #        for c in range(self.nb_firstRes):
# #            init.orthogonal(self.firstRes[c].net[0].weight, init.calculate_gain('relu'))
# #            init.orthogonal(self.firstRes[c].net[3].weight, init.calculate_gain('relu'))
# #        
     
            
#         init.orthogonal_(self.att.W_g[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.att.W_x[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.att.psi[0].weight, init.calculate_gain('relu'))
            
# #        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
            
        
        
    def forward(self, x,short=False):
     
        out = self.layer1(x)   
        
        
     
        
   
        
        out_r1 = self.res1(out) # debut encoder
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
          
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
 
        out = self.res_2(out)
        
        
#        if  short :
        if self.train_ :
            out_1=self.lastShort2(out)
#            return out_1
       
        for_att=self.convLSToForAtt(out)
        
        
        
        
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
         
        
        out = self.res_1(out)
        
        finalAtt=self.convFinalToAtt(out)
        
        for_att=self.upsample(for_att)
        res = self.att(finalAtt,for_att)
         
     
        return (out_1,res) if self.train_ else res
    
       


class RedNet2DV3L_3x3_Residual_2B_LS_Att(nn.Module ):
    
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64),init=True,train=True):
        super(RedNet2DV3L_3x3_Residual_2B_LS_Att, self).__init__()
        
        
        
        self.dims=dims
        self.num_classes=num_classes
        
        print("init  SargassumNet2DV3L_3x3_Residual__2B LS_Att3  ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        self.convFinalToAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
      
        
        
        self.upsample=nn.Upsample(scale_factor=2, mode='nearest')
       
#        
        self.unpool_Att = nn.MaxUnpool2d(2, stride=2)
#        
        
      
        self.convLSToForAtt = nn.Sequential(
            nn.Conv2d(dims[1], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
         
        self.att = Attention_block(F_g=num_classes , F_l=num_classes   , F_int=10) 
#        
#        self.convAttTo_Res3 = nn.Sequential(
#            nn.Conv2d(self.num_classes,dims[0] , kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(dims[2]),
#            nn.LeakyReLU(0.2, True))
#               
        
        
     
   
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        
        
       
        
        self.res_1=  ResidualBlock2D(dims[0]) 
        
        
        self.last=nn.Sequential(
                 nn.Conv2d(dims[0] , num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
                 
        self.lastShort2=nn.Sequential(
                 nn.Conv2d(dims[1] , num_classes+1, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes+1)
                )         
        if init :
            self._initialize_weights()
        
       
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    
                
                
                
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)               
#     def _initialize_weightsOLD(self):
       
#         init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.lastShort2[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
      
#         init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
       
        
#         init.orthogonal_(self.convLSToForAtt[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convFinalToAtt[0].weight, init.calculate_gain('relu'))
# #        init.orthogonal_(self.convAttTo_Res3[0].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
        
        
# #        
# #        for c in range(self.nb_firstRes):
# #            init.orthogonal(self.firstRes[c].net[0].weight, init.calculate_gain('relu'))
# #            init.orthogonal(self.firstRes[c].net[3].weight, init.calculate_gain('relu'))
# #        
     
            
#         init.orthogonal_(self.att.W_g[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.att.W_x[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.att.psi[0].weight, init.calculate_gain('relu'))
            
# #        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
            
        
        
    def forward(self, x,short=False):
     
        out = self.layer1(x)   
        
        
     
        
   
        
        out_r1 = self.res1(out) # debut encoder
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out  = self.res2(out)
        
        
        
        if  short :
            out_=self.lastShort2(out)
            return out_
       
        for_att=self.convLSToForAtt(out)
          
      
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
         
        
        out = self.res_1(out)
        
        finalAtt=self.convFinalToAtt(out)
        
        for_att=self.upsample(for_att)
        res = self.att(finalAtt,for_att)
         
        
        
        return  res      
    
    
        
    
    
    
    
    
class RedNet2DV3L_3x3_Residual_3B_MultiAttOLD(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_MultiAtt, self).__init__()
        
        
        
        self.dims=dims
        self.num_classes=num_classes
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        self.nb_firstRes=3
        firstListe=[]
        for b in range(self.nb_firstRes) :
            firstListe.append(ResidualBlock2D(dims[0]))
        self.firstRes = nn.Sequential(*firstListe) 
        
        
        self.convFirstToForAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.convFinalToAtt = nn.Sequential(
            nn.Conv2d(dims[0], self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_classes),
            nn.LeakyReLU(0.2, True))
        
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        
        
        self.res3=  ResidualBlock2D(dims[2])
#        
    
#        
        
        attrListe=[]
        for a in range(self.num_classes ):
            attrListe.append(Attention_block(F_g=1, F_l=1 , F_int=1))
        self.atts = nn.Sequential(*attrListe)
#        
#        self.convAttTo_Res3 = nn.Sequential(
#            nn.Conv2d(self.num_classes,dims[0] , kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(dims[2]),
#            nn.LeakyReLU(0.2, True))
#               
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
   
        
        self.res_2=  ResidualBlock2D(dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        
        
        
       
        
        self.res_1=  ResidualBlock2D(dims[0]) 
#        self.last=nn.Sequential(
#                 nn.Conv2d(dims[0]+num_classes, num_classes, kernel_size=3,
#                                           padding=1),
#                 nn.BatchNorm2d(num_classes)
#                )
        
        if init :
            self._initialize_weights()
        
            print("Fin init")

    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0, 0.01)
              nn.init.constant_(m.bias, 0)        
        
#     def _initialize_weights(self):
       
#         init.orthogonal_(self.layer1[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes2[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convMaxToRes3[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_3ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convRes_2ToUnPool[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res3.net[3].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.convFirstToForAtt[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.convFinalToAtt[0].weight, init.calculate_gain('relu'))
# #        init.orthogonal_(self.convAttTo_Res3[0].weight, init.calculate_gain('relu'))
        
#         init.orthogonal_(self.res_1.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_1.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_2.net[3].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[0].weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.res_3.net[3].weight, init.calculate_gain('relu'))
        
     
# #        init.orthogonal_(self.last[0].weight, init.calculate_gain('relu'))
        
        
#         for c in range(self.nb_firstRes):
#             init.orthogonal(self.firstRes[c].net[0].weight, init.calculate_gain('relu'))
#             init.orthogonal(self.firstRes[c].net[3].weight, init.calculate_gain('relu'))
        
#         for c in range(self.num_classes):
            
#             init.orthogonal(self.atts[c].W_g[0].weight, init.calculate_gain('relu'))
#             init.orthogonal(self.atts[c].W_x[0].weight, init.calculate_gain('relu'))
#             init.orthogonal(self.atts[c].psi[0].weight, init.calculate_gain('relu'))
            
             
        
        
    def forward(self, x):
     
        out = self.layer1(x)   
        
        
        
        out =self.firstRes(out)
        
        for_att=self.convFirstToForAtt(out)
        
        out_r1 = self.res1(out) # debut encoder
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
       
       
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        
        out = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        
        out = self.res_1(out)
        
        finalAtt=self.convFinalToAtt(out)
         
#        print("finalAtt",finalAtt.size(),"for_att",for_att.size())
        for c in range(self.num_classes) :
            forA=for_att[:,c,:,:] 
            forA=forA.unsqueeze( 1)
            finalA =finalAtt[:,c,:,:]
            finalA=finalA.unsqueeze( 1)
#            print("g=finalA",finalA.size(),"x=attfora",forA.size())
            if  c==0 :
               res = self.atts[c](g=finalA, x=forA)
            else :
               res=torch.cat((res,self.atts[c](g=finalA, x=forA)),1)
          
#        print(res.size())    
#        out =self.last(res)
    
        return res      
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
class RedNet2DV3L_3x3_Residual_3B_Att_C(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_Att_C, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B_Attr-C DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        self.att_2 = Attention_block(F_g=dims[1], F_l=dims[1], F_int=dims[0])
        
        
        self.res_2=  ResidualBlock2D(2*dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(2*dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        self.att_1 = Attention_block(F_g=dims[1]//2, F_l=dims[1]//2, F_int=dims[0])
        
        self.res_1=  ResidualBlock2D(2*dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(2*dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init:
            self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)             
     
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out_r1 = self.res1(out)
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out_up2 = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
    
        out = self.att_2(g=out_up2, x=out_r2)
        
        out=torch.cat((out_up2,out),1)
        
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        out_up1 = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        out = self.att_1(g=out_up1, x=out_r1)
        
        out=torch.cat((out_up1,out),1)
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 
 
    
    
    
class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = nn.BatchNorm2d 

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", BATCH_NORM(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())   
 
_BOTTLENECK_EXPANSION = 4
    
class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)
 



class FocalModulationL_Q_KV_V2(nn.Module):
     """ Focal Modulation

     Args:
         dim (int): Number of input channels.
         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
         focal_level (int): Number of focal levels
         focal_window (int): Focal window size at focal level 1
         focal_factor (int, default=2): Step to increase the focal window
         use_postln (bool, default=False): Whether use post-modulation layernorm
     """

     def __init__(self, dim, dim_q=2,proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False):

         super().__init__()
         
         
         print(f"init   FocalModulation L_Q_KV_V2  dim={dim} dim_q{dim_q}  focal_level{focal_level} focal_window{focal_window}")
         self.dim = dim

         # specific args for focalv3
         self.focal_level = focal_level
         self.focal_window = focal_window
         self.focal_factor = focal_factor
         self.use_postln = use_postln


         print("dim",dim,"focal_level",self.focal_level,"2*dim+(self.focal_level+1))",str((dim*dim+(self.focal_level+1))))
         # print("dim"+dim+"focal_level"+self.focal_level )
     
         self.convX = nn.Conv2d(dim, dim+(self.focal_level+1) , kernel_size=1, stride=1, padding=0, groups=1, bias=True)
         self.convQ = nn.Conv2d(dim_q, dim_q, kernel_size=1, stride=1, padding=0, groups=1, bias=True) 
         
         self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
         self.toMUL = nn.Conv2d(dim_q, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
         self.act = nn.GELU()
         self.proj = nn.Linear(dim, dim)
         self.proj_drop = nn.Dropout(proj_drop)
         self.focal_layers = nn.ModuleList()

         if self.use_postln:
             self.ln = nn.LayerNorm(dim)

         for k in range(self.focal_level):
             kernel_size = self.focal_factor*k + self.focal_window
             self.focal_layers.append(
                 nn.Sequential(
                     nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                         padding=kernel_size//2, bias=False),
                     nn.GELU(),
                     )
                 )
         #print(self)
         
         
     def forward(self, q,x):
         """ Forward function.

         Args:
             x: input features with shape of (B, H, W, C)
         """
         
         B, nH, nW, C = x.shape
     
         x = x.permute(0, 3, 1, 2).contiguous()
       
         # print('B, nH, nW, C',B, nH, nW, C)
         x = self.convX(x)
        
         q= q.permute(0, 3, 1, 2).contiguous()
         q = self.convQ(q)
        
          
         ctx, gates = torch.split(x, (C, self.focal_level+1), 1)
         
         # print("Q",q.size())
         
         # print("Qctx",ctx.size())
         # print("gatess",gates.size())
         ctx_all = 0
         for l in range(self.focal_level):                     
             ctx = self.focal_layers[l](ctx)
             ctx_all = ctx_all + ctx*gates[:, l:l+1]
         ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
         ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]
         h=self.h(ctx_all)
         q_=self.toMUL(q)
         x_out = q_ * h
         x_out = x_out.permute(0, 2, 3, 1).contiguous()
         if self.use_postln:
             x_out = self.ln(x_out)            
         x_out = self.proj(x_out)
         x_out = self.proj_drop(x_out)
         return x_out


     def forward_debug(self, q,x):
         """ Forward function.

         Args:
             x: input features with shape of (B, H, W, C)
         """
         print("q",q.size(), "x",x.size())
         B, nH, nW, C = x.shape
         print("f(x)",x.size())
         print('B, nH, nW, C',B, nH, nW, C)
         x = self.f(x)
         x = x.permute(0, 3, 1, 2).contiguous()
         q = self.fc_q(q)
         q= q.permute(0, 3, 1, 2).contiguous()
         
         ctx, gates = torch.split(x, (C, self.focal_level+1), 1)
         
         print("Q",q.size())
         
         print("ctx",ctx.size())
         print("gatess",gates.size())
         ctx_all = 0
         for l in range(self.focal_level):                     
             ctx = self.focal_layers[l](ctx)
             ctx_all = ctx_all + ctx*gates[:, l:l+1]
         ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
         ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]
         h=self.h(ctx_all)
         print("q",q.size(),"h",h.size())
     
         q_=self.toMUL(q)
         print("q_",q_.size(),"h ",h.size())
         x_out = q_ * h
         
         x_out = x_out.permute(0, 2, 3, 1).contiguous()
         if self.use_postln:
             x_out = self.ln(x_out)            
         x_out = self.proj(x_out)
         x_out = self.proj_drop(x_out)
         
         print("x_out",x_out.size())
          
         return x_out



   
    
class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)



    
    
class RedNet2DV3L_3x3_Residual_3B_FM_C(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_FM_C, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B_FM-C DIMS ",dims)
   
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        # self.att_2 = Attention_block(F_g=dims[1], F_l=dims[1], F_int=dims[0])
        self.att_2 =  FocalModulationL_Q_KV_V2(dim=dims[1] ,dim_q=dims[1],focal_level=8)
      
        
        self.res_2=  ResidualBlock2D(2*dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(2*dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        # self.att_1 = Attention_block(F_g=dims[1]//2, F_l=dims[1]//2, F_int=dims[0])
        self.att_1 = FocalModulationL_Q_KV_V2(dim=dims[0] ,dim_q=dims[0],focal_level=8)
       
        self.res_1=  ResidualBlock2D(2*dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(2*dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init:
            self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)             
     
        
        
    def forward(self, x):
     
        out = self.layer1(x)     
        out_r1 = self.res1(out)
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out_up2 = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
       
        # out = self.att_2(g=out_up2, x=out_r2)
        out_up2_231 = out_up2.permute(0, 2, 3, 1)
        out_r2_231  =out_r2.permute(0, 2, 3, 1)
        
        # print("SIZE",out_up2.size(),out_r2.size())
        # value1 =self.focalModulation(out_up2,out_r2)     
        # # # print("vakue1",value1.size())
        # # value1=value1.permute(0, 3,1,2) 
        out = self.att_2(q=out_up2_231, x=out_r2_231 )     
        out=out.permute(0, 3,1,2) 
         
             
        out=torch.cat((out_up2,out),1)
        
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        out_up1 = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        out_up1_231 = out_up1.permute(0, 2, 3, 1)
        out_r1_231  =out_r1.permute(0, 2, 3, 1)
        
        # print("SIZE",out_up1.size(),out_r1.size())
        # value1 =self.focalModulation(out_up2,out_r2)     
        # # # print("vakue1",value1.size())
        # # value1=value1.permute(0, 3,1,2) 
        out = self.att_1(q=out_up1_231, x=out_r1_231 )     
        out=out.permute(0, 3,1,2) 
        
        # out = self.att_1(g=out_up1, x=out_r1)
        
        out=torch.cat((out_up1,out),1)
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 
 
class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False):

        super().__init__()
        
        
        print(f"init   FocalModulation  dim={dim} focal_level{focal_level} focal_window{focal_window}")
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln


       
        self.f = nn.Linear(dim, 2*dim+(self.focal_level+1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                        padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )
        #print(self)
        
        
    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
      
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # print("x",x.size(),"B, nH, nW, C",B, nH, nW, C)
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):                     
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)            
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out
    
    
    def forward_debug(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
      
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        print("x",x.size(),"B, nH, nW, C",B, nH, nW, C)
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):                     
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)            
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out    
    
class RedNet2DV3L_3x3_Residual_3B_FullFM_C(nn.Module ):
    def __init__(self , input_channels=12,num_classes=2,dims=(32,64,128),init=True):
        super(RedNet2DV3L_3x3_Residual_3B_FullFM_C, self).__init__()
        
        
        
        self.dims=dims
        
        
        print("init  SargassumNet2DV3L_3x3_Residual_3B_FM-C DIMS ",dims)
        self.fm =  FocalModulation (dim=input_channels , focal_level=7)
 
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
            
        self.res1=  ResidualBlock2D(dims[0])      
        self.maxPool1= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))
        self.res2=  ResidualBlock2D(dims[1])
        
        
        self.maxPool2= nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.convMaxToRes3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, True))
        
        self.res3=  ResidualBlock2D(dims[2])
        
        
        
        
        
        
        self.res_3=  ResidualBlock2D(dims[2])
        
        
        self.convRes_3ToUnPool = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.LeakyReLU(0.2, True))        
        
        # self.att_2 = Attention_block(F_g=dims[1], F_l=dims[1], F_int=dims[0])
        self.att_2 =  FocalModulationL_Q_KV_V2(dim=dims[1] ,dim_q=dims[1],focal_level=8)
      
        
        self.res_2=  ResidualBlock2D(2*dims[1])
        #  unpool2d
        self.convRes_2ToUnPool = nn.Sequential(
            nn.Conv2d(2*dims[1], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.LeakyReLU(0.2, True))
        
        
        # self.att_1 = Attention_block(F_g=dims[1]//2, F_l=dims[1]//2, F_int=dims[0])
        self.att_1 = FocalModulationL_Q_KV_V2(dim=dims[0] ,dim_q=dims[0],focal_level=8)
       
        self.res_1=  ResidualBlock2D(2*dims[0]) 
        self.last=nn.Sequential(
                 nn.Conv2d(2*dims[0], num_classes, kernel_size=3,
                                           padding=1),
                 nn.BatchNorm2d(num_classes)
                )
        if init:
            self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)             
     
        
        
    def forward(self, x):
 
        
        x = x.permute(0, 2, 3, 1)   
        out = self.fm(x )     
        out=out.permute(0, 3,1,2) 
    
       
        out = self.layer1(out)     
        out_r1 = self.res1(out)
        x_size=out_r1.size()
       
        out,idx1 = self.maxPool1(out_r1)
       
        out=self.convMaxToRes2(out)
        out_r2 = self.res2(out)
        
        x_size1=out_r2.size()
        out,idx2 = self.maxPool2(out_r2)
        
        
        out=self.convMaxToRes3(out)
        out = self.res3(out)    
        
        
        out = self.res_3(out)
        out=self.convRes_3ToUnPool(out)
        
        
        out_up2 = F.max_unpool2d(out, idx2, kernel_size=2, stride=2, output_size=x_size1)
        
       
        # out = self.att_2(g=out_up2, x=out_r2)
        out_up2_231 = out_up2.permute(0, 2, 3, 1)
        out_r2_231  =out_r2.permute(0, 2, 3, 1)
        
        # print("SIZE",out_up2.size(),out_r2.size())
        # value1 =self.focalModulation(out_up2,out_r2)     
        # # # print("vakue1",value1.size())
        # # value1=value1.permute(0, 3,1,2) 
        out = self.att_2(q=out_up2_231, x=out_r2_231 )     
        out=out.permute(0, 3,1,2) 
         
             
        out=torch.cat((out_up2,out),1)
        
        
        out = self.res_2(out)
        out=self.convRes_2ToUnPool(out)
        
        out_up1 = F.max_unpool2d(out, idx1, kernel_size=2, stride=2, output_size=x_size)
        
        
        out_up1_231 = out_up1.permute(0, 2, 3, 1)
        out_r1_231  =out_r1.permute(0, 2, 3, 1)
        
        # print("SIZE",out_up1.size(),out_r1.size())
        # value1 =self.focalModulation(out_up2,out_r2)     
        # # # print("vakue1",value1.size())
        # # value1=value1.permute(0, 3,1,2) 
        out = self.att_1(q=out_up1_231, x=out_r1_231 )     
        out=out.permute(0, 3,1,2) 
        
        # out = self.att_1(g=out_up1, x=out_r1)
        
        out=torch.cat((out_up1,out),1)
        
        out = self.res_1(out)
        out =self.last(out)
    
        return out 
            