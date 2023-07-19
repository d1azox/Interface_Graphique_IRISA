"""Test for SegNet 



python3 applySegnetMultiband.py --input /data/Sargassum/SargassumHP/Sargassum/Images12B/S2B_20180929_12b.tif --output result20180929_12b_stride_PlusBGAug_ALL.tif  --weights segnet_Sargassum_valid_best.pth --cuda


python ~/soft/Sargassum2D_2021/applyStrideLucModelResidual12B2CL.py --weights ~/soft/Sargassum2D_2021/Sargassum64_12B_NEW_75000_V2_valid_best_best_v2.pth --cuda --input S2B_MSIL2A_20200129T143659_N0213_R096_T20PRV_20200129T165410.tif --output S2B_MSIL2A_20200129T143659_N0213_R096_T20PRV_20200129T165410_SAR.tif  

python3 applyModelResidual12B2CL.py --input /data/Sargassum/SargassumHP/Sargassum/Sargassum32_12B/Images/20180919_125.tif --output /tmp/result20180919_125.tif  --weights segnet_Sargassum_valid_best12B.pth --cuda
  
  
python3 predictImageSargasumModels3D12B.py  --img /data/Sargassum/ImagesCrop100x100/S2B_20180919_crop_800x1150_100x100_12b.tif  --output S2B_20180919_crop_800x1150_100x100_predict.png --weights  weights/model_lc_best5000_3D_12B.pth
python3 predictImageSargasumModels3D12B.py  --img S2B_20180919_crop_800x1150_100x100_12b.tif  --output S2B_20180919_crop_800x1150_100x100_predict.png --weights  weights/model_lc_best5000_3D_12B.pth
 
python3 predictImageSargasumModels3D12B.py  --img ../Sargassum/Images12B/S2B_20180919_12b.tif --output S2B_20180919_12bresult.tif  --weights  weights/model_lc_best5000_3D_12B.pth --cuda

python3 predictImageSargasumModels5x5_3D12B.py  --img ../Sargassum/Images12B/S2B_20180929_12b.tif --output S2B_20180929_5x5_12bresult.tif  --weights  weights/model_lc_best_5x5_3D_12B.pth --cuda


"""

from __future__ import print_function
 
 
import numpy as np
import torch
import argparse
import cv2
 
#from sargassum_model import  SargassumNet2DV3L_3x3_Residual as SargassumNet

from segnet import  SegNet as SargassumNet

from PIL import Image
import time 
import sys
#
 

parser = argparse.ArgumentParser(description='Predict Image Sargassum')

 
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--weights', default="Sargassum64_12B_NEW_75000_V2_valid_best_best_v2.pth",required=True)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--size', default=256, type=int,help='size of stride')
parser.add_argument('--model', default="64", type=str,help='model Sargaum  32 |64')
#parser.add_argument('--dims', type=str, default="(32,64)" )
parser.add_argument('--nor', action='store_true', help='use normalize')
 
                   
args = parser.parse_args()

 
# print("CUDA",args.cuda)

device="cpu"
if args.cuda :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
 
# print(device)
 
# RGB input
input_channels = args.channels
#  
output_channels = 2

# print("Create modele")
 
model = SargassumNet(input_channels,output_channels ) 
  
# print('Model nb parameters:', sum(param.numel() for param in model.parameters()))

if args.weights != "None" :
    
    file=args.weights
    # print("load",file)
    net_weights=torch.load(file, map_location=device)
 

    model.load_state_dict(net_weights)
 
    model.eval()
    

gpu=False
   

size=args.size
 

# 
#   
#mean=[0.53586466, 0.53273463 ,0.53218424]
#std= [0.29189997 ,0.28975334, 0.29118757]

mean=[0.46255107, 0.46274765, 0.46291102]
std=[0.10283611, 0.10214186, 0.10219456]
          
    
def load_imageRGB(path=None,normalize=False):
    
        # print("load Image",path)
      
        raw_image = Image.open(path)
      
        
#        raw_image = raw_image.resize((args.size, args.size))
        
#        if raw_image.size[2]==4 :# RGBD
#       
#            background = Image.new("RGB", raw_image.size, (255, 255, 255))
#
#            background.paste(raw_image, mask = raw_image.split()[3])
#            raw_image=background
#      
#        
        
        imx_t = np.array(raw_image,dtype=np.float32)
        
        if imx_t.shape[2]==4 :
            imx_tn = np.zeros((imx_t.shape[0],imx_t.shape[1],3),dtype=np.float32)
            imx_tn[:,:,:]=imx_t[:,:,:3]
            imx_t=imx_tn
        # border
         
        
#        imx_t=np.transpose(imx_t,(1,0,2))
        
        imx_t/=255
        
        
        
   
              
        imx_t=np.transpose(imx_t,(2,0,1))   
        
        if  normalize : 
            # print("normalize")
            for b in range(args.channels) :
              imx_t[b,::]=(imx_t[b,::]  -mean [b] )/std [b]         
         
              
        # print("Image loaded SIZE",imx_t.shape)   
        
        return imx_t
    
    
 

   
  
 
# CPU CPU ou GPUGPU
 
if  args.cuda :
    model=model.cuda()
else:
    model=model.cpu()

 


imgnp=load_imageRGB(args.input,normalize=args.nor)

# print("SIZE",imgnp.shape)



palettedata = [0,0,0,255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255, 127,0,0, 0,127,0,  237, 127, 16  , 127,127,0, 127,0,127, 0,127,127]
  

# print("img size ",imgnp .shape)
(height,width)=imgnp.shape[1:3]
# print("img size ", height,width)
resimage = Image.new('P',( width ,height ))
resimage.putpalette(palettedata )
pixels = resimage.load()



imgs = torch.autograd.Variable(torch.from_numpy(imgnp))

 



#
#B, C, H, W = 2, 3, 4, 4
#x = torch.arange(B*C*H*W).view(B, C, H, W)

k_size = args.size 

stride = k_size//2

bord=16
if args.size <= 32:
    stride =16
    bord =8
 

if args.size ==  64 :
    stride = 32
    bord =16
    
if args.size ==  64 :
    stride = 44
    bord =10 
    
if args.size ==  64 :
    stride = 48
    bord =8
    
    
if args.size ==  128 :
    stride = 92
    bord =8

if args.size ==  256 :
    stride = 128
    bord =64   
    
# print("Kernel",k_size,"stride",stride,"bordure",bord)


#
#if args.size  > 32:
#    stride = int(k_size/1.5)
    
#if args.size  >32:
#    stride = args.size -16 

#print(args.size,stride,imgs.size())


deb=time.time()

nb=((height-stride+2)//stride) *((width-stride+2)//stride)
print("NB",nb, flush=True)
i=0
for _y in range (0,height-stride+1,stride): 
    y=_y
    for x  in range (0,width-stride+1,stride):
                 
        # print("\r",i,'/',nb,end=" ")
        print("I",i, flush=True)

        #print(y,y+k_size ,height,"X",x,x+k_size,width)
        if y+k_size >= height :
            y = height -k_size
            
        if x+k_size  >= width :
            x=width  - k_size
        
        
        
    
        #print(y,(y+k_size),x,(x+k_size))
        crop_img = imgs[0:,y:y+k_size,x:x+k_size]
        
        
        #print(crop_img.shape)
#            zero=0
#            for yy in range(height ):
#                        for xx in range(width ):
#                              if imgcv[im][0,yy,xx]== 0:
#                          
#                                   
#                                    zero+=1
        
        
        crop_img =crop_img.unsqueeze(0)
        
        if  args.cuda :
            crop_img=crop_img.cuda()
    
    #    print(p,img.size())
        output = model(crop_img)
     
        _, predict = torch.max(output, dim=1)
        pred = predict.cpu().numpy()
        
        #print(pred.shape)
#        print()
#        print('Y',y,'X',x)
#        print('Y',(y+ +bord-1),(y+ +k_size-bord -1))
#        print('X',(x+ bord-1), (x+k_size-bord -1))
        for yy in range(bord-1,k_size-bord):
#            print('YY',(y+yy  ),(y+yy ))
#            print('X',(x+ bord-1),(x+k_size-bord -1))
            for xx in range(bord-1,k_size-bord):
               
                if pred[0, yy,xx] != 0:    
                    pixels[x+xx,y+yy]=1 #pred[0, yy,xx] 
                   
             
        i+=1 
        
        
        
fin=time.time()      
   
# print()
print("temps",(fin-deb))   
resimage.save(args.output ) 
# print('\nsave to ', args.output  )   
sys.exit()  



'''
patches = imgs.unfold(2, kernel_size, stride).unfold(3, kernel_size , stride)
patches = patches.reshape(1, input_channels , -1, kernel_size, kernel_size)
patches=patches.permute(0,2,1,3,4)
#
#patches = F.unfold(imgs, kernel_size, stride=stride)
#
#print("Unfold",patches.size())
# 

 
nbStrideX=imgs.size(3)//stride 
#if (imgs.size(3)//kernel_size) % 2 ==1:
if   nbStrideX % 2 ==1 :
     nbStrideX-=1
print("nb patches ",patches.size(1))

print(imgs.size(3),imgs.size(3)//stride,imgs.size(3)//kernel_size,nbStrideX)









 
for p  in range(patches.size(1)) :
    
#    print(imgs.size())
    
    Y=p //  nbStrideX
    X=p % nbStrideX 
    
    print('\r',p,"/",patches.size(1),end="")
    print("             P",p,"Y",Y,'X',X)
    img= patches[0,p,:,:,:]
    
    img=img.unsqueeze(0).cuda()
    
#    print(p,img.size())
    output = model(img)
 
    _, predict = torch.max(output, dim=1)
    pred = predict.cpu().numpy()
    
    
#    print
    
#    print(Y,Y+kernel_size,X,X+kernel_size)
#    pixels[Y:(Y+kernel_size),X:(X+kernel_size)] =pred[0,0:16,0:16]
#    

#    print(pred)
    
   
#    for  YY in range(kernel_size):    
#        for XX in range(kernel_size):
#            if pred[0,YY,XX] ==1 :
##                print((Y*kernel_size+YY),(X*kernel_size+XX))
##                pixels[Y*kernel_size+YY,X*kernel_size+XX]=1
##               print((X*kernel_size+XX),(Y*kernel_size+YY))
##                
##                print(X,Y,"xy",(X*stride+XX),(Y*stride+YY),imgs.size())
#                pixels[X*stride+XX,Y*stride+YY]=1
                
    bord=8
    for  YY in range(bord-1,kernel_size-bord):    
        for XX in range(bord-1,kernel_size-bord):
            if pred[0,YY,XX] ==1 :
#                print((Y*kernel_size+YY),(X*kernel_size+XX))
#                pixels[Y*kernel_size+YY,X*kernel_size+XX]=1
#               print((X*kernel_size+XX),(Y*kernel_size+YY))
#                
#                print(X,Y,"xy",(X*stride+XX),(Y*stride+YY),imgs.size())
                xxx=X*stride+XX
                yyy=Y*stride+YY
                if xxx < imgs.size(3) and yyy < imgs.size(2) :
                   pixels[xxx,yyy]=1           
#    print(pred)
    
    
#for  Y in range(32):    
#  for X in range(32):
#      print(pixels[Y,X],end=' ')
#  print()    
    
#print(pixels[0:32,0:32])    
resimage.save(args.output ) 
print('\nsave to ', args.output  )   
sys.exit()
''' 



'''
pred = pred.squeeze()
patches = imgs.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4)
#print(patches )
print("imgs.unfold(3 ",patches.size())
patches = imgs.unfold(2, kernel_size, stride)
print(patches.size())
patches =patches.unfold(3, kernel_size, stride)
print(patches.size())
sys.exit()


 
sizeStep=1
 
nb=0

yy=0

nbSar=0
while yy  < h - size  :
    ny=yy
#    if yy + size > h:
#        ny = h - size -1  
    print("ligne",yy, "/",(h-size),"{:4f}".format(yy/(h - size)))
    xx=0
    while xx < w -size  :
 
        nx=xx
#        if xx + size > w:
#            nx = w - size -1
#  
        
        nb+=1
         
       
        crop=imgnp[:,ny:ny+size,nx:nx+size]
        b,hc,wc=crop.shape
        
        if wc == size and hc == size : 
            img = torch.from_numpy(crop)  
           #input= img.view(1,1, input_channels, size, size)     
            input_var = torch.autograd.Variable(img,volatile=True) 
            
            if  args.cuda :
                input_var  =  input_var.cuda()
             
            output  = model(input_var)  
            
            _, predict = torch.max(output, dim=1)
            pred = predict.cpu().numpy()
            
            pred = pred.squeeze()
                
                
          
                
            plt.imsave(args.output, pred)
            
            _,idx=torch.max(output[0,:],0)
            # print("output",output[0,:])
          
            if int(idx)==1 :
              pixels[nx+size//2,ny+size//2]= 1
              nbSar+=1
            resimage.save(args.output )
            
            sys.exit()
#            if int(idx)!=1 and output[0,0] < 1.:
#                pixels[nx+size//2,ny+size//2]= 2
#             
#            if int(idx)!=1 and  1. < output[0,0] < 2.:
#                pixels[nx+size//2,ny+size//2]= 3
            
        xx += sizeStep
    yy+=sizeStep   
    
     
resimage.save(args.output )


 
print('output image saved to ', args.output )
   
print(nbSar, "pixel(s) sargassum detecté(s)")  
        
           

print("FIN .... ",nb)

'''

 