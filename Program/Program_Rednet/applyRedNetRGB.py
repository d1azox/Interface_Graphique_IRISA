"""Test for SegNet 



python3 applyStrideLucModelResidual12B2CL.py --input /data/Sargassum/SargassumHP/Sargassum/Images12B/S2B_20180929_12b.tif --output result20180929_12b_stride_PlusBGAug_ALL.tif  --weights segnet_Sargassum_valid_best.pth --cuda


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
import os
import cv2 
 
#from sargassum_model import  SargassumNet2DV3L_3x3_Residual as SargassumNet

#from sargassum_model import  SargassumNet2DV3L_3x3_Residual as SargassumNet64
 
from segnet import  SegNet  
from red_model import  RedNet2DV3L_3x3_Residual_2B,\
        RedNet2DV3L_3x3_Residual_3B,\
        RedNet2DV3L_3x3_Residual_4B,\
        RedNet2DV3L_3x3_Residual_5B,\
        RedNet2DV3L_3x3_Residual_3B_FullResiduel,\
        RedNet2DV3L_3x3_ConcatResidual,\
        RedNet2DV3L_3x3_Residual_2B_Att,\
        RedNet2DV3L_3x3_Residual_3B_Att   
        
from PIL import Image
 
import sys
#

def load_image_RGB(path=None,normalize=True):
      
        print(path)
        raw_image = Image.open(path)
      
      
        imx_t = np.array(raw_image,dtype=np.float32)
        
        if imx_t.shape[2]==4 :
            imx_tn = np.zeros((imx_t.shape[0],imx_t.shape[1],3),dtype=np.float32)
            imx_tn[:,:,:]=imx_t[:,:,:3]
            imx_t=imx_tn
        # border
         
        imx_t/=255
        
        imx_t=np.transpose(imx_t,(2,0,1))
        if  normalize :       
            for b in range(3) :
             imx_t  [b,::]=(imx_t[b,::]  -mean [b] )/std [b] 
        print("SIZE",imx_t.shape)
        return imx_t 
    
    
def load_image_RGB_(path=None,normalize=False):
     
         print("load Image",path)
       
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
             print("normalize")
             for b in range(args.channels) :
               imx_t[b,::]=(imx_t[b,::]  -mean [b] )/std [b]         
          
               
         print("Image loaded SIZE",imx_t.shape)   
         
         return imx_t
     
    
def load_image(  path=None,channels=3,normalize=True):
        
        
        imx_t=cv2.imread(path)
        
        if imx_t is None :
             print("problme sur le fichier ",path)
             exit()
             
        imx_t = cv2.cvtColor(imx_t, cv2.COLOR_BGR2RGB)
      
        if  normalize :   
            for b in range(channels) :
              imx_t[b,:,:]=(imx_t[b,::]  -mean [b] )/std [b] 
              
              
              
              
     
            
        return imx_t    
    
 
                  

if __name__ == '__main__'  :
  

#    print(sys.argv)
#    
#    print(sys.argv[0])
#    
#    print()
#    
#    print(os.path.abspath(sys.argv[0]))
#    
#    print(os.path.split(sys.argv[0]))
#    
#    print(os.path.join(os.path.split(sys.argv[0])[0],"Sargassum128_12B_NEW_50000_V3_N_128_valid_best_best_3264.pth"))
#    
#    exit()
    parser = argparse.ArgumentParser(description='Predict Image Sargassum')
    
    
    parser.add_argument('--model', choices=['SegNet','rednet2B', 'rednet3B','rednet4B','redne5B', 'rednet2BC','rednet3BC','rednet3BAtt','rednet3BAtt'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', default=os.path.join(os.path.split(sys.argv[0])[0],"Sargassum128_12B_NEW_50000_V3_N_128_valid_best_best_3264.pth"))
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--size', default=256, type=int,help='size of stride')
    #/parser.add_argument('--model', default="64", type=str,help='model Sargaum  32 |64')
    parser.add_argument('--dims', type=str, default="(32,64,128,256)" )
    parser.add_argument('--no_normalize', action='store_true', help='use normalize')
    parser.add_argument('--threshold', type=float, default=0.75) 
                       
    args = parser.parse_args()
    
    normalize=args.no_normalize==False
    
    print("CUDA",args.cuda)
    print("Normalize",normalize)
    
    device="cpu"
    if args.cuda :
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
     
    print(device)
     
    # RGB input
    input_channels = args.channels
    #  
    nb_classe=output_channels = 2
    
    print("Load weights")
    #if args.model == "32" :
    #    model = SargassumNet(input_channels,output_channels) 
    #elif    args.model == "64" : 
    dim=eval(args.dims)
    
    
        
    channels=args.channels
   
    if args.model == 'rednet2B' :
        model=RedNet2DV3L_3x3_Residual_2B(channels,nb_classe,dims=dim)
    elif args.model == 'rednet3B' :
        model=RedNet2DV3L_3x3_Residual_3B(channels,nb_classe,dims=dim)
    elif args.model == 'rednet4B' :
        model=RedNet2DV3L_3x3_Residual_4B(channels,nb_classe,dims=dim)
    elif args.model == 'rednet5B' :
        model=RedNet2DV3L_3x3_Residual_5B(channels,nb_classe,dims=dim)
        
    elif args.model == 'rednet2BC' :
        model=RedNet2DV3L_3x3_ConcatResidual(channels,nb_classe,dims=dim)
                                             
    elif  args.model == 'rednet3BC' :
        model= RedNet2DV3L_3x3_Residual_3B_FullResiduel(channels,nb_classe,dims=dim )
    elif args.model == 'rednet2BAtt' :
        model=RedNet2DV3L_3x3_Residual_2B_Att(channels,nb_classe,dims=dim)
                                             
    elif  args.model == 'rednet3BAtt' :
        model= RedNet2DV3L_3x3_Residual_3B_Att(channels,nb_classe,dims=dim )    
    elif args.model=='SegNet':
        model = SegNet(channels,nb_classe)#,dims=dim) 

    else :
        print("Model RedNet inconnu",args.model)

        exit()     
    
    
#    model = SargassumNet64(input_channels,output_channels,dims=dim) 
    #else :
    #     print("Model inconnu" ,args.model)
    #     sys.exit(1)
    
    print('Model nb parameters:', sum(param.numel() for param in model.parameters()))
    
    if args.weights != "None" :
        
        file=args.weights
        print("load",file)
        net_weights=torch.load(file, map_location=device)
     
    
        model.load_state_dict(net_weights)
     
        model.eval()
        
    
    gpu=False
       
    
    size=args.size
     
    
    #
    #mean=\
    #[ 0.09729894,  0.08877715,  0.08311013,  0.07391645,  0.09017547,  0.10118437,
    #  0.10725799,  0.10664269,  0.10172991,  0.17433181,  0.06167458,  0.04890004]
    #std=\
    #[ 0.00434938,  0.00643737,  0.00729247,  0.00673562,  0.00925067 , 0.00968437,
    #  0.01022551,  0.0092595,   0.01164183, 0.00630978,  0.00411137,  0.0039594 ]
    
    #mean = [ 0.0849, 0.0825,0.0812,0.0807,0.0804,0.0789,0.0787,0.0786,0.0784,0.0819,0.0792,0.0767]
    #std  = [ 0.0146,  0.0121,0.0114,0.0114,0.0114,0.0110,0.0113,0.0116,0.0122,0.0108,0.0104,0.0096]
    #  
#    mean =[0.11169522 ,0.09470984, 0.08556173, 0.08355924, 0.09068398, 0.08047504, 0.07923187, 0.07932469, 0.06782003, 0.08728155, 0.08173187 ,0.07665027]
#    std = [0.01287516, 0.01350931, 0.01127256, 0.01146039, 0.00900774, 0.00936009 ,0.00970963 ,0.00911251 ,0.01095198, 0.009438  , 0.01070465 ,0.0097286 ]
#    
    
    
    
    mean =[0.03363159, 0.03662483, 0.0589837 , 0.0677574 , 0.06007062, 0.05002976 , 0.10234683 ,0.15281147 ,0.19830297, 0.24645681]
    std = [0.01872431, 0.02367595, 0.03492092, 0.03969694 ,0.04277517 ,0.03694155 ,0.06369801, 0.09619151, 0.14364759 ,0.17916777]

    mean=[0.16330559 ,0.21317067 ,0.25411275 ,0.35244804]
    std=[0.03226267, 0.0352714,  0.03615333 ,0.14350551]      
    
    #Pleiade
    mean =[0.26996489 ,0.26992821, 0.26985497, 0.26976326]
    std= [0.1120737,  0.11181934, 0.11195498, 0.11206653]   
      
    mean=[0.34621867, 0.34645218, 0.34646181]
    std=[0.07183128, 0.07182466, 0.07191835]
    
    #PCRS
    mean= [0.34911372, 0.3685896 , 0.33463062]
    std= [0.08210357, 0.0793207,  0.06960755]

    
    mean=[0.46255107, 0.46274765, 0.46291102]
    std=[0.10283611, 0.10214186, 0.10219456]
 
    # CPU CPU ou GPUGPU
     
    if  args.cuda :
        model=model.cuda()
    else:
        model=model.cpu()
    
     
    
    
    # imgnp=load_image(args.input,channels=args.channels,normalize=normalize)
    imgnp=load_image_RGB(args.input)
    
#    print("SIZE",imgnp.shape)
    
    
    
    palettedata = [0,0,0,255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255, 127,0,0, 0,127,0,  237, 127, 16  , 127,127,0, 127,0,127, 0,127,127,255,255,255]
      
    
#    print("img size ",imgnp .shape)
    (height,width)=imgnp.shape[1:3]
#    print("img size ", height,width)
    resimage = Image.new('P',( width ,height ))
    resimage.putpalette(palettedata )
    pixels = resimage.load()
    
    
    
    imgs = torch.autograd.Variable(torch.from_numpy(imgnp))
    
     
     
    
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
        bord =48    
 
    if args.size ==  512 :
            stride = 364
            bord =24  
    print("SubWindow Kernel",k_size,"stride",stride,"bordure",bord)
    
    
   
    softmax=torch.nn.Softmax(dim=1)
    
    nb=((height-stride+2)//stride) *((width-stride+2)//stride)
    
    
    import time
    deb=time.time()    
    
    pix=0
    
    i=0
    for _y in range (0,height-stride+1,stride): 
        y=_y
        for x  in range (0,width-stride+1,stride):
                     
            print("\r",i,'/',nb,end=" ",flush=True)
            #print(y,y+k_size ,height,"X",x,x+k_size,width)
            if y+k_size >= height :
                y = height -k_size
                
            if x+k_size  >= width :
                x=width  - k_size
            
            
            
        
            #print(y,(y+k_size),x,(x+k_size))
            crop_img = imgs[0:,y:y+k_size,x:x+k_size]
            
   
            
            crop_img =crop_img.unsqueeze(0)
            
            if  args.cuda :
                crop_img=crop_img.cuda()
        
        #    print(p,img.size())
            output = model(crop_img)
         
            #SOFTMAX
            predict = softmax(output)
           
            pred = predict.detach().cpu().numpy()
            pred=pred[:,1,:,:] 
            
            # print(pred)
            #MAX
            # _, predict = torch.max(output, dim=1)
            # pred = predict.cpu().numpy()
            
            # print(pred)
            # exit()
             
            
            # print(pred)
            for yy in range(bord-1,k_size-bord):
  
                for xx in range(bord-1,k_size-bord):
                    # if pred[0, yy, xx] ==1:
                    if pred[0, yy, xx] > args.threshold:
                        pix+=1
                    #if pred[0, yy,xx] != 0:
                        
                        pixels[x+xx,y+yy]=1 #pred[0, yy,xx] 
                        # pixels[x+xx,y+yy]=255#[255,255,255]
                 
            i+=1  
            
    fin=time.time()      
       
    print()
    print("temps ",(fin-deb))         
    resimage.save(args.output ) 
    print('\nsave to ', args.output  ) 
    print("total pixel",pix)
    sys.exit()  
    
    
   