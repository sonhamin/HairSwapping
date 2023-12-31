import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import conv_block, up_conv, single_conv


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l*2, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
    
    
    
class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        
        
        self.Conv1_hair = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2_hair = conv_block(ch_in=64,ch_out=128)
        self.Conv3_hair = conv_block(ch_in=128,ch_out=256)
        self.Conv4_hair = conv_block(ch_in=256,ch_out=512)
        self.Conv5_hair = conv_block(ch_in=512,ch_out=1024)
        
        
        decoder = [ 
            nn.Conv2d(2048, 1024, 1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
          ]
        
        self.ConvRed = nn.Sequential(*decoder)        
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=512*3, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=256*3, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=128*3, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=64*3, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.tan = nn.Tanh()


    def forward(self,face, hair):
        # encoding path
        face1 = self.Conv1(face)

        face2 = self.Maxpool(face1)
        face2 = self.Conv2(face2)
        
        face3 = self.Maxpool(face2)
        face3 = self.Conv3(face3)

        face4 = self.Maxpool(face3)
        face4 = self.Conv4(face4)

        face5 = self.Maxpool(face4)
        face5 = self.Conv5(face5)
        
        hair1 = self.Conv1_hair(hair)

        hair2 = self.Maxpool(hair1)
        hair2 = self.Conv2_hair(hair2)
        
        hair3 = self.Maxpool(hair2)
        hair3 = self.Conv3_hair(hair3)

        hair4 = self.Maxpool(hair3)
        hair4 = self.Conv4_hair(hair4)

        hair5 = self.Maxpool(hair4)
        hair5 = self.Conv5_hair(hair5)

        # decoding + concat path
        combined = torch.cat((face5,
                              hair5,
                             ), dim=1)
        cont = self.ConvRed(combined)

        # decoding + concat path
        d5 = self.Up5(cont)
        hf4 = torch.cat((face4, hair4), dim=1)
        x4 = self.Att5(g=d5,x=hf4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        hf3 = torch.cat((face3, hair3), dim=1)
        x3 = self.Att4(g=d4,x=hf3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        hf2 = torch.cat((face2, hair2), dim=1)
        x2 = self.Att3(g=d3,x=hf2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        hf1 = torch.cat((face1, hair1), dim=1)
        x1 = self.Att2(g=d2,x=hf1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.tan(d1)


        return d1    
    
class AttU_Net_Small(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(AttU_Net_Small,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16)
        self.Conv2 = conv_block(ch_in=16,ch_out=32)
        self.Conv3 = conv_block(ch_in=32,ch_out=64)
        self.Conv4 = conv_block(ch_in=64,ch_out=128)
        self.Conv5 = conv_block(ch_in=128,ch_out=256)
        
        
        self.Conv1_hair = conv_block(ch_in=img_ch,ch_out=16)
        self.Conv2_hair = conv_block(ch_in=16,ch_out=32)
        self.Conv3_hair = conv_block(ch_in=32,ch_out=64)
        self.Conv4_hair = conv_block(ch_in=64,ch_out=128)
        self.Conv5_hair = conv_block(ch_in=128,ch_out=256)        
        
        decoder = [ 
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
          ]
        
        self.ConvRed = nn.Sequential(*decoder)        
        

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Att5 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv5 = conv_block(ch_in=128*3, ch_out=128)

        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Att4 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv4 = conv_block(ch_in=64*3, ch_out=64)
        
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Att3 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv3 = conv_block(ch_in=32*3, ch_out=32)
        
        self.Up2 = up_conv(ch_in=32,ch_out=16)
        self.Att2 = Attention_block(F_g=16,F_l=16,F_int=8)
        self.Up_conv2 = conv_block(ch_in=16*3, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)
        self.tan = nn.Tanh()



    def forward(self,face, hair):
        # encoding path
        face1 = self.Conv1(face)

        face2 = self.Maxpool(face1)
        face2 = self.Conv2(face2)
        
        face3 = self.Maxpool(face2)
        face3 = self.Conv3(face3)

        face4 = self.Maxpool(face3)
        face4 = self.Conv4(face4)

        face5 = self.Maxpool(face4)
        face5 = self.Conv5(face5)
        
        hair1 = self.Conv1_hair(hair)

        hair2 = self.Maxpool(hair1)
        hair2 = self.Conv2_hair(hair2)
        
        hair3 = self.Maxpool(hair2)
        hair3 = self.Conv3_hair(hair3)

        hair4 = self.Maxpool(hair3)
        hair4 = self.Conv4_hair(hair4)

        hair5 = self.Maxpool(hair4)
        hair5 = self.Conv5_hair(hair5)

        # decoding + concat path
        combined = torch.cat((face5,
                              hair5,
                             ), dim=1)
        cont = self.ConvRed(combined)

        # decoding + concat path
        d5 = self.Up5(cont)
        hf4 = torch.cat((face4, hair4), dim=1)
        x4 = self.Att5(g=d5,x=hf4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        hf3 = torch.cat((face3, hair3), dim=1)
        x3 = self.Att4(g=d4,x=hf3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        hf2 = torch.cat((face2, hair2), dim=1)
        x2 = self.Att3(g=d3,x=hf2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        hf1 = torch.cat((face1, hair1), dim=1)
        x1 = self.Att2(g=d2,x=hf1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.tan(d1)


        return d1        
    
    
