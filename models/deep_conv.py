import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import conv_block, up_conv, single_conv


    
class Deep_conv(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(Deep_conv,self).__init__()
        
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
        self.Up_conv5 = conv_block(ch_in=512, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

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
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.tan(d1)


        return d1