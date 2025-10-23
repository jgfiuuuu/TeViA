import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2
#import seaborn as sns

class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        #last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        #embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        #embed = self.project_head(embed)
        text_output = output['hidden_states'][-1]
        return text_output
        
            
class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        #embeds = output['pooler_output'].squeeze()
        #project = self.project_head(embeds)

        return {"feature":output['hidden_states']}


class TeViaSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(TeViaSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim) 

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        
        
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        
        self.conv1 = nn.Conv2d(24, 3, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(24, 6, kernel_size = 1, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(24, 3, kernel_size = 1, stride = 1, padding = 0)
                
        self.feat1 = []       #torch.zero  ()#   (768) list
        self.feat2 = []  #torch.zero  ()#   (384)
        self.feat3 = []  # torch.zero  ()#   (192)
        
        self.feat1_global = None
        self.feat2_global = None
        self.feat3_global = None
  
        self.alpha = 0.99
        
    
    def forward(self, data , epoch):

        image, text, gt = data 
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)
            
        if gt.shape[1] == 3:
            gt = gt[:, 0, :, :].unsqueeze(1)
        image_output = self.encoder(image)
        image_features = image_output['feature']
        
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map   (8, 96, 56, 56) (8, 192, 28, 28) (8, 384, 14, 14) (8, 768, 7, 7)
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]
        os16,txt16 = self.decoder16(os32,image_features[2], text_output,epoch, self.feat1_global ) #(8, 196, 384)
        os8,txt8 = self.decoder8(os16,image_features[1], text_output,epoch,self.feat2_global)   #(8,   , 192)
        os4,txt4 = self.decoder4(os8,image_features[0], text_output,epoch,self.feat3_global)   #(8, 3136, 96) 
        

        
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)    #(8,24,224,224)

        B = os1.size(0)
        
        vis_feat  = (os1.detach())*(gt == 1).float()  #(b,24,224,224)  * (b,1,224,224) -> (b,24,224,224)

        os1_v2t_s1 = F.interpolate(vis_feat, size=(16, 16), mode='bilinear', align_corners=False) #b*24*16*16
        os1_v2t_s1 = os1_v2t_s1.permute(0, 2, 3, 1).contiguous()  #b*16*16*24
        os1_v2t_s1 = F.interpolate(os1_v2t_s1, size=(16, 3), mode='bilinear', align_corners=False) #b*16*16*3
        os1_v2t_s1 = os1_v2t_s1.permute(0, 3, 1, 2).contiguous() #b*3*16*16  
        os1_v2t_s1 = os1_v2t_s1.view(B, -1) #(b, 768)
        os1_v2t_s1_global = os1_v2t_s1.mean(0)  #(768)
        #print(vis_feat.shape,vis_feat.min(),vis_feat.max(),gt.min(),gt.max())
        
        os1_v2t_s2 = F.interpolate(vis_feat, size=(8, 8), mode='bilinear', align_corners=False) #b*24*16*16
        os1_v2t_s2 = os1_v2t_s2.permute(0, 2, 3, 1).contiguous()  #b*8*8*6
        os1_v2t_s2 = F.interpolate(os1_v2t_s2, size=(8, 6), mode='bilinear', align_corners=False) #b*16*16*3
        os1_v2t_s2 = os1_v2t_s2.permute(0, 3, 1, 2).contiguous() #b*6*8*8
        os1_v2t_s2 = os1_v2t_s2.view(B, -1) #(b, 384)
        os1_v2t_s2_global = os1_v2t_s2.mean(0)   #384)
        
        
        os1_v2t_s3 = F.interpolate(vis_feat, size=(8, 8), mode='bilinear', align_corners=False) #b*24*16*16
        os1_v2t_s3 = os1_v2t_s3.permute(0, 2, 3, 1).contiguous()  #b*8*8*3
        os1_v2t_s3 = F.interpolate(os1_v2t_s3, size=(8, 3), mode='bilinear', align_corners=False) #b*16*16*
        os1_v2t_s3 = os1_v2t_s3.permute(0, 3, 1, 2).contiguous() #b*3*8*8
        os1_v2t_s3 = os1_v2t_s3.view(B, -1) #(b, 192)    
        os1_v2t_s3_global = os1_v2t_s3.mean(0)   # 192)  
        
        
        
        out = self.out(os1).sigmoid()
        
        
        # reshape text feature
        txt16 = txt16.mean(1)
        txt8 = txt8.mean(1)
        txt4 = txt4.mean(1)

        
        
  
        #normalization
        txt_means = [txt16, txt8, txt4]
        vis_means = [os1_v2t_s1, os1_v2t_s2, os1_v2t_s3]

        txt_means = [F.normalize(item, p=2, dim=-1, eps=1e-12) for item in txt_means]
        vis_means = [F.normalize(item, p=2, dim=-1, eps=1e-12) for item in vis_means]
        
        
        if epoch == 20:
            #self.feat1 = self.feat1.mul_(alpha).add_((1 - alpha) * os1_v2t_s1_global)
            
            self.feat1.append(os1_v2t_s1_global.unsqueeze(0))
            self.feat2.append(os1_v2t_s2_global.unsqueeze(0))
            self.feat3.append(os1_v2t_s3_global.unsqueeze(0))
            self.feat1_global = torch.cat(self.feat1,dim=0).mean(0)  #print(self.feat1_global.shape)   # 1 epoch  forward  110
            self.feat2_global = torch.cat(self.feat2,dim=0).mean(0)
            self.feat3_global = torch.cat(self.feat3,dim=0).mean(0)
        elif epoch > 20:
            #print(self.feat1_global.shape,os1_v2t_s1_global.shape)  #() (768,)

            self.feat1_global = self.feat1_global.mul_(self.alpha).add_((1 - self.alpha) * os1_v2t_s1_global)
            self.feat2_global = self.feat2_global.mul_(self.alpha).add_((1 - self.alpha) * os1_v2t_s2_global)
            self.feat3_global = self.feat3_global.mul_(self.alpha).add_((1 - self.alpha) * os1_v2t_s3_global)
            
        

        return out,txt_means,vis_means
      
