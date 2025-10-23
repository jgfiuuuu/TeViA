from utils.model import TeViaSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import os
import cv2
import sys
import numpy as np
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from thop import profile, clever_format
class TeViaWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(TeViaWrapper, self).__init__()
        
        self.model = TeViaSeg(args.bert_type, args.vision_type, args.project_dim)
        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),lr =  0.0003)   
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}         
        
             
    def sim(self, x, y):
    
        return F.cosine_similarity(x, y, dim=-1)
        
    def distanceloss(self,txt_means, vis_means):
        losses = []

        for i, (txt_mean, vis_mean) in enumerate(zip(txt_means, vis_means)):
            loss = (1 - self.sim(txt_mean, vis_mean)) 
            #loss = self.wasserstein_distance(txt_mean, vis_mean)
            losses.append(loss) 

        
        total_loss = torch.mean(torch.stack(losses))
        return total_loss    
        
             
    def forward(self,x):
       
       return self.model.forward(x,self.trainer.current_epoch)


    def shared_step(self,batch,batch_idx):
        x, y = batch 
        #image, text, gt = x
        #preds = self(x)
        preds, txt_means, vis_means= self(x)
        #print(image_feat.shape,text_feat.shape)
        
        #preds,os1 = self(x)
        distanceloss = self.distanceloss(txt_means, vis_means)  #info_nce_loss  distanceloss
        loss = self.loss_fn(preds, y) + 0.1 * distanceloss
                
        return {'loss': loss, 'preds': preds.detach(), 'y': y.detach()}    
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        if  batch_idx == 0:
            self._compute_complexity(batch)
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True,sync_dist=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True,sync_dist=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True,sync_dist=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True,sync_dist=True)
        
        # ✅ 在测试结束后计算 cosine similarity
        if hasattr(self, "all_image_feats") and hasattr(self, "all_text_feats"):
            image_all = torch.cat(self.all_image_feats, dim=0)  # (N, 512)
            text_all = torch.cat(self.all_text_feats, dim=0)    # (N, 512)

            # normalize
            image_all = F.normalize(image_all, dim=1)
            text_all = F.normalize(text_all, dim=1)

            sim = F.cosine_similarity(image_all, text_all, dim=1)  # (N,)
            sim_mean = sim.mean().item()

            self.print(f"[TEST] Cosine Similarity Mean: {sim_mean:.4f}")
            self.log("test_cosine_similarity_mean", sim_mean, prog_bar=True, sync_dist=True)

            del self.all_image_feats, self.all_text_feats
            
    def _compute_complexity(self, batch):
        
        # 解包真实输入数据
        (x, y) = batch
        image, text_dict,gt = x  # 假设x结构为[image, text, dataset_id]
        
        # 转换文本输入格式
        text_input = {
            'input_ids': text_dict['input_ids'],
            'attention_mask': text_dict['attention_mask']
        }
        
        # 构造模型输入格式
        model_input = ([image, text_input, gt],1)
        
        # 计算FLOPs和Params
        with torch.no_grad():
            flops, params = profile(
                self.model,
                inputs=model_input,
                verbose=False
            )
        
        # 格式化输出
        flops, params = clever_format([flops, params], "%.3f")
        
        # 记录到日志
        self.logger.experiment.add_text(
            "Model Complexity",
            f"FLOPs: {flops}<br>Params: {params}",
            global_step=self.global_step
        )
        
        # 打印到控制台
        print(f"\nModel Complexity Analysis:")
        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
        
        self._computed_complexity = True
        return 0
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)