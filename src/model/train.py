import os
import torch
import sys
import argparse
import wandb
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
# If required to import the path.
sys.path.insert(0, '/home/patel.aryam/SID/src')
sys.path.insert(0, '/home/patel.aryam/SID/src/model')
from PIL import Image, ImageOps
from tqdm import tqdm
import visdom
from util.util import sdmkdir
import time
import model.network as network
from data_loader import CustomDatasetDataLoader
from util.image_pool import ImagePool
import util.util as util
from model.base_model import BaseModel


class SIDModel(BaseModel):

    def initialize(self, opt):
            BaseModel.initialize(self, opt)
            self.isTrain = opt.isTrain

            self.loss_names = ['G_param','alpha','rescontruction']
            # specify the images you want to save/display. The program will call base_model.get_current_visuals
            self.visual_names = ['input_img', 'lit','alpha_pred','out','final']
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
            self.model_names = ['G','M']
            # load/define networks

            self.netG = network.define_vgg(4, 6, gpu_ids = self.gpu_ids)
                
            self.netM = network.define_G(7, 3, opt.ngf, 'unet_256', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG.to(self.device)
            self.netM.to(self.device)
            print ("##### Models #####")
            print(self.netG)
            print(self.netM) 
            if self.isTrain:
                self.fake_AB_pool = ImagePool(opt.pool_size)
                # define loss functions
                self.MSELoss = torch.nn.MSELoss()
                self.criterionL1 = torch.nn.L1Loss()
                self.bce = torch.nn.BCEWithLogitsLoss()
                # initialize optimizers
                self.optimizers = []
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-5)
                self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-5)
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_M)

    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        if 'isreal' in input:
            self.isreal = input['isreal']

    def forward(self):
        inputG = torch.cat([self.input_img, self.shadow_mask], 1)
        self.shadow_param_pred = torch.squeeze(self.netG(inputG))

        n = self.shadow_param_pred.shape[0]
        w = inputG.shape[2]
        h = inputG.shape[3]
        
        add = self.shadow_param_pred[:, [0, 2, 4]]
        mul = (self.shadow_param_pred[:, [1, 3, 5]] * 2) + 3
        
        #mul = (mul +2) * 5/3          
        add = add.view(n, 3, 1, 1).expand((n, 3, w, h))
        mul = mul.view(n, 3, 1, 1).expand((n, 3, w, h))
        
        
        addgt = self.shadow_param[:,[0,2,4]]
        mulgt = self.shadow_param[:,[1,3,5]]

        addgt = addgt.view(n, 3, 1, 1).expand((n, 3, w, h))
        mulgt = mulgt.view(n, 3, 1, 1).expand((n, 3, w, h))
        
        self.litgt = self.input_img.clone() / 2 + 0.5
        self.lit = self.input_img.clone() / 2 + 0.5
        self.lit = self.lit * mul + add
        self.litgt = (self.litgt * mulgt + addgt) * 2 - 1
        
        self.out = (self.input_img / 2 + 0.5) * (1 - self.shadow_mask_3d) + self.lit * self.shadow_mask_3d
        self.out = self.out * 2 - 1

        # Input to the M model.
        inputM = torch.cat([self.input_img, self.lit, self.shadow_mask], 1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred + 1) / 2        
        self.final = (self.input_img / 2 + 0.5) * (1 - self.alpha_pred) + self.lit * (self.alpha_pred)
        self.final = self.final * 2 - 1

    def backward(self):
        criterion = self.criterionL1 
        lambda_ = self.opt.lambda_L1 
        self.shadow_param[:, [1, 3, 5]] = (self.shadow_param[:, [1, 3, 5]]) / 2 - 1.5
        self.loss_G_param = criterion(self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_rescontruction = criterion(self.final,self.shadowfree_img) * lambda_
        self.loss = self.loss_rescontruction + self.loss_G_param
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_M.zero_grad()
        self.backward()
        self.optimizer_G.step()
        self.optimizer_M.step()

    def get_current_visuals(self):
        t= time.time()
        nim = self.input_img.shape[0]
        visual_ret = OrderedDict()
        all =[]
        for i in range(0,min(nim-1,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:])
                        row.append(im)
            
            row=tuple(row)
            row = np.hstack(row)
            if hasattr(self,'isreal'):
                if self.isreal[i] == 0:
                    row = ImageOps.crop(Image.fromarray(row),border =5)
                    row = ImageOps.expand(row,border=5,fill=(0,200,0))
                    row = np.asarray(row)
            all.append(row)
        all = tuple(all)
        
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])


def options():
    
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.dataroot = '/home/patel.aryam/SID/ISTD_dataset'
    opt.dir_param = '/home/patel.aryam/SID/ISTD_dataset/train/param'
    opt.name = 'test'
    opt.model = 'jointdistangle'

    opt.gpu_ids=[0]
    opt.pool_size = 50
    opt.lr = 0.0002
    opt.lr_policy = 'lambda'
    opt.lr_decay_iters = 50
    opt.epoch_count = 1
    opt.continue_train = True
    opt.finetuning = False
    opt.beta1 = 0.5
    opt.lambda_L1 = 100
    opt.log_scale = 0
    opt.ndf = 32
    opt.ngf = 64
    opt.norm ='batch'
    opt.save_epoch_freq = 10
    opt.niter = 100
    opt.niter_decay = 100
    opt.checkpoints_dir ='/home/patel.aryam/SID'  
    opt.isTrain = True
    opt.resize_or_crop = 'none'
    opt.loadSize = 256
    opt.init_type = 'xavier'
    opt.init_gain = 0.02
    opt.fineSize = 256
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = False  # no shuffle
    opt.no_flip = True  # no flip
    opt.no_dropout = True
    opt.use_our_mask = True
    opt.task ='sr'
    opt.batch_size = 16
    opt.epoch = '200'
    opt.verbose = 'False'

    return opt


def main():

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-adv_perception_project",
        
        # track hyperparameters and run metadata
        config={
        "architecture": "SID",
        "dataset": "ISTD_dataset",
        "epochs": 200,
        }
    )
    opt = options()
    model = SIDModel()
    model.initialize(opt)
    model.setup(opt)

    # Data Loader - 
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print("### Loaded Data ###")
    print("Length of the dataset : ", dataset_size)


    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.epoch = epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            model.epoch = epoch
        loss = model.get_current_losses()
        # Log the loss at wandb.
        wandb.log({"loss": loss})
            
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)


        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == "__main__":
    main()