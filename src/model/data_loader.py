import importlib
from collections import OrderedDict
import time
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os.path
from PIL import Image,ImageChops
from PIL import ImageFilter
from PIL import Image

from pdb import set_trace as st
import random
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self):
        pass

    def __len__(self):
        return 0

def get_transform(opt):
    transform_list = []
    resize_or_crop="resize_and_crop"
    loadsize=256
    fineSize=256
    isTrain = True
    no_flip = False
    
    if resize_or_crop == 'resize_and_crop':
        osize = [loadSize, loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))

    if isTrain and not no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def is_image_file(filename):
    # Define a list of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # Check if the filename's extension is in the list of image extensions
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def make_dataset(dir):
    images = []
    imname = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                imname.append(fname)
    return images,imname


class ShadowParamDataset(BaseDataset):
    def initialize(self, opt):
        dataroot= opt.dataroot #"/home/patel.aryam/SID/ISTD_dataset"
        self.root = dataroot
        self.dir_A = os.path.join(dataroot,'train/train_A')
        self.dir_B = os.path.join(dataroot,'train/train_B')
        self.dir_C = os.path.join(dataroot,'train/train_C')
        self.dir_param = opt.dir_param
    
        self.A_paths,self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5,0.5,0.5],
                                               std = [0.5,0.5,0.5])]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])
     
    def __getitem__(self,index):
        birdy = {}
        A_path = self.A_paths[index % self.A_size]
        imname = self.imname[index % self.A_size]
        index_A = index % self.A_size
        
        B_path = os.path.join(self.dir_B,imname.replace('.jpg','.png'))
        if not os.path.isfile(B_path):
            B_path = os.path.join(self.dir_B,imname)
        A_img = Image.open(A_path).convert('RGB')
        sparam = open(os.path.join(self.dir_param, imname + '.txt'))
        line = sparam.read()
        shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
        shadow_param = shadow_param[0:6]
        
        ow = A_img.size[0]
        oh = A_img.size[1]
        w = np.float64(A_img.size[0])
        h = np.float64(A_img.size[1])
        if os.path.isfile(B_path): 
            B_img = Image.open(B_path)
        else:
            print('MASK NOT FOUND : %s'%(B_path))
            B_img = Image.fromarray(np.zeros((int(w),int(h)),dtype = np.float64),mode='L')
        
        birdy['C'] = Image.open(os.path.join(self.dir_C,imname)).convert('RGB')
       
        loadSize = 256

        if w>h:
            ratio = np.float64(loadSize)/np.float64(h)
            neww = np.intc(w*ratio)
            newh = loadSize
        else:
            ratio = np.float64(loadSize)/np.float64(w)
            neww = loadSize
            newh = np.intc(h*ratio)

        birdy['A'] = A_img
        birdy['B'] = B_img
        t =[Image.FLIP_LEFT_RIGHT,Image.ROTATE_90]
        for i in range(0,4):
            c = np.random.randint(0,3,1,dtype=np.intc)[0]
            if c==2: continue
            for i in ['A','B','C']:
                if i in birdy:
                    birdy[i]=birdy[i].transpose(t[c])
                

        degree=np.random.randint(-20,20,1)[0]
        for i in ['A','B','C']:
            birdy[i]=birdy[i].rotate(degree)
        

        for k,im in birdy.items():
            birdy[k] = im.resize((neww, newh),Image.NEAREST)
        
        w = birdy['A'].size[0]
        h = birdy['A'].size[1]

        birdy['penumbra'] =  ImageChops.subtract(birdy['B'].filter(ImageFilter.MaxFilter(11)),birdy['B'].filter(ImageFilter.MinFilter(11))) 

                    
        for k,im in birdy.items():
            birdy[k] = self.transformB(im)
        
        for i in ['A','C','B']:
            if i in birdy:
                birdy[i] = (birdy[i] - 0.5)*2         
        
        no_crop=False
        fineSize= 256
        no_flip = True
        
        if not no_crop:        
            w_offset = random.randint(0,max(0,w-fineSize-1))
            h_offset = random.randint(0,max(0,h-fineSize-1))
            for k,im in birdy.items():   
                birdy[k] = im[:, h_offset:h_offset + fineSize, w_offset:w_offset + fineSize]
        
        if (not no_flip) and random.random() < 0.5:
            idx = [i for i in range(birdy['A'].size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            for k,im in birdy.items():
                birdy[k] = im.index_select(2, idx)
        for k,im in birdy.items():
            birdy[k] = im.type(torch.FloatTensor)
        birdy['imname'] = imname
        birdy['w'] = ow
        birdy['h'] = oh
        birdy['A_paths'] = A_path
        birdy['B_baths'] = B_path
         #if the shadow area is too small, let's not change anything:
        if torch.sum(birdy['B']>0) < 30 :
            shadow_param=[0,1,0,1,0,1]
            

        birdy['param'] = torch.FloatTensor(np.array(shadow_param))
        
        return birdy 
    
    def __len__(self):
        return max(self.A_size, self.B_size)

   
    def name(self):
        return 'ShadowParamDataset'


class CustomDatasetDataLoader():
    def initialize(self, opt):
        self.dataset = ShadowParamDataset()
        self.dataset.initialize(opt)
        self.batch_size = opt.batch_size
        # print("1.Done")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            self.batch_size,
            shuffle= True,
            num_workers=1)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), float("inf"))

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= float("inf"):
                break
            yield data

class SingleDataset(BaseDataset):
    def __init__(self, dataroot,opt):
        self.opt = opt
        self.root = dataroot
        self.dir_A = os.path.join(dataroot)
        self.dir_B = opt.mask_test       
        print('A path %s'%self.dir_A)

        self.A_paths,self.imname = make_dataset(self.dir_A)
        self.B_paths,tmp = make_dataset(self.dir_B)
        
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.imname = sorted(self.imname)
        self.transformB = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        imname = self.imname[index] 
        A_path= os.path.join(self.dir_A,imname)
        B_path= os.path.join(self.dir_B,imname)
        A_img = Image.open(A_path).convert('RGB')
        if not os.path.isfile(B_path):
            B_path=B_path[:-4]+'.png'
        B_img = Image.open(B_path).convert('L')
           
        ow = A_img.size[0]
        oh = A_img.size[1]
        loadsize = self.opt.fineSize if hasattr(self.opt,'fineSize') else 256
        A_img_ori = A_img
        A_img = A_img.resize((loadsize,loadsize))
        B_img = B_img.resize((loadsize,loadsize))
        A_img = torch.from_numpy(np.asarray(A_img,np.float32).transpose(2,0,1)).div(255)
        A_img_ori = torch.from_numpy(np.asarray(A_img_ori,np.float32).transpose(2,0,1)).div(255)
        B_img = self.transformB(B_img)
        B_img = B_img*2-1
        A_img = A_img*2-1
        A_img_ori = A_img_ori*2-1
        A_img = A_img.unsqueeze(0)
        A_img_ori = A_img_ori.unsqueeze(0)
        B_img = B_img.unsqueeze(0)
        B_img = (B_img>0.2).type(torch.float)*2-1

        return {'A': A_img,'B':B_img,'A_ori':A_img_ori, 'A_paths': A_path,'imname':imname,'w':ow,'h':oh}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'