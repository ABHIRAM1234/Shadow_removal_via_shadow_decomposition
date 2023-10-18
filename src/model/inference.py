import sys
import torch
from PIL import Image
sys.path.insert(0,'/home/patel.aryam/SID/src')
from model.train import SIDModel
from model.train import options
from model.data_loader import CustomDatasetDataLoader


def save_images(ordered_dict, directory):
    for key, image in ordered_dict.items():
        image = Image.fromarray(image)  # Convert numpy array to PIL Image
        image.save(f'{directory}/{key}.png')  #

def inference(opt):
    opt.gpu_ids=[0]
    opt.checkpoints_dir ='/home/patel.aryam/SID'  
    opt.netG = 'vgg'
    opt.fineSize = 256
    opt.loadSize = 256
    opt.isTrain = False
    model = SIDModel()
    model.initialize(opt)
    model.setup(opt)
    model.eval()
    

    # Dataloader - 
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize()
    dataset = data_loader.load_data()

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.forward()

        images = model.get_current_visuals()
        save_images(images, '/home/patel.aryam/SID/result')
        break


if __name__ == "__main__":
    opt = options()
    inference(opt)