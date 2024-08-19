import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class QRDataset(Dataset):
    def __init__(self, data_dir):
        super(QRDataset, self).__init__()
        self.hr_images_dir = os.path.join(data_dir, 'hr_img')
        self.lr_images_dir = os.path.join(data_dir, 'lr_img')
        
        self.hr_img = sorted(os.listdir(self.hr_images_dir))
        self.lr_img = sorted(os.listdir(self.lr_images_dir))
        
        self.transform_hr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform_lr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.hr_img)

    def __getitem__(self, index):
        hr_img_path = os.path.join(self.hr_images_dir, self.hr_img[index])
        lr_img_path = os.path.join(self.lr_images_dir, self.lr_img[index])
        
        hr_img = Image.open(hr_img_path).convert("RGB")
        lr_img = Image.open(lr_img_path).convert("RGB")
        
        hr_img = self.transform_hr(hr_img)
        lr_img = self.transform_lr(lr_img)
        
        return lr_img, hr_img