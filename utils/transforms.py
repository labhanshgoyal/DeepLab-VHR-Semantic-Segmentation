import torch
import numpy as np
import random
from torchvision import transforms
from PIL import Image

class JointCompose:
    def __init__(self, transforms_list):
        self.transforms=transforms_list

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class JointResize:
    def __init__(self, size):
        self.size = (size,size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        image = transforms.functional.resize(image, self.size, interpolation=Image.BILINEAR)
        mask = transforms.functional.resize(mask, self.size, interpolation=Image.NEAREST)
        return image, mask

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p=p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image=transforms.functional.hflip(image)
            mask=transforms.functional.hflip(mask)
        return image, mask

class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return image, mask

class JointRandomRotation:
    def __init__(self):
        self.angles = [0, 90, 180, 270]
    def __call__(self, image, mask):
        angle = random.choice(self.angles)
        if angle != 0:  
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle, interpolation=Image.NEAREST)
        return image, mask

class JointColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2):
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )
    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask

class JointToTensor:
    def __call__(self, image, mask):
        image = transforms.functional.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).float()
        if mask.dim() == 3:
            mask = mask.permute(2, 0, 1)
        return image, mask

class JointNormalize:
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    def __call__(self, image, mask):
        image = self.normalize(image)
        return image, mask

def get_train_transforms(image_size=512):
    return JointCompose([
        JointResize(image_size),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        JointToTensor(),
        JointNormalize(),
    ])

def get_val_transforms(image_size=512):    
    return JointCompose([
        JointResize(image_size),
        JointToTensor(),
        JointNormalize(),
    ])