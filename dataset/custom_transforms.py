import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        depth = np.array(depth).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'depth': depth,
                'label': mask}


class ToTensor(object):
    """Convert Image object in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(depth).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'depth': depth,
                'label': mask}


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = [max(1 - brightness, 0), 1 + brightness]
        self.contrast = [max(1 - contrast, 0), 1 + contrast]
        self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianBlur(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=self.radius*random.random()))

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianNoise(object):
    def __init__(self, mean=0, sigma=10):
        self.mean = mean
        self.sigma = sigma

    def gaussianNoisy(self, im, mean=0, sigma=10):
        noise = np.random.normal(mean, sigma, len(im))
        im = im + noise

        im = np.clip(im, 0, 255)
        return im

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = np.asarray(img)
            img = img.astype(np.int)
            width, height = img.shape[:2]
            img_r = self.gaussianNoisy(img[:, :, 0].flatten(), self.mean, self.sigma)
            img_g = self.gaussianNoisy(img[:, :, 1].flatten(), self.mean, self.sigma)
            img_b = self.gaussianNoisy(img[:, :, 2].flatten(), self.mean, self.sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
            img = Image.fromarray(np.uint8(img))
        return {'image': img,
                'depth': depth,
                'label': mask}


class Resize(object):
    """Resize rgb and label images, while keep depth image unchanged. """
    def __init__(self, size):
        self.size = size    # size: (w, h)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        assert img.size == depth.size == mask.size

        # resize rgb and label
        img = img.resize(self.size, Image.BILINEAR)
        # depth = depth.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}
