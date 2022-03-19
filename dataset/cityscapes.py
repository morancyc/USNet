import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2
import glob
import json
from model.sne_model import SNE
from dataset import custom_transforms as tr
import matplotlib.pyplot as plt


def Disp2depth(fx, baseline, disp):
    delta = 256
    disp_mask = disp > 0
    depth = disp.astype(np.float32)
    depth[disp_mask] = (depth[disp_mask] - 1) / delta
    disp_mask = depth > 0
    depth[disp_mask] = fx * baseline / depth[disp_mask]
    return depth


def read_calib_file(filepath):
    with open(filepath, 'r') as f:
        calib_info = json.load(f)
        baseline = calib_info['extrinsic']['baseline']
        fx = calib_info['intrinsic']['fx']
        fy = calib_info['intrinsic']['fy']
        u0 = calib_info['intrinsic']['u0']
        v0 = calib_info['intrinsic']['v0']
    return baseline, fx, fy, u0, v0


class Cityscapes_Dataset(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root='./data/Cityscapes/', split='train'):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.disparities = {}
        self.labels = {}
        self.calibs = {}

        self.image_base = os.path.join(self.root, 'leftImg8bit_trainvaltest/leftImg8bit', self.split)
        self.disparity_base = os.path.join(self.root, 'disparity_trainvaltest/disparity', self.split)
        self.label_base = os.path.join(self.root, 'gtFine_trainvaltest/gtFine', self.split)
        self.calib_base = os.path.join(self.root, 'camera_trainvaltest/camera', self.split)

        self.images[split] = []
        self.images[split] = self.recursive_glob(rootdir=self.image_base, suffix='.png')
        self.images[split].sort()

        self.disparities[split] = []
        self.disparities[split] = self.recursive_glob(rootdir=self.disparity_base, suffix='.png')
        self.disparities[split].sort()

        self.labels[split] = []
        self.labels[split] = self.recursive_glob(rootdir=self.label_base, suffix='_labelIds.png')
        self.labels[split].sort()

        self.calibs[split] = []
        self.calibs[split] = self.recursive_glob(rootdir=self.calib_base, suffix='.json')
        self.calibs[split].sort()

        self.sne_model = SNE(crop_top=False)

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.image_base))
        if not self.disparities[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.disparity_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s disparity images" % (len(self.disparities[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        disp_path = self.disparities[self.split][index].rstrip()
        calib_path = self.calibs[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()

        label_image = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        oriHeight, oriWidth = label_image.shape
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        # reserve the 'road' class
        label[label_image == 7] = 1

        _img = Image.open(img_path).convert('RGB')
        disp_image = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        baseline, fx, fy, u0, v0 = read_calib_file(calib_path)
        depth = Disp2depth(fx, baseline, disp_image)
        _depth = Image.fromarray(depth)

        _target = Image.fromarray(label)

        sample = {'image': _img, 'depth': _depth, 'label': _target}

        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)

        depth_image = np.array(sample['depth'])

        calib = np.array([[fx, 0, u0],
                          [0, fy, v0],
                          [0, 0, 1]])
        camParam = torch.tensor(calib, dtype=torch.float32)
        normal = self.sne_model(torch.tensor(depth_image.astype(np.float32)), camParam)
        normal = normal.cpu().numpy()
        normal = np.transpose(normal, [1, 2, 0])
        normal = cv2.resize(normal, (self.args.crop_width, self.args.crop_height))

        normal = transforms.ToTensor()(normal)

        sample['depth'] = normal

        sample['label'] = np.array(sample['label'])
        sample['label'] = torch.from_numpy(sample['label']).long()

        sample['oriHeight'] = oriHeight
        sample['oriWidth'] = oriWidth

        sample['img_path'] = img_path
        sample['depth_path'] = disp_path
        sample['calib_path'] = calib_path
        sample['lbl_path'] = lbl_path

        return sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.RandomGaussianNoise(),
            tr.Resize(size=(self.args.crop_width, self.args.crop_height)),
            tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(size=(self.args.crop_width, self.args.crop_height)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(size=(self.args.crop_width, self.args.crop_height)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)
