import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2
import glob
from model.sne_model import SNE
from dataset import custom_transforms as tr
import matplotlib.pyplot as plt


class kittiCalibInfo():
    """
    Read calibration files in the kitti dataset,
    we need to use the intrinsic parameter of the cam2
    """
    def __init__(self, filepath):
        """
        Args:
            filepath ([str]): calibration file path (AAA.txt)
        """
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        """
        Returns:
            [numpy.array]: intrinsic parameter of the cam2
        """
        return self.data['P2']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        P0 = np.reshape(rawdata['P0'], (3,4))
        P1 = np.reshape(rawdata['P1'], (3,4))
        P2 = np.reshape(rawdata['P2'], (3,4))
        P3 = np.reshape(rawdata['P3'], (3,4))
        R0_rect = np.reshape(rawdata['R0_rect'], (3,3))
        Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3,4))

        data['P0'] = P0
        data['P1'] = P1
        data['P2'] = P2
        data['P3'] = P3
        data['R0_rect'] = R0_rect
        data['Tr_velo_to_cam'] = Tr_velo_to_cam

        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data


class Kitti_Dataset(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root='./data/KITTI/', split='training'):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.depths = {}
        self.labels = {}
        self.calibs = {}

        self.image_base = os.path.join(self.root, self.split, 'image_2')
        self.depth_base = os.path.join(self.root, self.split, 'depth_u16')
        self.label_base = os.path.join(self.root, self.split, 'gt_image_2')
        self.calib_base = os.path.join(self.root, self.split, 'calib')

        self.images[split] = []
        self.images[split].extend(glob.glob(os.path.join(self.image_base, '*.png')))
        self.images[split].sort()

        self.depths[split] = []
        self.depths[split].extend(glob.glob(os.path.join(self.depth_base, '*.png')))
        self.depths[split].sort()

        self.labels[split] = []
        self.labels[split].extend(glob.glob(os.path.join(self.label_base, '*.png')))
        self.labels[split].sort()

        self.calibs[split] = []
        self.calibs[split].extend(glob.glob(os.path.join(self.calib_base, '*.txt')))
        self.calibs[split].sort()

        self.sne_model = SNE(crop_top=True)

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.image_base))
        if not self.depths[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.depth_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s depth images" % (len(self.depths[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        depth_path = self.depths[self.split][index].rstrip()
        calib_path = self.calibs[self.split][index].rstrip()

        useDir = "/".join(img_path.split('/')[:-2])
        name = img_path.split('/')[-1]
        
        _img = Image.open(img_path).convert('RGB')
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32)
        oriHeight, oriWidth = depth_image.shape
        _depth = Image.fromarray(depth_image)

        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        if not self.split == 'testing':
            lbl_path = os.path.join(useDir, 'gt_image_2', name[:-10] + 'road_' + name[-10:])
            label_image = cv2.cvtColor(cv2.imread(lbl_path), cv2.COLOR_BGR2RGB)
            label[label_image[:, :, 2] > 0] = 1

        _target = Image.fromarray(label)

        sample = {'image': _img, 'depth': _depth, 'label': _target}

        if self.split == 'training':
            sample = self.transform_tr(sample)
        elif self.split == 'validating':
            sample = self.transform_val(sample)
        elif self.split == 'testing':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)

        depth_image = np.array(sample['depth'])

        calib = kittiCalibInfo(calib_path)
        camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)
        normal = self.sne_model(torch.tensor(depth_image.astype(np.float32) / 1000), camParam)
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
        sample['depth_path'] = depth_path
        sample['calib_path'] = calib_path
        
        return sample

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
