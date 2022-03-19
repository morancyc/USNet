import os
import random
import cv2
import torch
import tqdm
import argparse
import numpy as np
from torch.utils.data import DataLoader
from dataset.kitti import Kitti_Dataset
from dataset.cityscapes import Cityscapes_Dataset
from model.usnet import USNet
from tensorboardX import SummaryWriter
from utils import poly_lr_scheduler, fast_hist, getScores
from loss import KL, ce_loss, mse_loss
import time


# validation
def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, sample in enumerate(dataloader):
            image, depth, label = sample['image'], sample['depth'], sample['label']
            oriHeight, oriWidth = sample['oriHeight'], sample['oriWidth']
            oriWidth = oriWidth.cpu().numpy()[0]
            oriHeight = oriHeight.cpu().numpy()[0]

            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                depth = depth.cuda()
                label = label.cuda()
            
            # get predict
            evidence, evidence_a, alpha, alpha_a = model(image, depth)

            s = torch.sum(alpha_a, dim=1, keepdim=True)
            p = alpha_a / (s.expand(alpha_a.shape))

            pred = p[:,1]
            pred = pred.view(args.crop_height, args.crop_width)
            pred = pred.detach().cpu().numpy()
            pred = np.array(pred)
            pred = np.uint8(pred > 0.5)
            pred = cv2.resize(pred, (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

            # get label
            label = label.squeeze()
            label = label.cpu().numpy()
            label = np.array(label)
            label = cv2.resize(np.uint8(label), (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

            hist += fast_hist(label.flatten(), pred.flatten(), args.num_classes)
        F_score, pre, recall, fpr, fnr = getScores(hist)
        print('F_score: %.3f' % F_score)
        print('pre : %.3f' % pre)
        print('recall: %.3f' % recall)
        print('fpr: %.3f' % fpr)
        print('fnr: %.3f' % fpr)
    return F_score, pre, recall, fpr, fnr


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.backbone_name))
    max_F_score = 0
    step = 0
    lambda_epochs = 50
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, sample in enumerate(dataloader_train):
            image, depth, label = sample['image'], sample['depth'], sample['label']
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                depth = depth.cuda()
                label = label.cuda()

            # network output
            evidence_sup, alpha_sup, evidence, evidence_a, alpha, alpha_a = model(image, depth)

            # compute loss
            label = label.flatten()
            loss = 0
            for v_num in range(len(alpha_sup)):
                loss += ce_loss(label, alpha_sup[v_num], args.num_classes, epoch, lambda_epochs)
            for v_num in range(len(alpha)):
                loss += ce_loss(label, alpha[v_num], args.num_classes, epoch, lambda_epochs)
            loss += 2 * ce_loss(label, alpha_a, args.num_classes, epoch, lambda_epochs)
            loss = torch.mean(loss)

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        # save checkpoints
        if epoch % args.checkpoint_step == 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'usnet_latest.pth'))

        if epoch % args.validation_step == 0:
            F_score, pre, recall, fpr, fnr = val(args, model, dataloader_val)
            file = open(os.path.join(args.save_model_path, 'F_score.txt'), mode='a+')
            file.write('epoch = %d, F_score = %f\n' % (epoch, F_score))
            file.close()
            if F_score > max_F_score:
                max_F_score = F_score
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'usnet_best.pth'))
            writer.add_scalar('epoch/F_score', F_score, epoch)
            writer.add_scalar('epoch/pre', pre, epoch)
            writer.add_scalar('epoch/recall', recall, epoch)
            writer.add_scalar('epoch/fpr', fpr, epoch)
            writer.add_scalar('epoch/fnr', fnr, epoch)


def main(params):
    # set initialization seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train for')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--dataset', type=str, default="Kitti", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=384, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1248, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone_name', type=str, default="resnet18",
                        help='The backbone model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')

    args = parser.parse_args(params)

    if args.dataset == 'Kitti':
        train_set = Kitti_Dataset(args, root=args.data, split='training')
        val_set = Kitti_Dataset(args, root=args.data, split='validating')
    elif args.dataset == 'Cityscapes':
        train_set = Cityscapes_Dataset(args, root=args.data, split='train')
        val_set = Cityscapes_Dataset(args, root=args.data, split='val')
    
    dataloader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model = USNet(args.num_classes, args.backbone_name)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    encoder_params = list(map(id, model.module.backbone.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_params, model.parameters())

    optimizer = torch.optim.AdamW([{'params': base_params},
                                    {'params':model.module.backbone.parameters(), 'lr': args.learning_rate*0.1}],
                                    lr=args.learning_rate, betas=(0.9,0.999), weight_decay=0.01)
    
    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

if __name__ == '__main__':
    params = [
        '--num_epochs', '500',
        '--learning_rate', '1e-3',
        '--data', './data/KITTI',
        '--dataset', 'Kitti',
        '--num_workers', '8',
        '--num_classes', '2',
        '--cuda', '0',
        '--batch_size', '2',
        '--save_model_path', './log/KITTI_model',
        '--backbone_name', 'resnet18',  # only support resnet18 and resnet101
        '--checkpoint_step', '1',
        '--validation_step', '20',
    ]
    main(params)
