import os
import cv2
import time
import random
import torch
import torchvision
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
from model import PCFormer
from dataset import TrainDataset
from loss import L1Loss, L2Loss, SSIMLoss, CharbonnierLoss, EdgeLoss, PSNRLoss, ContrastLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# set random_seed
setup_seed(2022)


parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset', type=str, default='./datasets/SIDD/train', help='dataset folder for train (change for different datasets)')
parser.add_argument('--val-dataset', type=str, default='./datasets/SIDD/test', help='dataset forder for val')
parser.add_argument('--save-epoch-freq', type=int, default=2 , help='how often to save model')
parser.add_argument('--print-freq', type=int, default=20, help='how often to print training information')
parser.add_argument('--val', type=bool, default=False, help='val during training or not')
parser.add_argument('--val-freq', type=int, default=5, help='how often to val model')
parser.add_argument('--train-visual-freq', type=int, default=20, help='how often to visualize training process')
parser.add_argument('--val-visual-freq', type=int, default=30, help='how often to visualize validation process')
parser.add_argument('--resume', type=str, default=None, help='continue training from this checkpoint')
parser.add_argument('--start-epoch', type=int, default=1, help='start epoch')
parser.add_argument('--output-dir', type=str, default='./checkpoints', help='model saved folder')
parser.add_argument('--log-dir', type=str, default='./log', help='save visual image')
parser.add_argument('--epochs', type=int, default=40, help='total number of epoch')
parser.add_argument('--image-size', type=int, default=256, help='image crop size')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='num of workers per GPU to use')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')


def build_model(opt):
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    model = PCFormer(channel=40)
    model.cuda()
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    #=============load pre-train model=========================
    checkpoint = torch.load('./checkpoints/pre_training.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    #============================================

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=opt.lr,
                                 betas=(0.9, 0.999),eps=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs,
                                                            eta_min=1e-6)

    l1loss = L1Loss().cuda()
    l2loss = L2Loss().cuda()
    ssimloss = SSIMLoss().cuda()
    carloss = CharbonnierLoss()
    edgeloss = EdgeLoss()
    contrastloss = ContrastLoss().cuda()
    psnrloss = PSNRLoss()
    loss = {'l1':l1loss,'l2':l2loss,'ssim':ssimloss,'carloss':carloss,'edgeloss':edgeloss,'contrastloss':contrastloss,'psnrloss':psnrloss}

    return model, optimizer, loss, scheduler

def get_trainval_loader(opt):
    train_dataset = TrainDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers)

    return train_dataloader

def save_image(epoch, name, img_lists, opt):
    data, pred, label = img_lists
    data = data.cpu().data
    pred = pred.cpu().data
    label = label.cpu().data

    data, label, pred = data * 255, label * 255, pred * 255
    pred = np.clip(pred, 0, 255)

    h, w = pred.shape[-2:]

    gen_num = (2, 2)
    img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
    for i in range(gen_num[0]):
        row = i * h
        for j in range(gen_num[1]):
            idx = i * gen_num[1] + j
            tmp_list = [data[idx], pred[idx], label[idx]]
            for k in range(3):
                col = (j * 3 + k) * w
                tmp = np.transpose(tmp_list[k], (1, 2, 0))
                img[row: row+h, col: col+w] = tmp

    img_file = os.path.join(opt.log_dir, '%d_%s.jpg' % (epoch, name))
    cv2.imwrite(img_file, img)

def load_checkpoint(opt, model, optimizer):
    print(f"=> loading checkpoint '{opt.resume}'")

    checkpoint = torch.load(opt.resume, map_location='cpu')
    opt.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"=> loaded successfully '{opt.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(epoch, model, optimizer):
    print('==> Saving Epoch: {}'.format(epoch))
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(opt.output_dir, 'latest.pth'))



def train(epoch, model, train_loader, optimizer, train_loss, opt):

    epoch_start_time = time.time()
    model.train()
    total_epoch_train_loss = []
    for batch_iter, data in enumerate(train_loader):
        image, target = data['in_img'].cuda(), data['gt_img'].cuda()

        optimizer.zero_grad()
        output_1, output_2, output_3 , output_4 = model(image)

        losses = train_loss['l1'](output_1, target) + \
                 train_loss['l1'](output_2, target) + \
                 train_loss['l1'](output_3, target) + \
                 train_loss['l1'](output_4, target)

        losses.backward()
        optimizer.step()

        total_epoch_train_loss.append(losses.cpu().data)

        if (batch_iter+1) % opt.print_freq == 0:
            print('Epoch: {}, Epoch_iter: {}, Loss: {}'.format(epoch, batch_iter+1, losses))
        if (batch_iter+1) % opt.train_visual_freq == 0:
            print('Saving training image epoch: {}'.format(epoch))
            save_image(epoch, 'train',[image, output_2, target] , opt)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.epochs, time.time() - epoch_start_time))

    return np.mean(total_epoch_train_loss) / opt.batch_size

def val(epoch, model, val_loader, val_loss, opt):
    model.eval()
    total_epoch_val_loss = []
    for i, data in enumerate(val_loader):
        image, target = data['in_img'].cuda(), data['gt_img'].cuda()

        output_1, output_2, output_3, output_4 = model(image)

        losses = val_loss['l1'](output_1, target) + \
                 val_loss['l1'](output_2, target) + \
                 val_loss['l1'](output_3, target) + \
                 val_loss['l1'](output_4, target)

        total_epoch_val_loss.append(losses.cpu().data)

        if (i+1) % opt.val_visual_freq == 0:
            print('Saving validation image epoch: {}'.format(epoch))
            save_image(epoch, 'val', [image, final, target], opt)
    return np.mean(total_epoch_val_loss) / opt.batch_size

def main(opt):

    model, optimizer, loss, scheduler = build_model(opt)
    scheduler.step()
    if opt.val:
        train_loader, val_loader = get_trainval_loader(opt)
    else:
        train_loader = get_trainval_loader(opt)

    total_train_loss = []
    total_val_loss = []

    if opt.resume:
        assert os.path.isfile(opt.resume)
        load_checkpoint(opt, model, optimizer)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        epoch_train_loss = train(epoch, model, train_loader, optimizer, loss, opt)
        total_train_loss.append(epoch_train_loss)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % epoch)
            save_checkpoint(epoch, model, optimizer)

        if opt.val and epoch % opt.val_freq == 0:
            epoch_val_loss = val(epoch, model, val_loader, loss, opt)
            print('Epoch: {}, Validation_Loss: {}'.format(epoch, epoch_val_loss))
            total_val_loss.append(epoch_val_loss)
        scheduler.step()

    plt.subplot(211)
    plt.plot(total_train_loss)
    plt.subplot(212)
    plt.plot(total_val_loss)
    plt.savefig(opt.log_dir + '/loss.png')

if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)
