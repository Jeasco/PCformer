import os
import cv2
import math
import torch
import torchvision
import argparse
import numpy as np
from model import PCFormer
from dataset import TestDataset, TestRealDataset
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--test-dataset', type=str, default='./datasets/test', help='dataset director for test')
parser.add_argument('--dataset-path', type=str, default='./datasets/test', help='dataset director for test')
parser.add_argument('--model-path', type=str, default='./checkpoints/latest.pth', help='load weights to test')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='num of workers per GPU to use')


def get_test_loader(opt):
    test_dataset = TestDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
    return test_dataloader

def load_checkpoint(opt, model):
    print(f"=> loading checkpoint '{opt.model_path}'")

    checkpoint = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    print(f"=> loaded successfully '{opt.model_path}'")


def test(model, test_loader, dataset=None):
    model.eval()

    ssim = []
    psnr = []
    if not dataset==None:
        path = "results/" + str(dataset)
        if not os.path.exists(path):
            os.makedirs(path)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image, target = batch['in_img'].cuda(), batch['gt_img'].cuda()

            target = np.transpose(target.cpu().data[0].numpy(), (1, 2, 0))
            target = np.clip(target, 0, 1) * 255.

            _,_,_,rgb_restored= model(image)

            pred = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255.

            image_ssim = calculate_ssim(pred, target, 0)
            image_psnr = calculate_psnr(pred, target, 0)

            ssim.append(image_ssim)
            psnr.append(image_psnr)

            image_name = batch['image_name'][0]
            print("Processing image {}".format(image_name))
            if not dataset==None:
                cv2.imwrite(path + '/' + image_name + ".png", pred)
            else:
                cv2.imwrite(f"results/{image_name}-%.3g-%.3g.png" % (image_psnr, image_ssim),pred)

    return [np.mean(psnr), np.mean(ssim)]




def main(opt):
    model = PCFormer().cuda()
    model = nn.DataParallel(model)
    datasets = ['Test100','Rain100H','Rain100L','Test1200','Test2800']
    metrics = {}
    load_checkpoint(opt, model)
    for dataset in datasets:
        opt.test_dataset = opt.dataset_path + '/' + dataset
        test_dataloader = get_test_loader(opt)
        metric = test(model, test_dataloader, dataset)
        metrics[dataset] = metric
    total = np.array([0., 0.])
    for name in datasets:
        print(name+':')
        total += np.array(metrics[name])
        print(metrics[name])
    print(total / 5.)

if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)






