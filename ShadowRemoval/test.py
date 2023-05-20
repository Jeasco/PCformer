import os
import cv2
import math
import torch
import torchvision
import argparse
import numpy as np
from model import PCFormer
from dataset import TestDataset, TestRealDataset
from rgb2lab import RGB2Lab

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

    rmse = []
    rmse_s = []
    rmse_ns = []
    if not dataset==None:
        path = "results/" + str(dataset)
        if not os.path.exists(path):
            os.makedirs(path)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image, target, mask = batch['in_img'].cuda(), batch['gt_img'].cuda(), batch['mask_img'].cuda()

            target = np.transpose(target.cpu().data[0].numpy(), (1, 2, 0))
            target = np.clip(target, 0, 1) * 255.
            mask = mask.cpu().data[0].numpy()

            _, _, _, rgb_restored= model(image)
            pred = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255.
            w = pred.shape[0]
            h = pred.shape[1]

            pred_1 = np.zeros((w, h, 3))
            for i in range(w):
                for j in range(h):
                    Lab = RGB2Lab(pred[i, j])
                    pred_1[i, j] = (Lab[0], Lab[1], Lab[2])


            target_1 = np.zeros((w, h, 3))
            for i in range(w):
                for j in range(h):
                    Lab = RGB2Lab(target[i, j])
                    target_1[i, j] = (Lab[0], Lab[1], Lab[2])

            image_rmse = np.sqrt(((pred_1-target_1)**2).sum()/(w*h))
            image_rmse_s = np.sqrt(((pred_1*mask-target_1*mask)**2).sum()/(mask.sum()/3))
            image_rmse_ns = np.sqrt(((pred_1*(1-mask)-target_1*(1-mask))**2).sum()/((1-mask).sum()/3))


            rmse.append(image_rmse)
            rmse_s.append(image_rmse_s)
            rmse_ns.append(image_rmse_ns)

            image_name = batch['image_name'][0]
            print("Processing image {}".format(image_name))
            if not dataset==None:
                cv2.imwrite(path + '/' + image_name + ".png", pred)
            else:
                cv2.imwrite(f"results/{image_name}-%.3g.png" % (image_rmse),pred)

    return [np.mean(rmse), np.mean(rmse_s), np.mean(rmse_ns)]




def main(opt):
    model = PCFormer().cuda()
    model = nn.DataParallel(model)
    load_checkpoint(opt, model)
    test_dataloader = get_test_loader(opt)
    metric = test(model, test_dataloader, None)
    print(metric)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)






