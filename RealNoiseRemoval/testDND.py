import os
import cv2
import math
import torch
import torchvision
import argparse
import torch.nn as nn
import numpy as np
from model import PCFormer
import scipy.io as sio
from datasetDND import TestDataset
from bundle_submissions import bundle_submissions_srgb_v1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--test-dataset', type=str, default='./datasets/DND/input', help='dataset director for test')
parser.add_argument('--dataset-path', type=str, default='./datasets/DND/input', help='dataset director for test')
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

    if not dataset==None:
        path = "results/" + str(dataset)
        if not os.path.exists(path):
            os.makedirs(path)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch['in_img'].cuda()

            _,_,_,rgb_restored = model(image)

            pred = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

            save_file = os.path.join('DNDresults/' + 'matfile/', batch['image_name'][0] + '.mat')
            sio.savemat(save_file, {'Idenoised_crop': np.float32(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))})

            image_name = batch['image_name'][0]
            print("Processing image {}".format(image_name))
            if not dataset==None:
                cv2.imwrite(path + '/' + image_name + ".png", pred)
            else:
                cv2.imwrite(f"results/{image_name}.png" ,pred)
    bundle_submissions_srgb_v1('DNDresults/' + 'matfile/', 'srgb_results_for_server_submission/')
    os.system("rm {}".format('DNDresults/' + 'matfile/*.mat'))





def main(opt):
    model = PCFormer(channel=40).cuda()
    model = nn.DataParallel(model)
    load_checkpoint(opt, model)
    test_dataloader = get_test_loader(opt)
    test(model, test_dataloader, None)


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)






