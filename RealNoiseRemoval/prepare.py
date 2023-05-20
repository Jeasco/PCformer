import os
import cv2

with open("./datasets/SIDD/train/train.txt","w") as f:

    path_name = os.listdir("./datasets/SIDD/train/input")
    for path in path_name:
        f.write('/input/'+path+' '+'/groundtruth/'+path+'\n')

with open("./datasets/SIDD/test/test.txt","w") as f:

    path_name = os.listdir("./datasets/SIDD/test/input")
    for path in path_name:
        f.write('/input/'+path+' '+'/groundtruth/'+path+'\n')

with open("./datasets/DND/test.txt","w") as f:

    path_name = os.listdir("./datasets/DND/input")
    for path in path_name:
        f.write('/input/'+path+'\n')
