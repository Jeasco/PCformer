import os
import cv2

with open("./datasets/train/train.txt","w") as f:

    path_name = os.listdir("./datasets/train/train_A")
    for path in path_name:
        f.write('/train/train_A/'+path+' '+'/train/train_C/'+path+'\n')

with open("./datasets/test/test.txt","w") as f:

    path_name = os.listdir("./datasets/test/test_A")
    for path in path_name:
        f.write('/test/test_A/'+path+' '+'/test/test_C/'+path+'\n')

