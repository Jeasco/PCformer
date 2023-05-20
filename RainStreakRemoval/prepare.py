import os
import cv2

datasets = ['Rain13k','Rain200H','Rain200L','SPAData']
for dataset in datasets:
    with open("./datasets/"+dataset+"/train/train.txt","w") as f:

        path_name = os.listdir("./datasets/"+dataset+"/train/input")
        for path in path_name:
            f.write('/train/input/'+path+' '+'/train/target/'+path+'\n')

for dataset in datasets:

        if dataset == 'Rain13k':
            datasets_1 = ['Test100','Rain100H','Rain100L','Test2800','Test1200']
            for dataset_1 in datasets_1:
                with open("./datasets/" + dataset + dataset_1 + "/test/test.txt", "w") as f:
                    path_name = os.listdir("./datasets/" + dataset + dataset_1 + "/test/input")
                    for path in path_name:
                        f.write('/test/input/' + path + ' ' + '/test/target/' + path + '\n')

        else:
            with open("./datasets/" + dataset + "/test/test.txt", "w") as f:
                path_name = os.listdir("./datasets/"+dataset+"/test/input")
                for path in path_name:
                    f.write('/test/input/'+path+' '+'/test/target/'+path+'\n')