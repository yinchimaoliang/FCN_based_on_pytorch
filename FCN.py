import torch
import torch.nn as nn
import os
import torchvision.transforms as tfs
from PIL import Image
import numpy as np




DATA = "./data"
WIDTH = 300
HEIGHT = 200

class Main():
    def __init__(self):


        #种类
        self.classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']


        #种类对应的RGB值
        self.colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]


        #将RGB值映射为一个数值
        self.cm2lbl = np.zeros(256**3)
        for i,cm in enumerate(self.colormap):
            self.cm2lbl[cm[0] * 256 * 256 + cm[1] * 256 + cm[2]] = i


    #将numpy数组替换为对应种类
    def image2label(self,im):
        data = np.array(im, dtype='int32')
        # print(data)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵

    #加载数据图像,train参数决定
    def loadImage(self,root = DATA,train = True):
        if train:
            txt = root + "/ImageSets/Segmentation/" + "train.txt"
        else:
            txt = root + "/ImageSets/Segmentation/" + "val.txt"
        with open(txt,'r') as f:
            images = f.read().split()

        data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
        label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
        return data, label

    def rand_crop(self,data,label,height,width):
        data = tfs.CenterCrop((height, width))(data)
        data.show()
        label = tfs.CenterCrop((height, width))(label)
        label.show()
        # label = tfs.FixedCrop(*rect)(label)
        return data, label

    def main(self):
        data,label = self.loadImage(train = True)
        img1 = Image.open(data[0])
        img2 = Image.open(label[0]).convert('RGB')
        # label = self.image2label(img2)
        label = self.image2label(img2)
        print(label[100:160, 240:250])
        # data,label = self.rand_crop(img1,img2,200,300)
        # # data.show()
        # # label.show()
        # print(data)
        # print(label)










if __name__ == "__main__":
    t = Main()
    t.main()
