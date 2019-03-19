import torch
import torch.nn as nn
import os
import torchvision.transforms as tfs
from PIL import Image


DATA = "./data"

class Main():
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
        img2 = Image.open(label[0])
        data,label = self.rand_crop(img1,img2,200,300)
        # data.show()
        # label.show()
        print(data)
        print(label)










if __name__ == "__main__":
    t = Main()
    t.main()
