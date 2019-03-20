import torch
import torch.nn as nn
import os
import torchvision.transforms as tfs
from PIL import Image
import numpy as np

#不是import torch.utils.data.Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




DATA = "./data"
WIDTH = 300
HEIGHT = 200







class VOCSegDataSet(Dataset):
    # 加载数据图像,train参数决定
    def loadImage(self, root=DATA, train=True):
        if train:
            txt = root + "/ImageSets/Segmentation/" + "train.txt"
        else:
            txt = root + "/ImageSets/Segmentation/" + "val.txt"
        with open(txt, 'r') as f:
            images = f.read().split()

        data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
        label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
        return data, label



    def __init__(self,train,crop_size):
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'potted plant',
                        'sheep', 'sofa', 'train', 'tv/monitor']

        # 种类对应的RGB值
        self.colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                         [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                         [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                         [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                         [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        # 将RGB值映射为一个数值
        self.cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(self.colormap):
            self.cm2lbl[cm[0] * 256 * 256 + cm[1] * 256 + cm[2]] = i
        self.crop_size = crop_size
        data_list,label_list = self.loadImage(train = train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

    # 将numpy数组替换为对应种类
    def image2label(self, im):
        data = np.array(im, dtype='int32')
        # print(data)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵

    # 选取固定区域
    def rand_crop(self, data, label, height, width):
        data = tfs.CenterCrop((height, width))(data)
        data.show()
        label = tfs.CenterCrop((height, width))(label)
        label.show()
        # label = tfs.FixedCrop(*rect)(label)
        return data, label

    def img_transforms(self, im, label, height, width):
        im, label = self.rand_crop(im, label, height, width)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),  # [0-255]--->[0-1]
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值方差
        ])
        im = im_tfs(im)
        label = self.image2label(label)
        label = torch.from_numpy(label)
        return im, label



    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]







    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)




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

    def loadImage(self, root=DATA, train=True):

        if train:
            txt = root + "/ImageSets/Segmentation/" + "train.txt"
        else:
            txt = root + "/ImageSets/Segmentation/" + "val.txt"
        with open(txt, 'r') as f:
            images = f.read().split()

        data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
        label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
        return data, label

    #将numpy数组替换为对应种类
    def image2label(self,im):
        data = np.array(im, dtype='int32')
        # print(data)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵




    #选取固定区域
    def rand_crop(self,data,label,height,width):
        data = tfs.CenterCrop((height, width))(data)
        data.show()
        label = tfs.CenterCrop((height, width))(label)
        label.show()
        # label = tfs.FixedCrop(*rect)(label)
        return data, label

    def img_transforms(self,im,label,height,width):
        im,label = self.rand_crop(im,label,height,width)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),#[0-255]--->[0-1]
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值方差
        ])
        im = im_tfs(im)
        label = self.image2label(label)
        label = torch.from_numpy(label)
        return im,label

    def main(self):
        crop_size = [HEIGHT,WIDTH]
        voc_train = VOCSegDataSet(train = True,crop_size = crop_size)
        voc_test = VOCSegDataSet(train = False,crop_size = crop_size)
        train_data = DataLoader(voc_train, 64, shuffle=True, num_workers=4)
        valid_data = DataLoader(voc_test, 128, num_workers=4)
        print(train_data)
        print(valid_data)
        # data,label = self.loadImage(train = True)
        # img1 = Image.open(data[0])
        # img2 = Image.open(label[0])
        # im,label = self.img_transforms(img1,img2,HEIGHT,WIDTH)
        # print(img1)
        # print(img2)
        # label = self.image2label(img2)
        # label = self.image2label(img2)
        # print(label[100:160, 200:250])
        # data,label = self.rand_crop(img1,img2,200,300)
        # # data.show()
        # # label.show()
        # print(data)
        # print(label)










if __name__ == "__main__":
    t = Main()
    t.main()
