import torch
import torch.nn as nn
import os
import torchvision.transforms as tfs
from PIL import Image
import numpy as np
import datetime
import torch.nn.functional as F
import torchvision



#不是import torch.utils.data.Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo



DATA = "./data"
WIDTH = 480
HEIGHT = 320
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32







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
    def rand_crop(self, data, label, crop_size):
        data = tfs.CenterCrop((crop_size[0], crop_size[1]))(data)
        label = tfs.CenterCrop((crop_size[0], crop_size[1]))(label)
        # label = tfs.FixedCrop(*rect)(label)
        return data, label

    def img_transforms(self, im, label, crop_size):
        im, label = self.rand_crop(im, label, crop_size)
        im_tfs = tfs.Compose([
            tfs.ToTensor(),  # [0-255]--->[0-1]
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值方差。Normalize之后，神经网络在训练的过程中，梯度对每一张图片的作用都是平均的，也就是不存在比例不匹配的情况
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
        img, label = self.img_transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)




class FCN(nn.Module):
    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)


    def __init__(self,num_classes):
        super(FCN,self).__init__()

        # self.pretrained_net = model_zoo.resnet34(pretrained=True)
        self.pretrained_net = torchvision.models.resnet34(pretrained = True)
        self.stage1 = nn.Sequential(*list(self.pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(self.pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(self.pretrained_net.children())[-3]  # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 8, 4, 2, bias=False)
        self.upsample_4x.weight.data = self.bilinear_kernel(num_classes, num_classes, 8)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8

        x = self.stage2(x)
        s2 = x  # 1/16

        x = self.stage3(x)
        s3 = x  # 1/32

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_2x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s


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


        self.num_classes = len(self.classes)


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

    def _fast_hist(self,label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def label_accuracy_score(self,label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc




    def main(self):
        crop_size = [HEIGHT,WIDTH]
        voc_train = VOCSegDataSet(train = True,crop_size = crop_size)
        voc_test = VOCSegDataSet(train = False,crop_size = crop_size)
        train_data = DataLoader(voc_train, TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
        valid_data = DataLoader(voc_test, VALID_BATCH_SIZE, num_workers=4)
        self.net = FCN(len(self.classes))
        self.net.cuda()
        criterion = nn.NLLLoss2d()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        for e in range(80):
            if e > 0 and e % 50 == 0:
                optimizer.set_learning_rate(optimizer.learning_rate * 0.1)
            train_loss = 0
            train_acc = 0
            train_acc_cls = 0
            train_mean_iu = 0
            train_fwavacc = 0

            prev_time = datetime.datetime.now()
            net = self.net.train()
            for data in train_data:
                im = data[0].cuda()
                label = data[1].cuda()
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)  # (b, n, h, w)
                loss = criterion(out, label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = self.label_accuracy_score(lbt, lbp, self.num_classes)
                    train_acc += acc
                    train_acc_cls += acc_cls
                    train_mean_iu += mean_iu
                    train_fwavacc += fwavacc

            net = net.eval()
            eval_loss = 0
            eval_acc = 0
            eval_acc_cls = 0
            eval_mean_iu = 0
            eval_fwavacc = 0
            for data in valid_data:
                im = data[0].cuda()
                label = data[1].cuda()
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, label)
                eval_loss += loss.data[0]

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = self.label_accuracy_score(lbt, lbp, self.num_classes)
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc

            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
        Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
                e, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
                   eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))
            time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
            print(epoch_str + time_str + ' lr: {}'.format(optimizer.learning_rate))




        # print(train_data)
        # print(valid_data)
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
