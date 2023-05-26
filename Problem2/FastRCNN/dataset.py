from torch.utils.data import Dataset,DataLoader
import os
import glob
from PIL import Image,ImageDraw
from xml.etree.ElementTree import parse
import numpy as np
import torch
import random
import utils.dataProcess as T


def build_dataset(root):
    # 获取训练集文件名
    train_path = os.path.join(root,"VOCtrainval_06-Nov-2007")
    train_img_list = glob.glob(os.path.join(train_path, '*/*/JPEGImages/*.jpg'))
    train_anno_list = glob.glob(os.path.join(train_path, '*/*/Annotations/*.xml'))

    # 将图片路径和标注路径对齐
    train_img_list = sorted(train_img_list)
    train_anno_list = sorted(train_anno_list)

    # 获取测试集文件名
    test_path = os.path.join(root,"VOCtest_06-Nov-2007")
    test_anno_list = glob.glob(os.path.join(test_path, '*/*/Annotations/*.xml'))
    test_img_list = glob.glob(os.path.join(test_path, '*/*/JPEGImages/*.jpg'))
    test_anno_list = sorted(test_anno_list)
    test_img_list = sorted(test_img_list)


    # 数据处理方式
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize([800], max_size=1333),
        normalize
    ])

    transform_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize
    ])

    trainDataset = VOC_Dataset("train",train_img_list,train_anno_list,transform_train)
    testDataset = VOC_Dataset("test",test_img_list, test_anno_list, transform_test)
    trainDataloader = DataLoader(trainDataset,batch_size=1,shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=1, shuffle=True)
    return trainDataloader,testDataloader


class VOC_Dataset(Dataset):
    # 检测物体的类别，共20个
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable',
                   'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,split,img_list,anno_list,transform):
        super(VOC_Dataset, self).__init__()
        self.img_list = img_list
        self.anno_list = anno_list
        self.transform = transform
        self.split = split

        # 将分类类别映射到0~19的数字，形成两个字典
        self.class_idx_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.idx_class_dict = {i: class_name for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, idx):
        # 加载对应的图片
        image = Image.open(self.img_list[idx]).convert('RGB')
        basename = os.path.basename(self.img_list[idx])
        # 解析标注文件
        boxes, labels = self.parse_voc(self.anno_list[idx])
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        # 保留当前图像路径和宽高信息
        info = {}
        img_name = os.path.basename(self.anno_list[idx]).split('.')[0]
        info['name'] = img_name
        info['original_wh'] = [image.size[0], image.size[1]]

        image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels,info

    def __len__(self):
        return len(self.img_list)

    # 解析anno的xml文件
    def parse_voc(self, xml_file_path):
        tree = parse(xml_file_path)
        root = tree.getroot()
        boxes = []
        labels = []
        # 读取其中的每一个object
        for obj in root.iter("object"):
            name = obj.find('./name')
            class_name = name.text.lower().strip()
            # 把标签转化为0~19的数字
            labels.append(self.class_idx_dict[class_name])

            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            #
            x_min = float(x_min.text) - 1
            y_min = float(y_min.text) - 1
            x_max = float(x_max.text) - 1
            y_max = float(y_max.text) - 1

            boxes.append([x_min, y_min, x_max, y_max])
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))


if __name__ == '__main__':
    trainDataloader,testDataloader = build_dataset("data/voc")
    for image,boxes,labels,info in testDataloader:
        continue

    '''
    # 数据翻转效果演示
    image = Image.open('data/voc\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000346.jpg').convert('RGB')
    boxes = torch.tensor([[123,106,229,342],[136,77,496,374],[88,201,128,246],[71,208,110,258]])
    labels = [0,0,0,0]
    f = T.RandomHorizontalFlip(1)
    image,boxes,labels = f(image,boxes,labels)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(list(box), outline=(255, 255, 0))
    image.show()
    '''