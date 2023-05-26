from torch.utils.data import Dataset,DataLoader
import os
import glob
from PIL import Image,ImageDraw
from xml.etree.ElementTree import parse
import numpy as np
import torch
import random
import utils.dataProcess as T
from Evaluation.evaluator import Evaluator

class VOC_Dataset(Dataset):
    # 检测物体的类别，共20个
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable',
                   'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,split,img_list,anno_list):
        super(VOC_Dataset, self).__init__()
        self.img_list = img_list
        self.anno_list = anno_list
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
        boxes = boxes / np.array([image.size[0], image.size[1], image.size[0], image.size[1]], dtype=float)

        return np.array(image), boxes, labels,info

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
    root = "./data/voc"
    # 获取测试集文件名
    test_path = os.path.join(root, "VOCtest_06-Nov-2007")
    train_anno_list = glob.glob(os.path.join(test_path, '*/*/Annotations/*.xml'))
    train_img_list = [path.replace("Annotations", "JPEGImages").replace("xml", "jpg") for path in train_anno_list]

    # 将图片路径和标注路径对齐
    train_img = sorted(train_img_list)
    train_anno = sorted(train_anno_list)

    trainDataset = VOC_Dataset("train", train_img, train_anno)
    trainDataloader = DataLoader(trainDataset,batch_size=1,shuffle=True)
    evaluator = Evaluator()
    for idx,(image,boxes,labels,info) in enumerate(trainDataloader):
        info = (boxes[0].numpy(), labels[0].numpy(),
                np.ones_like(labels[0].numpy()), info['name'], info['original_wh'])
        evaluator.get_info(info)
    mAP = evaluator.evaluate("./data/voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations")
    print(mAP)