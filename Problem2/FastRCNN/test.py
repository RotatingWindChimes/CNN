import torch
from config import config
from model.FasterRCNN import FRCNN
from dataset import build_dataset
from tqdm import tqdm
from PIL import ImageDraw,Image
import numpy as np
import utils.dataProcess as T
import argparse


class_names = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable',
                   'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

def test():
    conf = config()
    model = FRCNN(conf).to(conf.device)
    checkpoint = torch.load("./logs/FasterRCNN/saves/FasterRCNN.18.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    trainDataloader, testDataloader = build_dataset("data/voc")
    for idx, (image, boxes, labels, info) in enumerate(tqdm(testDataloader)):
        pred_bboxes, pred_labels, pred_scores, rois = model.predict(image.to(conf.device))
        img_path = "./data/voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"+info["name"][0]+".jpg"
        image = Image.open(img_path)
        image2 = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        draw2 = ImageDraw.Draw(image2)
        boxes = boxes[0] * torch.tensor([image.size[0],image.size[1],image.size[0],image.size[1]])
        pred_bboxes = pred_bboxes * np.array([image.size[0],image.size[1],image.size[0],image.size[1]])
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0))
        for i,box in enumerate(pred_bboxes):
            if pred_scores[i]>0.65:
                x1, y1, x2, y2 = box
                draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(0,255,0))
                draw.text((int(x1), int(y1)),"Class:"+str(class_names[pred_labels[i]]),fill=(255, 255, 255))
                draw.text((int(x1), int(y1)+20), "Confidence:"+str(pred_scores[i]), fill=(255, 255, 255))
        rois = rois[:,0].cpu()* torch.tensor([image.size[0],image.size[1],image.size[0],image.size[1]])
        for box in rois:
            x1, y1, x2, y2 = box
            draw2.rectangle([x1, y1, x2, y2], outline=(255,0,0))
        image.save("./data/save/"+info["name"][0]+"_pred.jpg")
        image2.save("./data/save/" + info["name"][0] + "_rois.jpg")



def predict(imgPath):
    conf = config()
    model = FRCNN(conf).to(conf.device)
    checkpoint = torch.load("./logs/FasterRCNN/saves/FasterRCNN.18.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize
    ])
    image = Image.open(imgPath).convert('RGB')
    draw = ImageDraw.Draw(image)
    image1,_,_ = transform_test(image,torch.FloatTensor([[0,0,0,0]]),torch.LongTensor([1]))

    pred_bboxes, pred_labels, pred_scores, rois = model.predict(image1.to(conf.device).unsqueeze(0))
    pred_bboxes = pred_bboxes * np.array([image.size[0], image.size[1], image.size[0], image.size[1]])
    for i, box in enumerate(pred_bboxes):
        if pred_scores[i] > 0.65:
            x1, y1, x2, y2 = box
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(0, 255, 0))
            draw.text((int(x1), int(y1)),  str(class_names[pred_labels[i]]), fill=(255,0,0))
            draw.text((int(x1), int(y1) + 10), str(round(pred_scores[i],2)), fill=(255,0,0))

    image.save("./data/unseen/" + name.partition(".")[0] + "_pred.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Figure Name")
    parser.add_argument('--name', nargs="+", default='03.jpg', help="Figure Name")
    args = parser.parse_args()
    names = args.name

    for name in names:
        print(name)
        predict("./data/unseen/" + name)
