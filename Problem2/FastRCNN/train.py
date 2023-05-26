import torch
from tqdm import tqdm
from Evaluation.evaluator import Evaluator
import numpy as np
import os

def train(dataloader,model,conf,criterion, optimizer, scheduler,epoch):
    model.train()
    device = conf.device
    total_loss = []
    total_rpn_cls_loss = []
    total_rpn_reg_loss = []
    total_fast_rcnn_cls_loss = []
    total_fast_rcnn_reg_loss = []
    total_lr = []
    for idx,(image,boxes,labels,_) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        pred, target = model(image,boxes,labels)
        loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # 统计数据
        total_loss.append(loss.item())
        total_rpn_cls_loss.append(rpn_cls_loss.item())
        total_rpn_reg_loss.append(rpn_reg_loss.item())
        total_fast_rcnn_cls_loss.append(fast_rcnn_cls_loss.item())
        total_fast_rcnn_reg_loss.append(fast_rcnn_reg_loss.item())
        total_lr.append(lr)
    save_path = os.path.join("./logs", "FasterRCNN", 'saves')
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(save_path, "FasterRCNN" + '.{}.pth.tar'.format(epoch)))

    return sum(total_loss)/len(total_loss),sum(total_rpn_cls_loss)/len(total_rpn_cls_loss),sum(total_rpn_reg_loss)/len(total_rpn_reg_loss),sum(total_fast_rcnn_cls_loss)/len(total_fast_rcnn_cls_loss),sum(total_fast_rcnn_reg_loss)/len(total_fast_rcnn_reg_loss),sum(total_lr)/len(total_lr)


def test(testDataloader,model,conf,criterion):
    model.eval()
    device = conf.device
    total_loss = []
    total_rpn_cls_loss = []
    total_rpn_reg_loss = []
    total_fast_rcnn_cls_loss = []
    total_fast_rcnn_reg_loss = []
    evaluator = Evaluator()
    for idx, (image, boxes, labels,info) in enumerate(tqdm(testDataloader)):

        image = image.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        pred, target = model(image, boxes, labels)

        loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss = criterion(pred, target)
        # 统计数据
        total_loss.append(loss.item())
        total_rpn_cls_loss.append(rpn_cls_loss.item())
        total_rpn_reg_loss.append(rpn_reg_loss.item())
        total_fast_rcnn_cls_loss.append(fast_rcnn_cls_loss.item())
        total_fast_rcnn_reg_loss.append(fast_rcnn_reg_loss.item())

        pred_bboxes, pred_labels, pred_scores, _ = model.predict(image)

        info_ = (pred_bboxes, pred_labels, pred_scores, info['name'], info['original_wh'])
        evaluator.get_info(info_)

    mAP = evaluator.evaluate("./data/voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations")




    return sum(total_loss) / len(total_loss), sum(total_rpn_cls_loss) / len(total_rpn_cls_loss), sum(
        total_rpn_reg_loss) / len(total_rpn_reg_loss), sum(total_fast_rcnn_cls_loss) / len(total_fast_rcnn_cls_loss),sum(total_fast_rcnn_reg_loss) / len(total_fast_rcnn_reg_loss),mAP