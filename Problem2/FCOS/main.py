from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
from torch.utils.tensorboard import SummaryWriter
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores

def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_2d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        # print(recall, precision)
    return all_ap



def initSeed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)

def main():
    # 获取训练参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=str, default='1', help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()
    # 初始化种子，使训练过程可重复
    initSeed()
    # 画图
    writer = SummaryWriter('./logs')
    # 加载训练和测试数据
    transform = Transforms()
    train_dataset = VOCDataset(root_dir='data/voc/VOCtrainval_11-May-2012/VOCdevkit/VOC2012', resize_size=[800, 1333],
                               split='trainval', use_difficult=False, is_train=True, augment=transform)

    device = "cuda:0"
    model = FCOSDetector(mode="training").to(device)

    eval_dataset = VOCDataset(root_dir='data/voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007', resize_size=[800, 1333],
                              split='test', use_difficult=False, is_train=False, augment=None)
    print("INFO===>eval dataset has %d imgs" % len(eval_dataset))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                              collate_fn=eval_dataset.collate_fn)

    test_model = FCOSDetector(mode="inference").to(device)


    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, worker_init_fn=np.random.seed(0))
    print("total_images : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = 501

    GLOBAL_STEPS = 1
    LR_INIT = 2e-3
    LR_END = 2e-5
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=0.0001)


    for epoch in range(EPOCHS):

        losses_list = []
        for epoch_step, data in enumerate(tqdm(train_loader)):
            model.train()
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.to(device)
            batch_boxes = batch_boxes.to(device)
            batch_classes = batch_classes.to(device)

            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            if GLOBAL_STEPS == 20001:
                lr = LR_INIT * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr
            if GLOBAL_STEPS == 27001:
                lr = LR_INIT * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr
            GLOBAL_STEPS += 1

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            optimizer.step()

            losses_list.append(loss.mean().item())


        writer.add_scalar("train_loss", sum(losses_list)/len(losses_list), epoch)
        torch.save(model.state_dict(),"./checkpoint/model_{}.pth".format(epoch + 1))


        # 测试模型
        test_model.load_state_dict(torch.load("./checkpoint/model_{}.pth".format(epoch + 1), map_location="cpu"))

        test_model = test_model.eval()
        gt_boxes = []
        gt_classes = []
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        pred_losses = []
        num = 0
        for img, boxes, classes in tqdm(eval_loader):
            with torch.no_grad():
                # 计算loss
                img = img.to(device)
                boxes = boxes.to(device)
                classes = classes.to(device)
                losses = model([img, boxes, classes])
                loss = losses[-1]
                pred_losses.append(loss.mean().item())
                # 计算map
                out = test_model(img.to(device))
                pred_boxes.append(out[2][0].cpu().numpy())
                pred_classes.append(out[1][0].cpu().numpy())
                pred_scores.append(out[0][0].cpu().numpy())
            gt_boxes.append(boxes[0].cpu().numpy())
            gt_classes.append(classes[0].cpu().numpy())
            num += 1

        pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
        all_AP = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5,
                            len(eval_dataset.CLASSES_NAME))

        mAP = 0.
        for class_id, class_mAP in all_AP.items():
            mAP += float(class_mAP)
        mAP /= (len(eval_dataset.CLASSES_NAME) - 1)
        writer.add_scalar("test_loss", sum(pred_losses) / len(pred_losses), epoch)
        writer.add_scalar("mAP", mAP, epoch)
    writer.close()


if __name__=="__main__":
    main()