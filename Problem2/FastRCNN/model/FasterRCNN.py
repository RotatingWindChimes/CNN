import torch.nn as nn
from torchvision.models import vgg16
from model.anchor import FRCNNAnchorMaker
from torchvision.ops import nms
import torch
from model.proposal import RegionProposalNetwork,RegionProposal,RPNTargetMaker,FastRcnnTargetMaker,FastRCNNHead
from utils.util import xy_to_cxcy,decode,cxcy_to_xy
import numpy as np


class FRCNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.num_classes = 21
        backbone = vgg16(pretrained=True)
        self.extractor = nn.Sequential(
            *list(backbone.features.children())[:-1]
        )
        self.anchor_maker = FRCNNAnchorMaker()
        self.rpn = RegionProposalNetwork()
        self.rp = RegionProposal()

        # anchor cls和reg的正例标签
        self.rpn_target_maker = RPNTargetMaker()
        self.fast_rcnn_target_maker = FastRcnnTargetMaker()

        self.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=4096, out_features=4096),
                                        nn.ReLU(inplace=True))
        self.fast_rcnn_head = FastRCNNHead(num_classes=self.num_classes, roi_size=7, classifier=self.classifier)

    def forward(self, x, bbox, label,mode="train"):
        # vgg16特征提取
        features = self.extractor(x)
        device = features.get_device()

        # 获得所有备选的anchors,shape为(w//16 * h//16 * 9,4)
        anchor = self.anchor_maker._enumerate_shifted_anchor(x.size()[2:])
        anchor = torch.from_numpy(anchor).to(device)

        # 计算anchor的正负类别和偏移
        # pred_rpn_cls:(1,anchor数,2)
        # pred_rpn_reg:(1,anchor数,4)
        pred_rpn_cls, pred_rpn_reg = self.rpn(features)

        # 利用pred_rpn_cls筛选anchor，并利用pred_rpn_reg微调anchor，rois:(2000,4)
        rois = self.rp(cls=pred_rpn_cls.squeeze(0),
                       reg=pred_rpn_reg.squeeze(0),
                       anchor=anchor,
                       mode = mode)

        # anchor类别和微调的训练目标
        target_rpn_cls, target_rpn_reg = self.rpn_target_maker(bbox=bbox, anchor=anchor)

        # 在上面得到的rois中，采样128个，获取他们的物体标签和第二次微调系数标签
        # (128),(128,4),(128,4)
        target_fast_rcnn_cls, target_fast_rcnn_reg, sample_rois = self.fast_rcnn_target_maker(bbox=bbox,
                                                                                              label=label,
                                                                                              rois=rois)
        # roi pooling步骤，获取框的物体预测标签和第二次微调，(128,21),(128,21*4)
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.fast_rcnn_head(features, sample_rois)

        # 获取对应类别的微调预测
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(128, -1, 4)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg[torch.arange(0, 128).long(), target_fast_rcnn_cls.long()]

        return (pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg), \
               (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg)

    def predict(self,x,mode="test"):
        # vgg16特征提取
        features = self.extractor(x)
        device = features.get_device()

        # 获得所有备选的anchors,shape为(w//16 * h//16 * 9,4)
        anchor = self.anchor_maker._enumerate_shifted_anchor(x.size()[2:])
        anchor = torch.from_numpy(anchor).to(device)

        # 计算anchor的正负类别和偏移
        # pred_rpn_cls:(1,anchor数,2)
        # pred_rpn_reg:(1,anchor数,4)
        pred_rpn_cls, pred_rpn_reg = self.rpn(features)

        # 利用pred_rpn_cls筛选anchor，并利用pred_rpn_reg微调anchor，rois:(2000,4)
        rois = self.rp(cls=pred_rpn_cls.squeeze(0),
                       reg=pred_rpn_reg.squeeze(0),
                       anchor=anchor,
                       mode=mode)

        # roi pooling步骤，获取框的物体预测标签和第二次微调，(anchor数,21),(anchor数,21*4)
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.fast_rcnn_head(features, rois)

        # 获取物体类别
        pred_cls = (torch.softmax(pred_fast_rcnn_cls, dim=-1))
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(-1, self.num_classes, 4)

        # 逆归一化
        pred_fast_rcnn_reg = pred_fast_rcnn_reg * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(
            torch.get_device(pred_fast_rcnn_reg))

        rois = rois.reshape(-1, 1, 4).expand_as(pred_fast_rcnn_reg)

        pred_bbox = decode(pred_fast_rcnn_reg.reshape(-1, 4), xy_to_cxcy(rois.reshape(-1, 4)))
        pred_bbox = cxcy_to_xy(pred_bbox)

        pred_bbox = pred_bbox.clamp(min=0, max=1)
        pred_bbox = pred_bbox.reshape(-1, self.num_classes * 4)
        bbox, label, score = self._suppress(pred_bbox, pred_cls, self.config)
        return bbox, label, score, rois

    def _suppress(self, raw_cls_bbox, raw_prob, config):
        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.num_classes):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.num_classes, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > config.thres
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, iou_threshold=0.3)
            bbox.append(cls_bbox_l[keep].detach().cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].detach().cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score



