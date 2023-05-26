import torch.nn as nn
import torch
from utils.util import normal_init
from utils.util import decode,xy_to_cxcy,cxcy_to_xy,find_jaccard_overlap,encode
from torchvision.ops import nms,RoIPool


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()

        num_anchors = 9
        self.inter_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # 给模型中的参数初始化
        normal_init(self.inter_layer, 0, 0.01)
        normal_init(self.cls_layer, 0, 0.01)
        normal_init(self.reg_layer, 0, 0.01)

    def forward(self, features):
        batch_size = features.size(0)
        # 3*3卷积核进一步特征提取
        x = torch.relu(self.inter_layer(features))
        # 用1*1卷积对512的维度进行线性映射，到 9 * 2
        pred_cls = self.cls_layer(x)
        # 用1*1卷积对512的维度进行线性映射，到 9 * 4
        pred_reg = self.reg_layer(x)
        # 将[1,9*4,w//16,h//16]reshape为[1,9*(w//16)*(h//16),4]
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        # 将[1,9*2,w//16,h//16]reshape为[1,9*(w//16)*(h//16),2]
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        return pred_cls, pred_reg

class RegionProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_size = 1
    def forward(self, cls, reg, anchor, mode):
        # cls :(anchor数，2)，reg:(anchor数,4),anchor:(anchor数,4)

        # 计算每个anchor positive概率
        cls = torch.softmax(cls, dim=-1)[..., 1]

        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        anchor_tensor = anchor
        # 用reg微调anchor的位置和大小，roi_tensor:(anchor数,4)
        roi_tensor = decode(reg,xy_to_cxcy(anchor_tensor.to(reg.get_device())))
        roi_tensor = cxcy_to_xy(roi_tensor).clamp(0, 1)

        # 移除掉特别小的box
        ws = roi_tensor[:, 2] - roi_tensor[:, 0]
        hs = roi_tensor[:, 3] - roi_tensor[:, 1]
        keep = (hs >= (self.min_size / 1000)) & (ws >= (self.min_size / 1000))
        roi_tensor = roi_tensor[keep, :]
        softmax_pred_cls_scores = cls[keep]

        # 对cls的结果降序排序，获得前pre_nms_top_k个最大概率是正例的anchor
        sorted_scores, sorted_scores_indices = softmax_pred_cls_scores.sort(descending=True)
        if len(sorted_scores_indices) < pre_nms_top_k:
            pre_nms_top_k = len(sorted_scores_indices)
        roi_tensor = roi_tensor[sorted_scores_indices[:pre_nms_top_k]]
        sorted_scores = sorted_scores[:pre_nms_top_k]

        # 非极大抑制,返回值是保留的index
        keep_idx = nms(boxes=roi_tensor, scores=sorted_scores, iou_threshold=0.7)
        keep_idx = keep_idx[:post_num_top_k]
        roi_tensor = roi_tensor[keep_idx].detach()

        return roi_tensor


class RPNTargetMaker(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bbox, anchor):
        bbox = bbox[0]
        # 筛掉超出边界的anchor
        anchor_keep = ((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] <= 1) & (anchor[:, 3] <= 1))
        anchor = anchor[anchor_keep]
        num_anchors = anchor.size(0)
        device = bbox.get_device()
        # 初始化label为-1
        label = -1 * torch.ones(num_anchors, dtype=torch.float32, device=bbox.get_device())

        # 计算iou，输入anchor(x,4),bbox(y,4)，那么计算的iou(x,y)
        iou = find_jaccard_overlap(anchor, bbox)

        # 对于每一个anchor最大的iou
        IoU_max, IoU_argmax = iou.max(dim=1)
        label[IoU_max < 0.3] = 0
        label[IoU_max >= 0.7] = 1

        # 对于每一个bbox中的object最大的iou
        IoU_max_per_object, IoU_argmax_per_object = iou.max(dim=0)
        label[IoU_argmax_per_object] = 1

        n_pos = (label == 1).sum()
        n_neg = (label == 0).sum()

        if n_pos > 128:
            pos_indices = torch.arange(label.size(0), device=device)[label == 1]
            perm = torch.randperm(pos_indices.size(0))
            label[pos_indices[perm[128:]]] = -1

        if n_neg > 256 - n_pos:
            if n_pos > 128:
                n_pos = 128
            neg_indices = torch.arange(label.size(0), device=device)[label == 0]
            perm = torch.randperm(neg_indices.size(0))
            label[neg_indices[perm[(256 - n_pos):]]] = -1

        # （原始anchor数）
        pad_label = -1 * torch.ones(len(anchor_keep), dtype=torch.float32, device=bbox.get_device())
        keep_indices = torch.arange(len(anchor_keep), device=device)[anchor_keep]
        pad_label[keep_indices] = label
        rpn_tg_cls = pad_label.type(torch.long)

        # (anchor数,4)
        tg_cxywh = encode(xy_to_cxcy(bbox[IoU_argmax]), xy_to_cxcy(anchor))

        pad_bbox = torch.zeros([len(anchor_keep), 4], dtype=torch.float32, device=bbox.get_device())
        pad_bbox[keep_indices] = tg_cxywh
        rpn_tg_reg = pad_bbox

        return rpn_tg_cls, rpn_tg_reg


class FastRcnnTargetMaker(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bbox, label, rois):
        bbox = bbox[0]
        label = label[0]
        device = bbox.get_device()

        #rois = torch.cat([rois, bbox], dim=0)
        # 找到对于每个框IOU最大的标注框，获取对应的标签
        iou = find_jaccard_overlap(rois, bbox)
        IoU_max, IoU_argmax = iou.max(dim=1)
        fast_rcnn_tg_cls = label[IoU_argmax] + 1

        # 最多采样32个正例 ，算上负例一共采样128个
        n_pos = int(min((IoU_max >= 0.5).sum(), 32))

        pos_index = torch.arange(IoU_max.size(0), device=device)[IoU_max >= 0.5]

        perm = torch.randperm(pos_index.size(0))
        pos_index = pos_index[perm[:n_pos]]
        n_neg = 128 - n_pos
        neg_index = torch.arange(IoU_max.size(0), device=device)[(IoU_max < 0.5) & (IoU_max >= 0.0)]
        perm = torch.randperm(neg_index.size(0))
        neg_index = neg_index[perm[:n_neg]]
        assert n_neg + n_pos == 128
        keep_index = torch.cat([pos_index, neg_index], dim=-1)

        #只保留上面采样的那128个的标签和预测框
        fast_rcnn_tg_cls = fast_rcnn_tg_cls[keep_index]
        fast_rcnn_tg_cls[n_pos:] = 0
        fast_rcnn_tg_cls = fast_rcnn_tg_cls.type(torch.long)
        sample_rois = rois[keep_index, :]
        fast_rcnn_tg_reg = encode(xy_to_cxcy(bbox[IoU_argmax][keep_index]), xy_to_cxcy(sample_rois))
        device = torch.get_device(fast_rcnn_tg_reg)
        mean = torch.FloatTensor([0., 0., 0., 0.]).to(device)
        std = torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(device)
        fast_rcnn_tg_reg = (fast_rcnn_tg_reg - mean) / std

        return fast_rcnn_tg_cls, fast_rcnn_tg_reg, sample_rois


class FastRCNNHead(nn.Module):
    def __init__(self,num_classes,roi_size,classifier):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Linear(4096, num_classes)
        self.reg_head = nn.Linear(4096, num_classes * 4)
        self.roi_pool = RoIPool(output_size=(roi_size, roi_size), spatial_scale=1.)
        self.classifier = classifier

        normal_init(self.cls_head, 0, 0.01)
        normal_init(self.reg_head, 0, 0.001)

    def forward(self, features, roi):
        device = features.get_device()
        f_height, f_width = features.size()[2:]
        scale_from_roi_to_feature = torch.FloatTensor([f_width, f_height, f_width, f_height]).to(device)
        # 把框按照feature map的尺寸放大
        scaled_roi = roi * scale_from_roi_to_feature
        scaled_roi_list = [scaled_roi]
        # roi pooling会给每个框获得一个(521,7,7)的特征向量，所以pool的输出是（框数，512，7，7）
        pool = self.roi_pool(features, scaled_roi_list)
        # reshape为(框数,512*7*7)
        x = pool.view(pool.size(0), -1)
        # 线性变换为（框数，4096）
        x = self.classifier(x)
        # (框数,num_classes)
        pred_fast_rcnn_cls = self.cls_head(x)
        # (框数,num_classes*4)
        pred_fast_rcnn_reg = self.reg_head(x)
        return pred_fast_rcnn_cls, pred_fast_rcnn_reg