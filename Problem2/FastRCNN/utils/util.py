import os
import math
import torch
import torch.distributed as dist


def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()

def bar_custom(current, total, width=30):
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d byte]" % (current / total * 100, percent_bar, current, total)
    return progress

# 将框的表示由(中心点坐标，wh)->(左上角点坐标，右下角点坐标)
def cxcy_to_xy(cxcy):
    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)

# 将框的表示由(左上角点坐标，右下角点坐标)->(中心点坐标，wh)
def xy_to_cxcy(xy):
    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=-1)


def pred2target(pred, num_classes):
    # pred to target
    out_size = pred.size(1)  # 13, 13
    pred_targets = pred.view(-1, out_size, out_size, 3, 5 + num_classes)
    pred_target_xy = pred_targets[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
    pred_target_wh = pred_targets[..., 2:4]  # 2, 3
    pred_objectness = pred_targets[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
    pred_classes = pred_targets[..., 5:].sigmoid()  # 20 / 80  classes
    return pred_target_xy, pred_target_wh, pred_objectness, pred_classes

def encode(gt_cxywh, anc_cxywh):
    tg_cxy = (gt_cxywh[:, :2] - anc_cxywh[:, :2]) / anc_cxywh[:, 2:]
    tg_wh = torch.log(gt_cxywh[:, 2:] / anc_cxywh[:, 2:])
    tg_cxywh = torch.cat([tg_cxy, tg_wh], dim=1)
    return tg_cxywh

# 微调anchor的位置和大小
def decode(tcxcy, center_anchor):
    cxcy = tcxcy[:, :2] * center_anchor[:, 2:] + center_anchor[:, :2]
    wh = torch.exp(tcxcy[:, 2:]) * center_anchor[:, 2:]
    cxywh = torch.cat([cxcy, wh], dim=1)
    return cxywh


def center_to_corner(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)


def corner_to_center(xy):

    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=-1)

# 计算IOU
def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    intersection = find_intersection(set_1, set_2)
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps
    return intersection / union


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    return


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def resume(opts, model, optimizer, scheduler):
    if opts.start_epoch != 0:
        f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(opts.start_epoch - 1))
        device = torch.device('cuda:0')
        checkpoint = torch.load(f=f, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))
    else:
        print('\nNo check point to resume.. train from scratch.\n')
    return model, optimizer, scheduler