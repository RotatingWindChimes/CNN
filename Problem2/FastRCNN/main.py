from config import config
from dataset import build_dataset
from train import train,test
from model.FasterRCNN import FRCNN
from torch.utils.tensorboard import SummaryWriter
from loss import build_loss
from torch.optim.lr_scheduler import MultiStepLR
import torch

def main():
    # 读取超参数
    conf = config()
    # 读取数据集
    trainDataloader, testDataloader = build_dataset("data/voc")
    # 模型
    model = FRCNN(conf).to(conf.device)
    # 损失函数
    criterion = build_loss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=conf.lr,
                                momentum=conf.momentum,
                                weight_decay=conf.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[16, 22])
    # 可视化训练中的数据
    writer = SummaryWriter('./logs')
    
    if conf.start_epoch!=0:
        checkpoint = torch.load("./logs/FasterRCNN/saves/FasterRCNN."+str(conf.start_epoch-1)+".pth.tar")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    for epoch in range(conf.start_epoch,conf.epoch):
        # 训练
        
        loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss,lr = train(trainDataloader, model, conf, criterion, optimizer, scheduler,epoch)
        writer.add_scalar("train_loss",loss,epoch)
        writer.add_scalar("train_rpn_cls_loss", rpn_cls_loss, epoch)
        writer.add_scalar("train_rpn_reg_loss", rpn_reg_loss, epoch)
        writer.add_scalar("train_rpn_fast_rcnn_cls_loss", fast_rcnn_cls_loss, epoch)
        writer.add_scalar("train_fast_rcnn_reg_loss", fast_rcnn_reg_loss, epoch)
        writer.add_scalar("train_lr", lr, epoch)

        # 验证
        loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss,mAP = test(testDataloader,model,conf,criterion)
        writer.add_scalar("val_loss", loss, epoch)
        writer.add_scalar("val_rpn_cls_loss", rpn_cls_loss, epoch)
        writer.add_scalar("val_rpn_reg_loss", rpn_reg_loss, epoch)
        writer.add_scalar("val_rpn_fast_rcnn_cls_loss", fast_rcnn_cls_loss, epoch)
        writer.add_scalar("val_fast_rcnn_reg_loss", fast_rcnn_reg_loss, epoch)
        writer.add_scalar("mAP", mAP, epoch)
    writer.close()




if __name__ == '__main__':
    main()



