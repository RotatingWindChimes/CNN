""" 训练模型 """


import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from preprocess import cutmix_data, cutout_data, mixup_data


def train_model(n_epochs, model, optimizer, loss_fn, train_iter, test_iter, device, type="base"):

    model = model.to(device=device)

    # Tensorboard可视化
    writer = SummaryWriter("./logs")

    for epoch in range(1, n_epochs+1, 1):

        # 模型验证，当前模型在训练集和测试集上的损失函数与精度
        model.eval()
        for name, loader in [("train", train_iter), ("test", test_iter)]:
            total_correct = 0
            total_num = 0
            loss_sum = 0.0

            for feature, label in tqdm.tqdm(loader):
                feature = feature.to(device=device)
                label = label.to(device=device)

                output = model(feature)

                loss_sum += loss_fn(output, label).item()

                _, predicted = torch.max(output, dim=1)

                total_num += len(label)
                total_correct += (predicted == label).sum()

            writer.add_scalar(type+name+"Acc", total_correct/total_num, epoch)
            writer.add_scalar(type+name+"Loss", loss_sum, epoch)

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch}, Type {type}, {name} loss: {loss_sum}")
                print(f"Epoch {epoch}, Type {type}, {name} acc: {total_correct/total_num}")

        # 训练模型
        model.train()
        for train_feature, train_label in tqdm.tqdm(train_iter):
            train_feature = train_feature.to(device=device)
            train_label = train_label.to(device=device)

            if type == "cutmix":
                train_feature, train_label = cutmix_data(train_feature, train_label)
            elif type == "cutout":
                train_feature = cutout_data(train_feature, length=16)
            elif type == "mixup":
                train_feature, train_label = mixup_data(train_feature, train_label, alpha=1.0)

            train_output = model(train_feature)

            loss = loss_fn(train_output, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
