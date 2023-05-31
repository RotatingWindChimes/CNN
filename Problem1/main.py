import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from initialize import init_weight
from load_and_save import save_model
from datasets import load_data
from models import create_model
from train import train_model


def main():
    # 命令行参数，训练种类
    parser = argparse.ArgumentParser(description="Type of training")
    parser.add_argument('--type', nargs="+", default='base', help="Training Type")
    parser.add_argument('--time', nargs="+", default=100, help="Training Epochs")
    args = parser.parse_args()
    train_types = args.type
    train_times = args.time

    # 数据批量
    train_iter, test_iter = load_data()

    # 训练准备
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training on {device}.")

    for (train_type, train_time) in zip(train_types, train_times):
        # 模型与参数初始化, ResNet-18
        model = create_model()
        model.apply(init_weight)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        train_model(n_epochs=int(train_time), model=model, optimizer=optimizer, loss_fn=loss_fn, train_iter=train_iter,
                    test_iter=test_iter, device=device, type=train_type)

        save_model(train_type, model)


if __name__ == "__main__":
    main()
