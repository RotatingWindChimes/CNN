import argparse
import torch
import tqdm
from datasets import load_data
from models import create_model
from load_and_save import load_model


def test():
    # 命令行参数，需要加载的测试模型种类
    parser = argparse.ArgumentParser(description="Type of testing")
    parser.add_argument('--type', nargs="+", default='base', help="Testing Type")

    args = parser.parse_args()
    test_types = args.type

    train_iter, test_iter = load_data()

    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

    for test_type in test_types:
        msg = "Train and test accuracy on the " + test_type + " model."
        print(msg.center(20, "="))

        model = create_model()
        load_model(model, test_type + "_model_params.pkl")
        model.to(device=device)

        for name, loader in [("train", train_iter), ("test", test_iter)]:
            total_num = 0
            total_correct = 0

            for feature, label in tqdm.tqdm(loader):
                feature = feature.to(device=device)
                label = label.to(device=device)

                output = model(feature)

                _, predicted = torch.max(output, dim=1)

                total_num += len(label)
                total_correct += (predicted == label).sum()

            print(name + " accuracy on the " + test_type + f" model is {total_correct/total_num}")


if __name__ == "__main__":
    test()
