"""

模型参数保存与加载

"""


import torch


def save_model(train_type, model):
    """ 模型参数保存 """
    torch.save(model.state_dict(), str(train_type) + "_model_params.pkl")


def load_model(model):
    """ 模型参数加载 """
    model.load_state_dict(torch.load("model_params.pkl"))
