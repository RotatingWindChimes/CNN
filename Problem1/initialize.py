"""

模型参数初始化

"""

import torch.nn as nn


def init_weight(m):
    """ xavier初始化 """
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
