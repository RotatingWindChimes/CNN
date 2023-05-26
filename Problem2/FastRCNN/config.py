
class config():
    def __init__(self):
        self.data_root="./data/voc"
        self.epoch = 20
        self.lr = 2e-3
        self.num_classes = 21
        self.resize = 800
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.log_dir = './logs'
        self.thres = 0.05
        self.device = "cuda:0"
        self.start_epoch = 7

