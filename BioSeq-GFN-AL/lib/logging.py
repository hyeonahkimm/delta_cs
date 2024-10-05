import pickle
import gzip
# from torch.utils.tensorboard import SummaryWriter
import copy

def get_logger(args):
    return Logger(args)

class Logger:
    def __init__(self, args):
        self.data = {}
        self.args = copy.deepcopy(vars(args))
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args}, gzip.open(save_path, 'wb'))
