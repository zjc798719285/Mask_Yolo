import scipy.io as sio

class SummaryWriter(object):

    def __init__(self, path):
        self.path = path
        self.dict = {}

    def write(self, name, value):

        if name in self.dict:
            self.dict[name].append(value)
        else:
            self.dict[name] = []
            self.dict[name].append(value)
    def savetomat(self):
        sio.savemat(self.path, self.dict)




