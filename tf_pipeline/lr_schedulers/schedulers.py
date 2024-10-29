import math
from typing import Sequence
import logging

LOG = logging.getLogger()
TAG_NAME = "[LR-Schedulers]"

class CosineScheduler(object):
    def __init__(self, total_epochs: int, init_lr: float, final_lr: float):
        self.total_epochs = total_epochs
        self.init_lr = init_lr
        self.final_lr = final_lr

        LOG.info(f"{TAG_NAME} [CosineScheduler]: total_epochs -> {total_epochs}, \
                 init_lr -> {init_lr}, final_lr -> {final_lr}")
    
    def scheduler(self, epoch):
        alpha = ((1 - math.cos(epoch * math.pi / self.total_epochs)) / 2) * (self.final_lr - 1) + 1
        return self.init_lr * alpha

class CosineFlatScheduler(object):
    def __init__(self, total_epochs: int, init_lr: float, final_lr: float):
        self.total_epochs = total_epochs
        self.init_lr = init_lr
        self.final_lr = final_lr

        LOG.info(f"{TAG_NAME} [CosineFlatScheduler]: total_epochs -> {total_epochs}, \
                 init_lr -> {init_lr}, final_lr -> {final_lr}")
    
    def scheduler(self, epoch):
        if (epoch > (self.total_epochs // 2)):
            alpha = ((1 - math.cos((epoch - (self.total_epochs // 2)) * math.pi / (self.total_epochs // 2))) / 2) * (self.final_lr - 1) + 1 
        else:
            alpha = 1
        return self.init_lr * alpha

class DefaultStepScheduler(object):
    def __init__(self, init_lr:float, steps_to_reduce: Sequence[int]):
        self.init_lr = init_lr
        self.steps_to_reduce = list(map(int, steps_to_reduce))

        LOG.info(f"{TAG_NAME} [DefaultStepScheduler]: init_lr -> {init_lr} ,\
                 steps_to_reduce -> {steps_to_reduce}")
    
    def scheduler(self, epoch):
        lr = self.init_lr * (0.1 ** (self.steps_to_reduce.index(epoch) + 1))
        return lr

class LinearScheduler(object):
    def __init__(self, total_epochs: int, init_lr: float, final_lr: float):
        self.total_epochs = total_epochs
        self.init_lr = init_lr
        self.final_lr = final_lr

        LOG.info(f"{TAG_NAME} [LinearScheduler]: total_epochs -> {total_epochs}, \
                 init_lr -> {init_lr}, final_lr -> {final_lr}")
    
    def scheduler(self, epoch):
        alpha = (1 - epoch / self.total_epochs) * (1.0 - self.final_lr) + self.final_lr
        return self.init_lr * alpha

if __name__ == "__main__":
    total_epochs = 150
    cosine_sl = CosineFlatScheduler(total_epochs, 1e-3, 0.01)

    import matplotlib.pyplot as plt
    epochs_range = list(range(total_epochs))
    lrs = []
    for epoch in epochs_range:
        lr = cosine_sl.scheduler(epoch)
        lrs.append(lr)
    
    plt.plot(epochs_range, lrs)
    plt.show()
