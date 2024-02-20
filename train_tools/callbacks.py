from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt

class Visualizer(Callback):
    def __init__(self):
        self.state = {"epochs": 0}
    def on_train_epoch_end(self, trainer, pl_module):
        self.state["epochs"] += 1