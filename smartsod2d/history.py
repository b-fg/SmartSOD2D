import os
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.framework import tensor_util
from smartsod2d.analysis import plot_xy

class History(object):
    def __init__(self, data_path="train/"):
        self.data_path = os.path.join(data_path, '')
        self.event_acc = EventAccumulator(self.data_path, size_guidance={event_accumulator.TENSORS: 0})
        self.event_acc.Reload()
        self.neglect_tensor_list = ["params"]
        self.events = {}
        if not os.path.exists(os.path.join(self.data_path, "plots")):
            os.makedirs(os.path.join(self.data_path, "plots"))

    def reload(self):
        self.event_acc.Reload()
        self.events = self._tfevents_to_numpy(self.event_acc, self.tags)

    def plot(self, tags=None, fmt="png"):
        self.reload()
        if not tags: tags = self.tags
        events = self.events
        plots_path = os.path.join(self.data_path, "plots")
        for k, v in events.items():
            if k not in tags: continue
            plot_xy(v['x'], v['y'], fname=os.path.join(plots_path, k.replace("/", "-") + "." + fmt),
                    x_label=v['x_label'], y_label=v['y_label'], fmt=fmt)

    @property
    def tags(self):
        return self.event_acc.Tags()['tensors']

    def _tfevents_to_numpy(self, event_acc, tags):
        events = {}
        for tag in tags:
            if tag in self.neglect_tensor_list: continue
            elist = event_acc.Tensors(tag)
            values = np.array([tf.make_ndarray(e.tensor_proto) for e in elist])
            steps = np.array([e.step for e in elist])
            x_label = "GlobalStep" if "_vs_" not in tag else tag.split("/")[0].split("_vs_")[-1]
            y_label = tag.split("/")[-1]
            events[tag] = {"x": steps, "y": values, "x_label": x_label, "y_label": y_label}
        return events