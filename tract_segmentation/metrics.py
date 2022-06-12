import monai
import torch
from torchmetrics import Metric


class DiceMetric(Metric):
    def __init__(self):
        super().__init__()

        self.post_processing = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.add_state("dice", default=[])

    def update(self, y_pred, y_true):
        y_pred = self.post_processing(y_pred)
        self.dice.append(monai.metrics.compute_meandice(y_pred, y_true))

    def compute(self):
        return torch.mean(torch.stack(self.dice))
