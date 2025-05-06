import torch

class ThresholdFilter:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def __call__(self, probs):
        """
        probs: Tensor of shape [batch_size, num_classes]
        Returns filtered predictions: if max prob < threshold -> label = -1 (rejected)
        """
        max_probs, preds = torch.max(probs, dim=1)
        preds[max_probs < self.threshold] = -1  # mark low-confidence predictions
        return preds