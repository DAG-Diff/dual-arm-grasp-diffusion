import torch
import torch.nn as nn
from se3dif.utils import SO3_R3


class ClassifierLoss():
    def __init__(self, field='classifier', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad
        self.bce_loss = nn.BCELoss()

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        
        gt_labels = ground_truth['labels']
        
        # Compute classifier loss
        if model.classifier is not None:
            classifier_loss = self.bce_loss(model.pred_label.reshape(-1), gt_labels.reshape(-1))/5
        else:
            classifier_loss = 0
        
        acc = torch.sum((model.pred_label.reshape(-1) > 0.5).float() == gt_labels.reshape(-1)).item() / gt_labels.numel()

        info = {self.field: model.pred_label.reshape(-1)}
        loss_dict = {"Classifier Loss": classifier_loss,
                     'Classifier Accuracy': acc}
        
        return loss_dict, info
