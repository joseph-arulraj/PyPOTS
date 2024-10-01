import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.csai import BackboneBCSAI
from ...data.utils import DiceBCELoss






class _BCSAI(nn.Module):
    def __init__(
            self,
            n_steps: int,
            n_features: int,
            rnn_hidden_size: int,
            imputation_weight: float,
            consistency_weight: float,
            classification_weight: float,
            n_classes: int,
            step_channels: int,
            dropout: float = 0.5,
            intervals: list = None,
            device = None,
            
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.classification_weight = classification_weight
        self.n_classes = n_classes
        self.step_channels = step_channels
        self.intervals = intervals
        self.device = device


        # create models
        self.model = BackboneBCSAI(n_steps, n_features, rnn_hidden_size, step_channels, intervals, self.device)
        self.f_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.b_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict, training: bool = True) -> dict:

        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(inputs)

        results = {
            "imputed_data": imputed_data,
        }

        f_logits = self.f_classifier(self.dropout(f_hidden_states))
        b_logits = self.b_classifier(self.dropout(b_hidden_states))

        f_prediction = torch.sigmoid(f_logits)
        b_prediction = torch.sigmoid(b_logits).flip(dims=[1])

        classification_pred = (f_prediction + b_prediction) / 2

        results = {
            "imputed_data": imputed_data,
            "classification_pred": classification_pred,
        }

        # if in training mode, return results with losses
        if training:
            criterion = DiceBCELoss().to(self.device)
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            # print(inputs["labels"].unsqueeze(1))
            f_classification_loss, _ = criterion(f_prediction, f_logits, inputs["labels"].unsqueeze(1).float())
            b_classification_loss, _ = criterion(b_prediction, b_logits, inputs["labels"].unsqueeze(1).float())
            classification_loss = (f_classification_loss + b_classification_loss) 

            loss = (
                self.consistency_weight * consistency_loss + 
                self.imputation_weight * reconstruction_loss +
                self.classification_weight * classification_loss
            )

            results["loss"] = loss
            results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["classification_loss"] = classification_loss
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results