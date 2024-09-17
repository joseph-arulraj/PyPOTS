import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.deari import BackboneDEARI
from ...data.utils import DiceBCELoss




class _DEARI(nn.Module):
    def __init__(
            self,
            n_steps: int,
            n_features: int,
            rnn_hidden_size: int,
            imputation_weight: float,
            consistency_weight: float,
            classification_weight: float,
            n_classes: int,
            num_encoderlayer: int,
            component_error: bool,
            is_gru: bool,
            multi: int,
            model_name: str,
            device = None,
            dropout: float = 0.5,            
            ):

        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.classification_weight = classification_weight
        self.n_classes = n_classes
        self.num_encoderlayer = num_encoderlayer
        self.component_error = component_error
        self.is_gru = is_gru
        self.multi = multi
        self.model_name = model_name
        self.device = device
        self.dropout = dropout


        # create models
        self.model = BackboneDEARI(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi, self.device)
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
        

        if self.model_name == 'Brits_multilayer_attention_without_residual':
            x = inputs['forward']['X']
            m = inputs['forward']['missing_mask']
            imputed_data = (x * m)+ ((1-m) * imputed_data)
        results = {
            "imputed_data": imputed_data,
        }

        f_logits = self.f_classifier(self.dropout(f_hidden_states))
        b_logits = self.b_classifier(self.dropout(b_hidden_states))

        f_predications = torch.sigmoid(f_logits)
        b_predications = torch.sigmoid(b_logits)

        classification_pred = (f_predications + b_predications) / 2

        results['classification_pred'] = classification_pred

        # if in training mode, return results with losses
        if training:
            criterion = DiceBCELoss().to(self.device)
            results["consistency_loss"] = consistency_loss 
            results["reconstruction_loss"] = reconstruction_loss
            f_classification_loss, _ = criterion(f_predications, f_logits, inputs["labels"].unsqueeze(1).float())
            b_classification_loss, _ = criterion(b_predications, b_logits, inputs["labels"].unsqueeze(1).float())
            classification_loss = (f_classification_loss + b_classification_loss)

            loss = (
                self.consistency_weight * consistency_loss +
                self.imputation_weight * reconstruction_loss +
                self.classification_weight * classification_loss
            )

            results["loss"] = loss
            results['classification_loss'] = classification_loss
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results