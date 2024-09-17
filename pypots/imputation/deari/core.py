import torch.nn as nn
from ...nn.modules.deari.backbone import BackboneDEARI



class _DEARI(nn.Module):
    def __init__(self, n_steps,
                    n_features,
                    rnn_hidden_size,
                    num_encoderlayer,
                    component_error,
                    is_gru,
                    multi,
                    consistency_weight,
                    imputation_weight,
                    model_name,
                    device):
            super().__init__()
            self.n_steps = n_steps
            self.n_features = n_features
            self.rnn_hidden_size = rnn_hidden_size
            self.num_encoderlayer = num_encoderlayer
            self.component_error = component_error
            self.is_gru = is_gru
            self.multi = multi
            self.consistency_weight = consistency_weight
            self.imputation_weight = imputation_weight
            self.model_name = model_name
            self.device = device
    
            self.model = BackboneDEARI(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi, self.device)

    def forward(self, inputs:dict, training:bool = True) -> dict:
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

        # if in training mode, return results with losses
        if training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            loss = self.consistency_weight * consistency_loss + self.imputation_weight * reconstruction_loss

            # `loss` is always the item for backward propagating to update the model
            results["loss"] = loss
            # results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction
        if not training:
            results["X_ori"] = inputs["X_ori"]
            results["indicating_mask"] = inputs["indicating_mask"]

        return results
                 