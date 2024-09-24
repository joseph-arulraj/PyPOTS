import torch.nn as nn
import torch
from ...nn.modules.bayesian_deari.backbone import Bayesian_deari
from ...nn.modules.bayesian_deari.layers import minibatch_weight
class _Bayesian_DEARI(nn.Module):
    def __init__(self, n_steps,
                    n_features,
                    rnn_hidden_size,
                    num_encoderlayer,
                    component_error,
                    is_gru,
                    multi,
                    consistency_weight,
                    imputation_weight,
                    device,
                    number_of_batches,
                    unfreeze_step = 10,
                    sample_nbr = 10
                    ):
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
        self.device = device
        self.unfreeze_step = unfreeze_step
        self.sample_nbr = sample_nbr
        self.number_of_batches = number_of_batches
        self.batch_number = 0
        self.num_update_batch = 0

        self.complexity_cost_weight = minibatch_weight(self.batch_number, self.number_of_batches)
        self.exploitation_weight = self.complexity_cost_weight

        self.model = Bayesian_deari(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi, self.device)


    def forward(self, inputs:dict, training:bool = True) -> dict:
        if training:
            if (self.batch_number % self.unfreeze_step) == 0:
                self.model.update_states('unfreeze')
                kl, kl_loss, xreg_loss, loss_consistency = 0, 0, 0, 0
                x_imp = []
                for _ in range(self.sample_nbr):
                    (
                        imputed_data,
                        f_reconstruction,
                        b_reconstruction,
                        f_hidden_states,
                        b_hidden_states,
                        consistency_loss,
                        reconstruction_loss,
                        loss_KL
                    ) = self.model(inputs)
                    kl += loss_KL
                    kl_loss += loss_KL
                    xreg_loss += reconstruction_loss
                    loss_consistency += consistency_loss
                    x_imp.append(imputed_data.unsqueeze(dim=-1))

                kl /= self.sample_nbr
                kl_loss /= self.sample_nbr
                xreg_loss /= self.sample_nbr
                loss_consistency /= self.sample_nbr
                imputed_data = torch.cat(x_imp, dim=-1).mean(dim=-1)

                self.model.update_states('freeze')
                (
                    freeze_imputed_data,
                    freeze_f_reconstruction,
                    freeze_b_reconstruction,
                    freeze_f_hidden_states,
                    freeze_b_hidden_states,
                    freeze_consistency_loss,
                    freeze_reconstruction_loss,
                    freeze_loss_KL
                ) = self.model(inputs)

                imputed_data = imputed_data * self.exploitation_weight + freeze_imputed_data * (1 - self.exploitation_weight)
                xreg_loss = xreg_loss * self.exploitation_weight + freeze_reconstruction_loss * (1 - self.exploitation_weight)
                loss_consistency = loss_consistency * self.exploitation_weight + freeze_consistency_loss * (1 - self.exploitation_weight)

                self.num_update_batch += 1

            else:
                self.model.update_states('freeze')
                (
                    freeze_imputed_data,
                    freeze_f_reconstruction,
                    freeze_b_reconstruction,
                    freeze_f_hidden_states,
                    freeze_b_hidden_states,
                    freeze_consistency_loss,
                    freeze_reconstruction_loss,
                    freeze_loss_KL
                ) = self.model(inputs)

                imputed_data = freeze_imputed_data
                xreg_loss = freeze_reconstruction_loss
                loss_consistency = freeze_consistency_loss


               

            loss = self.consistency_weight * loss_consistency + self.imputation_weight * xreg_loss + self.complexity_cost_weight * kl_loss
            results = {
                "imputed_data": imputed_data,
                "loss": loss,
            }

            self.batch_number += 1
            
        else:

            self.model.update_states('freeze')
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
            "X_ori": inputs["X_ori"],
            "indicating_mask": inputs["indicating_mask"]
            }

        return results
            





        