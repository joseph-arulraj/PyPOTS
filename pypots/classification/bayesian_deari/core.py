import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules.bayesian_deari.layers import minibatch_weight

from ...nn.modules.bayesian_deari import Bayesian_deari
from ...data.utils import DiceBCELoss




class _Bayesian_DEARI(nn.Module):
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
            number_of_batches,
            device = None,
            dropout: float = 0.5, 
            unfreeze_step: int = 10,
            sample_nbr: int = 10           
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
        self.device = device
        self.dropout = dropout
        self.number_of_batches = number_of_batches
        self.unfreeze_step = unfreeze_step
        self.sample_nbr = sample_nbr
        self.batch_number = 0
        self.num_update_batch = 0


        self.complexity_cost_weight = minibatch_weight(self.batch_number, self.number_of_batches)
        self.exploitation_weight = self.complexity_cost_weight

        self.model = Bayesian_deari(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi, self.device)
        self.f_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.b_classifier = nn.Linear(self.rnn_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        if training:
            if (self.batch_number % self.unfreeze_step) == 0:
                self.model.update_states('unfreeze')
                kl, kl_loss, xreg_loss, loss_consistency = 0, 0, 0, 0
                X_imp, Y_out_f, Y_score_f, Y_out_b, Y_score_b = [], [], [], [], []
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
                    X_imp.append(imputed_data.unsqueeze(dim=-1))
                    f_logits = self.f_classifier(self.dropout(f_hidden_states))
                    b_logits = self.b_classifier(self.dropout(b_hidden_states))
                    f_predictions = torch.sigmoid(f_logits)
                    b_predictions = torch.sigmoid(b_logits)

                    Y_out_f.append(f_logits.unsqueeze(dim=-1))
                    Y_score_f.append(f_predictions.unsqueeze(dim=-1))
                    Y_out_b.append(b_logits.unsqueeze(dim=-1))
                    Y_score_b.append(b_predictions.unsqueeze(dim=-1))

                
                kl /= self.sample_nbr
                kl_loss /= self.sample_nbr
                xreg_loss /= self.sample_nbr
                loss_consistency /= self.sample_nbr
                imputed_data = torch.cat(X_imp, dim=-1).mean(dim=-1)
                F_logits = torch.cat(Y_out_f, dim=-1).mean(dim=-1)
                B_logits = torch.cat(Y_out_b, dim=-1).mean(dim=-1)
                F_predictions = torch.cat(Y_score_f, dim=-1).mean(dim=-1)
                B_predictions = torch.cat(Y_score_b, dim=-1).mean(dim=-1)


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

                f_logits = self.f_classifier(self.dropout(freeze_f_hidden_states))
                b_logits = self.b_classifier(self.dropout(freeze_b_hidden_states))
                f_predictions = torch.sigmoid(f_logits)
                b_predictions = torch.sigmoid(b_logits)

                imputed_data = imputed_data * self.exploitation_weight + freeze_imputed_data * (1 - self.exploitation_weight)
                xreg_loss = xreg_loss * self.exploitation_weight + freeze_reconstruction_loss * (1 - self.exploitation_weight)
                loss_consistency = loss_consistency * self.exploitation_weight + freeze_consistency_loss * (1 - self.exploitation_weight)

                f_logits = F_logits * self.exploitation_weight + f_logits * (1 - self.exploitation_weight)
                b_logits = B_logits * self.exploitation_weight + b_logits * (1 - self.exploitation_weight)
                f_predictions = F_predictions * self.exploitation_weight + f_predictions * (1 - self.exploitation_weight)
                b_predictions = B_predictions * self.exploitation_weight + b_predictions * (1 - self.exploitation_weight)

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
                kl_loss = freeze_loss_KL

                f_logits = self.f_classifier(self.dropout(freeze_f_hidden_states))
                b_logits = self.b_classifier(self.dropout(freeze_b_hidden_states))
                f_predictions = torch.sigmoid(f_logits)
                b_predictions = torch.sigmoid(b_logits)

            # average predictions
            classification_pred = (f_predictions + b_predictions) / 2

            # calculate loss
            criterion = DiceBCELoss().to(self.device)
            f_classification_loss, _ = criterion(f_predictions, f_logits, inputs['labels'].unsqueeze(1).float())
            b_classification_loss, _ = criterion(b_predictions, b_logits, inputs['labels'].unsqueeze(1).float())
            classification_loss = (f_classification_loss + b_classification_loss)

            loss = self.consistency_weight * loss_consistency + self.imputation_weight * xreg_loss + self.complexity_cost_weight * kl_loss + self.classification_weight * classification_loss
            results = {
                'imputed_data': imputed_data,
                'loss': loss,
                'classification_loss': classification_loss,
                'classification_pred': classification_pred
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
            kl_loss
            ) = self.model(inputs)

            f_logits = self.f_classifier(self.dropout(f_hidden_states))
            b_logits = self.b_classifier(self.dropout(b_hidden_states))

            f_predications = torch.sigmoid(f_logits)
            b_predications = torch.sigmoid(b_logits)

            classification_pred = (f_predications + b_predications) / 2

            results = {
                "imputed_data": imputed_data,
                "classification_pred": classification_pred
            }

        return results