import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import FeatureRegression, Decay, Decay_obs, PositionalEncoding, Conv1dWithInit, TorchTransformerEncoder


class BackboneCSAI(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size, step_channels, medians_df=None, device=None):
        super(BackboneCSAI, self).__init__()


        if medians_df is not None:
            self.medians_tensor = torch.tensor(list(medians_df.values())).float()
        else:
            self.medians_tensor = None

        self.n_steps = n_steps
        self.step_channels = step_channels
        self.input_size = n_features
        self.hidden_size = rnn_hidden_size
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.weighted_obs = Decay_obs(self.input_size, self.input_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        
        self.pos_encoder = PositionalEncoding(self.step_channels)
        self.input_projection = Conv1dWithInit(self.input_size, self.step_channels, 1)
        self.output_projection1 = Conv1dWithInit(self.step_channels, self.hidden_size, 1)
        self.output_projection2 = Conv1dWithInit(self.n_steps*2, 1, 1)
        self.time_layer = TorchTransformerEncoder(channels=self.step_channels)
        self.device = device

        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, mask, deltas, last_obs, h=None):

        # Get dimensionality
        [B, _, _] = x.shape

        if self.medians_tensor is not None:
            medians_t = self.medians_tensor.unsqueeze(0).repeat(B, 1).to(self.device)

        decay_factor = self.weighted_obs(deltas - medians_t.unsqueeze(1))

        if h == None:
            data_last_obs = self.input_projection(last_obs.permute(0, 2, 1)).permute(0, 2, 1)
            data_decay_factor = self.input_projection(decay_factor.permute(0, 2, 1)).permute(0, 2, 1)

            data_last_obs = self.pos_encoder(data_last_obs.permute(1, 0, 2)).permute(1, 0, 2)
            data_decay_factor = self.pos_encoder(data_decay_factor.permute(1, 0, 2)).permute(1, 0, 2)
            
            data = torch.cat([data_last_obs, data_decay_factor], dim=1)

            data = self.time_layer(data)
            data = self.output_projection1(data.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.output_projection2(data).squeeze()

        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        reconstruction = []
        for t in range(self.n_steps):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h
            
            # history based estimation
            x_h = self.hist(h)        
            
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # feature based estimation
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)
            
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)

            h = self.gru(input_t, h)
            Hiddens.append(h.unsqueeze(dim=1))
            reconstruction.append(x_comb_t.unsqueeze(dim=1))
        Hiddens = torch.cat(Hiddens, dim=1)

        return x_imp, reconstruction, h, x_loss


class BackboneBCSAI(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size, step_channels, medians_df=None, device=None):
        super(BackboneBCSAI, self).__init__()

        self.model_f = BackboneCSAI(n_steps, n_features, rnn_hidden_size, step_channels, medians_df, device)
        self.model_b = BackboneCSAI(n_steps, n_features, rnn_hidden_size, step_channels, medians_df, device)
        

    def forward(self, xdata):

        # Fetching forward data from xdata
        x = xdata['forward']['X']
        m = xdata['forward']['missing_mask']
        d_f = xdata['forward']['deltas']
        last_obs_f = xdata['forward']['last_obs']

        # Fetching backward data from xdata
        x_b = xdata['backward']['X']
        m_b = xdata['backward']['missing_mask']
        d_b = xdata['backward']['deltas']
        last_obs_b = xdata['backward']['last_obs']

        # Call forward model
        (
            f_imputed_data,
            f_reconstruction,
            f_hidden_states,
            f_reconstruction_loss,
        ) = self.model_f(x, m, d_f, last_obs_f)

        # Call backward model
        (
            b_imputed_data,
            b_reconstruction,
            b_hidden_states,
            b_reconstruction_loss,
        ) = self.model_b(x_b, m_b, d_b, last_obs_b)

        # Averaging the imputations and prediction
        x_imp = (f_imputed_data + b_imputed_data.flip(dims=[1])) / 2
        imputed_data = (x * m)+ ((1-m) * x_imp)

        # average consistency loss
        consistency_loss = torch.abs(f_imputed_data - b_imputed_data.flip(dims=[1])).mean() * 1e-1

        # Merge the regression loss
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss
        return (
                imputed_data,
                f_reconstruction,
                b_reconstruction,
                f_hidden_states,
                b_hidden_states,
                consistency_loss,
                reconstruction_loss,
            )
