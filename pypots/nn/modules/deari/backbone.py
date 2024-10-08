from .layers import FeatureRegression, Decay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class BackboneRITS_attention(nn.Module):
    """
    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the GRU cell

    num_encoderlayer :
        number of encoder layers for the transformer

    component_error :
        whether to include component estimation error in the loss

    is_gru :
        whether to use GRU or LSTM cell

    device :
        the device (CPU/GPU) to run the model on
    
    attention :
        whether to use attention in the model
    
    hidden_agg :
        how to aggregate the hidden states
    
    temp_decay_h :
        the temporal decay module to decay the hidden state of the GRU
    
    temp_decay_x :
        the temporal decay module to decay data in the raw feature space
    
    hist :
        the temporal-regression module that projects the GRU hidden state into the raw feature space
    
    feat_reg_v :
        the feature-regression module used for feature-based estimation
    
    weight_combine :
        the module that generates the weight to combine history regression and feature regression
    
    class_token :
        the class token used in the transformer
    
    encoder_layer :
        the transformer encoder layer
    
    transformer_encoder :
        the transformer encoder
    
    gru :
        the GRU cell that models temporal data for imputation
    
    lstm :
        the LSTM cell that models temporal data for imputation

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)
    
    n_features :    
        number of features (input dimensions)   

    rnn_hidden_size :
        the hidden size of the GRU cell

    num_encoderlayer :
        number of encoder layers for the transformer

    component_error :
        whether to include component estimation error in the loss

    is_gru :
        whether to use GRU or LSTM cell

    attention :
        whether to use attention in the model   

    hidden_agg :
        how to aggregate the hidden states

    
    
    """
    def __init__(self, n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, attention=True, hidden_agg='cls'):
        super(BackboneRITS_attention, self).__init__()
        self.attention = attention
        self.hidden_agg = hidden_agg
        self.n_steps = n_steps
        self.input_size = n_features
        self.hidden_size = rnn_hidden_size
        self.num_encoderlayer = num_encoderlayer
        self.is_gru = is_gru
        self.component_error = component_error
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        if self.attention == True:
            if self.hidden_size % 4 == 0:
                self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.hidden_size))
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_encoderlayer)
            else:
                print('embed_dim must be divisible by num_heads!')

        if self.is_gru == True:
            self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        else:
            self.lstm = nn.LSTMCell(self.input_size * 2, self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, mask, deltas, last_hiddens=None, h=None, c=None):
        # Get dimensionality
        [B, T, V] = x.shape
        
        if h == None:
            h = Variable(torch.zeros(B, self.hidden_size)).to(x.device)
        else:
            raw_h = h.clone()
            if self.attention:
                h = self.transformer_encoder(h)
            if self.hidden_agg == 'cls':
                raw_h = torch.cat([self.class_token.expand(B, -1, -1), raw_h], dim=1)
                h = self.transformer_encoder(raw_h)[:, 0, :]
            elif self.hidden_agg == 'last':
                h = h[:, -1, :]
            elif self.hidden_agg == 'mean':
                h = torch.mean(h, dim=1)

        if c == None:
            c = Variable(torch.zeros(B, self.hidden_size)).to(x.device)

        x_loss = 0
        x_imp = []
        Hiddens = []
        reconstruction = []
        for t in range(T):
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

            # component estimation error
            if self.component_error == True:
                x_loss += torch.sum(torch.abs(x_t - x_h) * m_t) / (torch.sum(m_t) + 1e-5)
                x_loss += torch.sum(torch.abs(x_t - xu) * m_t) / (torch.sum(m_t) + 1e-5)

            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp_t, m_t], dim=1)

            if self.is_gru == True:
                h = self.gru(input_t, h)
            else:
                h, c = self.lstm(input_t, (h, c))

            # Keep the imputation
            x_imp.append(x_imp_t.unsqueeze(dim=1))
            Hiddens.append(h.unsqueeze(dim=1))
            reconstruction.append(x_comb_t.unsqueeze(dim=1))

        x_imp = torch.cat(x_imp, dim=1)
        Hiddens = torch.cat(Hiddens, dim=1)

        return x_imp, reconstruction, h, x_loss, Hiddens




class BackboneDEARI(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi):
        super(BackboneDEARI, self).__init__()
        self.multi = multi
        self.model_f = nn.ModuleList([BackboneRITS_attention(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru) for i in range(self.multi)])
        self.model_b = nn.ModuleList([BackboneRITS_attention(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru) for i in range(self.multi)])

    def forward(self, xdata):
        # Fetching forward data from xdata
        x = xdata['forward']['X']
        m = xdata['forward']['missing_mask']
        d_f = xdata['forward']['deltas']

        # Fetching backward data from xdata
        m_b = xdata['backward']['missing_mask']
        d_b = xdata['backward']['deltas']

        xreg_loss, loss_consistency, x_imp, Hidden_f, Hidden_b = [], [], [], [], []

        for i in range(self.multi):
            hiddens_f = None if i == 0 else Hidden_f[-1]
            hiddens_b = None if i == 0 else Hidden_b[-1]
            x_f = x if i == 0 else x_imp[-1]

            (
            f_imputed_data,
            f_reconstruction,
            f_last_hidden_states,
            f_reconstruction_loss,
            f_Hiddens,
            ) = self.model_f[i](x=x_f, mask=m, deltas=d_f, h=hiddens_f)
            # Set data to be backward
            x_b = x_f.flip(dims=[1])
            (
            b_imputed_data,
            b_reconstruction,
            b_last_hidden_states,
            b_reconstruction_loss,
            b_Hiddens,
            ) = self.model_b[i](x=x_b, mask=m_b, deltas=d_b, h=hiddens_b)
            
            # Averaging the imputations and prediction
            imp = (f_imputed_data + b_imputed_data.flip(dims=[1])) / 2
            # merge the regression loss
            xreg_loss.append(f_reconstruction_loss + b_reconstruction_loss)
            # average consistency loss
            loss_consistency.append( torch.abs(f_imputed_data - b_imputed_data.flip(dims=[1])).mean() * 1e-1 )
            
            x_imp.append((x * m)+ ((1-m) * imp))
            Hidden_f.append(f_Hiddens)
            Hidden_b.append(b_Hiddens)

        x_imp = torch.stack(x_imp,dim=1)        # [batchsize x number_layer x timesetps x number_feature]
        xreg_loss = torch.stack(xreg_loss,dim=0)        # [number_layer]
        loss_consistency = torch.stack(loss_consistency,dim=0)      # [number_layer]

        imputed_data = torch.mean(x_imp, dim=1)
        reconstruction_loss = torch.mean(xreg_loss, dim=0)
        consistency_loss = torch.mean(loss_consistency, dim=0)

        return (
                imputed_data,
                f_reconstruction,
                b_reconstruction,
                f_last_hidden_states,
                b_last_hidden_states,
                consistency_loss,
                reconstruction_loss,
            )
    