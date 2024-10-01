import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
# from torch.nn.utils.rnn import PackedSequence
import math
import numpy as np
import os
import copy
import torch.optim as optim
import pandas as pd
from .bayesiancell import BayesianLSTMcell, BayesianGRUcell
from ..deari.backbone import BackboneDEARI, BackboneRITS_attention
from blitz.utils import variational_estimator

@variational_estimator
class Backbone_Bayesian_rits_attention(BackboneRITS_attention):
    def __init__(self,
                 n_steps,
                 n_features,
                 rnn_hidden_size,
                 num_encoderlayer,
                 component_error,
                 is_gru,
                 device=None,
                 attention=True,
                 hidden_agg='cls'):
        super().__init__(n_steps,
                         n_features,
                         rnn_hidden_size,
                         num_encoderlayer,
                         component_error,
                         is_gru,
                         device,
                         attention,
                         hidden_agg)
        
        # override the LSTM and GRU cells
        if self.is_gru == True:
            self.gru = BayesianGRUcell(self.input_size * 2, self.hidden_size)
        else:
            self.lstm = BayesianLSTMcell(self.input_size * 2, self.hidden_size)

    

class Bayesian_deari(BackboneDEARI):
    def __init__(self, n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi, device = None):
        super(Bayesian_deari, self).__init__(n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, multi, device)
        self.multi = multi
        self.device = device
        self.state = None
        params = (n_steps, n_features, rnn_hidden_size, num_encoderlayer, component_error, is_gru, device)
        self.model_f = nn.ModuleList([Backbone_Bayesian_rits_attention(*params) for i in range(self.multi)])
        self.model_b = nn.ModuleList([Backbone_Bayesian_rits_attention(*params) for i in range(self.multi)])

    def update_states(self, new_state=None):
        if new_state is not None:
            self.state = new_state
        
        if self.state == 'unfreeze':
            [self.model_f[i].unfreeze_() for i in range(self.multi)]
            [self.model_b[i].unfreeze_() for i in range(self.multi)]
            print('Model state change to unfreeze!')
        elif self.state == 'freeze':
            [self.model_f[i].freeze_() for i in range(self.multi)]
            [self.model_b[i].freeze_() for i in range(self.multi)]

    def forward(self, xdata):
        
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_last_hidden_states,
            b_last_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = super().forward(xdata)

        
        kl_loss = 0
        if self.state == 'unfreeze':
            for i in range(self.multi):
                kl_loss += self.model_f[i].nn_kl_divergence() + self.model_b[i].nn_kl_divergence()

        return (
                imputed_data,
                f_reconstruction,
                b_reconstruction,
                f_last_hidden_states,
                b_last_hidden_states,
                consistency_loss,
                reconstruction_loss,
                kl_loss
            )
    