import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import TrainableRandomDistribution, PriorWeightDistribution
import math
import numpy as np

# https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/master/blitz/modules/gru_bayesian_layer.py
class BayesianRNN(BayesianModule):
    """
    implements base class for B-RNN to enable posterior sharpening
    """
    def __init__(self,
                 sharpen=False):
        super().__init__()
        
        self.weight_ih_mu = None
        self.weight_hh_mu = None
        self.bias_ih_mu = None
        self.bias_hh_mu = None
        
        self.weight_ih_sampler = None
        self.weight_hh_sampler = None
        self.bias_ih_sampler = None
        self.bias_hh_sampler = None

        self.weight_ih = None
        self.weight_hh = None
        self.bias_ih = None
        self.bias_hh = None

        self.sharpen = sharpen
        
        self.weight_ih_eta = None
        self.weight_hh_eta = None
        self.bias_ih_eta = None
        self.bias_hh_eta = None
        self.ff_parameters = None
        self.loss_to_sharpen = None
        
    
    def sample_weights(self):
        pass
    
    def init_sharpen_parameters(self):
        if self.sharpen:
            self.weight_ih_eta = nn.Parameter(torch.Tensor(self.weight_ih_mu.size()))
            self.weight_hh_eta = nn.Parameter(torch.Tensor(self.weight_hh_mu.size()))
            self.bias_ih_eta = nn.Parameter(torch.Tensor(self.bias_ih_mu.size()))
            self.bias_hh_eta = nn.Parameter(torch.Tensor(self.bias_hh_mu.size()))
            
            self.ff_parameters = []

            self.init_eta()
    
    def init_eta(self):
        stdv = 1.0 / math.sqrt(self.weight_hh_eta.shape[0]) #correspond to hidden_units parameter
        self.weight_ih_eta.data.uniform_(-stdv, stdv)
        self.weight_hh_eta.data.uniform_(-stdv, stdv)
        self.bias_ih_eta.data.uniform_(-stdv, stdv)
        self.bias_hh_eta.data.uniform_(-stdv, stdv)
    
    def set_loss_to_sharpen(self, loss):
        self.loss_to_sharpen = loss
    
    def sharpen_posterior(self, loss, input_shape):
        """
        sharpens the posterior distribution by using the algorithm proposed in
        @article{DBLP:journals/corr/FortunatoBV17,
          author    = {Meire Fortunato and
                       Charles Blundell and
                       Oriol Vinyals},
          title     = {Bayesian Recurrent Neural Networks},
          journal   = {CoRR},
          volume    = {abs/1704.02798},
          year      = {2017},
          url       = {http://arxiv.org/abs/1704.02798},
          archivePrefix = {arXiv},
          eprint    = {1704.02798},
          timestamp = {Mon, 13 Aug 2018 16:48:21 +0200},
          biburl    = {https://dblp.org/rec/journals/corr/FortunatoBV17.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }
        """
        bs, seq_len, in_size = input_shape
        gradients = torch.autograd.grad(outputs=loss,
                                        inputs=self.ff_parameters,
                                        grad_outputs=torch.ones(loss.size()).to(loss.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)
        
        grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh = gradients
        
        #to generate sigmas on the weight sampler
        _ = self.sample_weights()
        
        weight_ih_sharpened = self.weight_ih_mu - self.weight_ih_eta * grad_weight_ih + self.weight_ih_sampler.sigma
        weight_hh_sharpened = self.weight_hh_mu - self.weight_hh_eta * grad_weight_hh + self.weight_hh_sampler.sigma
        bias_ih_sharpened = self.bias_ih_mu - self.bias_ih_eta * grad_bias_ih + self.bias_ih_sampler.sigma
        bias_hh_sharpened = self.bias_hh_mu - self.bias_hh_eta * grad_bias_hh + self.bias_hh_sampler.sigma
        
        if self.bias_ih is not None:
            b_ih_log_posterior = self.bias_ih_sampler.log_posterior(w=bias_ih_sharpened)
            b_ih_log_prior_ = self.bias_ih_prior_dist.log_prior(bias_ih_sharpened)
            
        else:
            b_ih_log_posterior = b_ih_log_prior = 0
        
        if self.bias_hh is not None:
            b_hh_log_posterior = self.bias_hh_sampler.log_posterior(w=bias_hh_sharpened)
            b_hh_log_prior_ = self.bias_hh_prior_dist.log_prior(bias_hh_sharpened)
            
        else:
            b_hh_log_posterior = b_hh_log_prior = 0

        self.log_variational_posterior += (self.weight_ih_sampler.log_posterior(w=weight_ih_sharpened) + b_ih_log_posterior + self.weight_hh_sampler.log_posterior(w=weight_hh_sharpened) + b_hh_log_posterior) / seq_len
        
        self.log_prior += (self.weight_ih_prior_dist.log_prior(weight_ih_sharpened) + b_ih_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh_sharpened) + b_hh_log_prior) / seq_len
        

        return weight_ih_sharpened, weight_hh_sharpened, bias_ih_sharpened, bias_hh_sharpened



class BayesianLSTMcell(BayesianRNN):
    """
    Bayesian LSTM layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).
    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None,
                #  peephole = False,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze
        # self.peephole = peephole
        
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        
        # Variational weight parameters and sample for weight ih
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_ih_sampler = TrainableRandomDistribution(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        
        # Variational weight parameters and sample for weight hh
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_hh_sampler = TrainableRandomDistribution(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        
        # Variational weight parameters and sample for bias ih
        self.bias_ih_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        self.bias_ih_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_ih_sampler = TrainableRandomDistribution(self.bias_ih_mu, self.bias_ih_rho)
        self.bias_ih=None

        # Variational weight parameters and sample for bias hh
        self.bias_hh_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        self.bias_hh_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_hh_sampler = TrainableRandomDistribution(self.bias_hh_mu, self.bias_hh_rho)
        self.bias_hh=None

        #our prior distributions
        self.weight_ih_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.weight_hh_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_ih_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_hh_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)

        
        self.init_sharpen_parameters()
        
        self.log_prior = 0
        self.log_variational_posterior = 0
    
    
    def sample_weights(self):
        #sample weights
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        
        #if use bias, we sample it, otherwise, we are using zeros
        if self.use_bias:
            b_ih = self.bias_ih_sampler.sample()
            b_ih_log_posterior = self.bias_ih_sampler.log_posterior()
            b_ih_log_prior = self.bias_ih_prior_dist.log_prior(b_ih)

            b_hh = self.bias_hh_sampler.sample()
            b_hh_log_posterior = self.bias_hh_sampler.log_posterior()
            b_hh_log_prior = self.bias_hh_prior_dist.log_prior(b_hh)
        else:
            b_ih = None
            b_ih_log_posterior = 0
            b_ih_log_prior = 0

            b_hh = None
            b_hh_log_posterior = 0
            b_hh_log_prior = 0

        bias_ih = b_ih
        bias_hh = b_hh
        
        #gather weights variational posterior and prior likelihoods
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_hh_log_posterior + self.weight_ih_sampler.log_posterior() + b_ih_log_posterior
        
        self.log_prior = self.weight_ih_prior_dist.log_prior(weight_ih) + b_ih_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh) + b_hh_log_prior
        
        
        self.ff_parameters = [weight_ih, weight_hh, bias_ih, bias_hh]
        
        return weight_ih, weight_hh, bias_ih, bias_hh
        
    def get_frozen_weights(self):
        
        #get all deterministic weights
        weight_ih = self.weight_ih_mu
        weight_hh = self.weight_hh_mu
        if self.use_bias:
            bias_ih = self.bias_ih_mu
            bias_hh = self.bias_hh_mu
        else:
            bias_ih = 0
            bias_hh = 0

        return weight_ih, weight_hh, bias_ih, bias_hh

    
    def forward_(self,
                 x,
                 hidden_states,
                 sharpen_loss):
        
        if self.loss_to_sharpen is not None:
            sharpen_loss = self.loss_to_sharpen
            weight_ih, weight_hh, bias_ih, bias_hh = self.sharpen_posterior(loss=sharpen_loss, input_shape=x.shape)

        elif (sharpen_loss is not None):
            sharpen_loss = sharpen_loss
            weight_ih, weight_hh, bias_ih, bias_hh = self.sharpen_posterior(loss=sharpen_loss, input_shape=x.shape)
        
        else:
            weight_ih, weight_hh, bias_ih, bias_hh = self.sample_weights()

        #Assumes x is of shape (batch, sequence, feature)
        bs, _ = x.size()
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            hx, cx = (torch.zeros(bs, self.out_features).to(x.device), 
                        torch.zeros(bs, self.out_features).to(x.device))
        else:
            hx, cx = hidden_states
        
        #simplifying our out features, and hidden seq list
        gates = torch.mm(x, weight_ih) + bias_ih + torch.mm(hx, weight_hh) + bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        
        return hy, cy

    def forward_frozen(self,
                       x,
                       hidden_states):

        weight_ih, weight_hh, bias_ih, bias_hh = self.get_frozen_weights()

        #Assumes x is of shape (batch, sequence, feature)
        bs, _ = x.size()
        # hidden_seq = []
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            hx, cx = (torch.zeros(bs, self.out_features).to(x.device), 
                        torch.zeros(bs, self.out_features).to(x.device))
        else:
            hx, cx = hidden_states

        #simplifying our out features, and hidden seq list
        # batch the computations into a single matrix multiplication
        
        gates = torch.mm(x, weight_ih) + bias_ih + torch.mm(hx, weight_hh) + bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)

        
        return hy, cy

    def forward(self,
                x,
                hidden_states=None,
                sharpen_loss=None):

        if self.freeze:
            return self.forward_frozen(x, hidden_states)
        
        if not self.sharpen:
            sharpen_posterior = False
            
        return self.forward_(x, hidden_states, sharpen_loss)



class BayesianGRUcell(BayesianRNN):
    """
    Bayesian GRU layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).
    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze
        
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        
        # Variational weight parameters and sample for weight ih
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 3).normal_(posterior_mu_init, 0.1))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 3).normal_(posterior_rho_init, 0.1))
        self.weight_ih_sampler = TrainableRandomDistribution(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        
        # Variational weight parameters and sample for weight hh
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 3).normal_(posterior_mu_init, 0.1))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 3).normal_(posterior_rho_init, 0.1))
        self.weight_hh_sampler = TrainableRandomDistribution(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        
        # Variational weight parameters and sample for bias
        self.bias_ih_mu = nn.Parameter(torch.Tensor(out_features * 3).normal_(posterior_mu_init, 0.1))
        self.bias_ih_rho = nn.Parameter(torch.Tensor(out_features * 3).normal_(posterior_rho_init, 0.1))
        self.bias_ih_sampler = TrainableRandomDistribution(self.bias_ih_mu, self.bias_ih_rho)
        self.bias_ih=None

        # Variational weight parameters and sample for bias
        self.bias_hh_mu = nn.Parameter(torch.Tensor(out_features * 3).normal_(posterior_mu_init, 0.1))
        self.bias_hh_rho = nn.Parameter(torch.Tensor(out_features * 3).normal_(posterior_rho_init, 0.1))
        self.bias_hh_sampler = TrainableRandomDistribution(self.bias_hh_mu, self.bias_hh_rho)
        self.bias_hh=None

        #our prior distributions
        self.weight_ih_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.weight_hh_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_ih_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_hh_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        
        self.init_sharpen_parameters()
        
        self.log_prior = 0
        self.log_variational_posterior = 0
    
    def sample_weights(self):
        #sample weights
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        
        #if use bias, we sample it, otherwise, we are using zeros
        if self.use_bias:
            b_ih = self.bias_ih_sampler.sample()
            b_ih_log_posterior = self.bias_ih_sampler.log_posterior()
            b_ih_log_prior = self.bias_ih_prior_dist.log_prior(b_ih)

            b_hh = self.bias_hh_sampler.sample()
            b_hh_log_posterior = self.bias_hh_sampler.log_posterior()
            b_hh_log_prior = self.bias_hh_prior_dist.log_prior(b_hh)

        else:
            b_ih = None
            b_ih_log_posterior = 0
            b_ih_log_prior = 0

            b_hh = None
            b_hh_log_posterior = 0
            b_hh_log_prior = 0

        bias_ih = b_ih
        bias_hh = b_hh

        #gather weights variational posterior and prior likelihoods
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_hh_log_posterior + self.weight_ih_sampler.log_posterior() + b_ih_log_posterior
        
        self.log_prior = self.weight_ih_prior_dist.log_prior(weight_ih) + b_ih_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh) + b_hh_log_prior
        
        self.ff_parameters = [weight_ih, weight_hh, bias_ih, bias_hh]
        return weight_ih, weight_hh, bias_ih, bias_hh

       
    def get_frozen_weights(self):
        
        #get all deterministic weights
        weight_ih = self.weight_ih_mu
        weight_hh = self.weight_hh_mu
        if self.use_bias:
            bias_ih = self.bias_ih_mu
            bias_hh = self.bias_hh_mu
        else:
            bias_ih = 0
            bias_hh = 0

        return weight_ih, weight_hh, bias_ih, bias_hh

    
    def forward_(self,
                 x,
                 hidden_states,
                 sharpen_loss):
        
        if self.loss_to_sharpen is not None:
            sharpen_loss = self.loss_to_sharpen
            weight_ih, weight_hh, bias_ih, bias_hh = self.sharpen_posterior(loss=sharpen_loss, input_shape=x.shape)
        
        elif (sharpen_loss is not None):
            sharpen_loss = sharpen_loss
            weight_ih, weight_hh, bias_ih, bias_hh = self.sharpen_posterior(loss=sharpen_loss, input_shape=x.shape)
        
        else:
            weight_ih, weight_hh, bias_ih, bias_hh = self.sample_weights()

        #Assumes x is of shape (batch, sequence, feature)
        bs, _ = x.size()
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            hx = torch.zeros(bs, self.out_features).to(x.device)
        else:
            hx = hidden_states
        
        # batch the computations into a single matrix multiplication
        gi = torch.mm(x, weight_ih) + bias_ih
        gh = torch.mm(hx, weight_hh) + bias_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        hy = newgate + inputgate * (hx - newgate)

        return hy

    def forward_frozen(self,
                       x,
                       hidden_states):

        weight_ih, weight_hh, bias_ih, bias_hh = self.get_frozen_weights()

        #Assumes x is of shape (batch, sequence, feature)
        bs, _ = x.size()
        
        #if no hidden state, we are using zeros
        if hidden_states is None:
            hx = torch.zeros(bs, self.out_features).to(x.device)
        else:
            hx = hidden_states
        
        gi = torch.mm(x, weight_ih) + bias_ih
        gh = torch.mm(hx, weight_hh) + bias_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        hy = newgate + inputgate * (hx - newgate)
        
        return hy         

    def forward(self,
                x,
                hidden_states=None,
                sharpen_loss=None):

        if self.freeze:
            return self.forward_frozen(x, hidden_states)
        
        if not self.sharpen:
            sharpen_loss = None
            
        return self.forward_(x, hidden_states, sharpen_loss)