# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 20:00:23 2026

@author: xwyma
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def ck(name, t):
    if t is None: return
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"[BAD] {name}: nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} ")

    
# Define a custom dataset for image and gene expression pairs
class ETM_Dataset(Dataset):
    def __init__(self, X_data, batch_info):        
        self.X_data = X_data
        self.batch_info = batch_info

    def __len__(self):
        return len(self.X_data)

    def __transform_data(self, data):        
        data = torch.tensor(data, dtype=torch.float32)
        return data
        
    def __getitem__(self, idx):
        data = self.X_data[idx]
        data = self.__transform_data(data)
        batch_idx = self.batch_info[idx]
        return data, batch_idx
    
    
#%%
class ETM_MutiModal_Dataset(Dataset):
    def __init__(self, X_data_1, X_data_2, batch_info):
        
        self.X_data_1 = X_data_1
        self.X_data_2 = X_data_2
        self.batch_info = batch_info

    def __len__(self):
        return len(self.X_data_1)

    def __transform(self, X_data):        
        X_data = torch.tensor(X_data, dtype=torch.float32)
        return X_data
        
    def __getitem__(self, idx):
        data_1 = self.X_data_1[idx]
        data_1 = self.__transform(data_1)
        data_2 = self.X_data_2[idx]
        data_2 = self.__transform(data_2)
        batch_idx = self.batch_info[idx]
        return data_1, data_2, batch_idx
    
#%%
class Encoder_Branch(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, dropout=0.05, norm_type="batch"):
        super(Encoder_Branch, self).__init__()

        # choose normalization
        if norm_type == "batch":
            Norm = nn.BatchNorm1d
        elif norm_type == "layer":
            Norm = nn.LayerNorm
        else:
            raise ValueError("norm_type must be 'batch' or 'layer'")

        self.encode = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            Norm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            Norm(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encode(x)
#%%
class ETM_v(nn.Module):
    # vanilla ETM
    def __init__(self, num_topics,
                 vocab_size, 
                 t_hidden_size=256, 
                 rho_size=256, 
                 enc_drop=0.05):
        super(ETM_v, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.encoder = Encoder_Branch(vocab_size, t_hidden_size, t_hidden_size, enc_drop)

        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logvar_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        ## Ensemble of word embedding matrices \rho
        self.rho = nn.Parameter(torch.randn(vocab_size, rho_size)) 

        ## Ensemble of topic embedding matrices
        self.alphas = nn.Linear(rho_size, num_topics, bias=False) 


    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    
    def encode(self, normalized_bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.encoder(normalized_bows)
        mu_theta = self.mu_q_theta(q_theta)
        logvar_theta = self.logvar_q_theta(q_theta)

        return mu_theta, logvar_theta
    

    def get_beta(self):

        beta = self.alphas(self.rho) #[V,D]x[D,K]=[V,K]

        beta = F.softmax(beta, dim=0).transpose(1, 0) # after transpose: [K,V]

        return beta


    def decode(self, mu, logvar):
        # get theta
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z , dim=-1)
        
        # get beta
        beta = self.get_beta()
        # p = theta @ beta
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)

        return preds

    def forward(self, bows, normalized_bows):
        # encode
        mu, logvar = self.encode(normalized_bows)
        
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # get prediction loss
        preds = self.decode(mu, logvar)

        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()

        return recon_loss, kld_loss
    
#%%
class InSTaPath(nn.Module):
    def __init__(self, num_topics, n_batches,
                 vocab_size_gene, vocab_size_img, 
                 t_hidden_size_gene=256, t_hidden_size_img=64,
                 rho_size=256, 
                 enc_drop_gene=0.05, enc_drop_img=0.05,enc_norm = "batch",
                 normalize_beta=False,
                 enable_gene_batch_bias=True, enable_img_batch_bias=True,
                 enable_gene_global_bias=True, enable_img_global_bias=True):
        super(InSTaPath, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size_gene = vocab_size_gene
        self.vocab_size_img = vocab_size_img
        self.t_hidden_size_gene = t_hidden_size_gene
        self.t_hidden_size_img = t_hidden_size_img
        self.rho_size = rho_size


        ## define variational distribution for \theta_{1:D} via amortizartion
        self.gene_encoder = Encoder_Branch(vocab_size_gene, t_hidden_size_gene, t_hidden_size_gene, enc_drop_gene, enc_norm) 
        self.img_encoder = Encoder_Branch(vocab_size_img, t_hidden_size_img, t_hidden_size_img, enc_drop_img, enc_norm) 

        self.mu_q_theta_gene = nn.Linear(t_hidden_size_gene, num_topics, bias=True)
        self.mu_q_theta_img = nn.Linear(t_hidden_size_img, num_topics, bias=True)

        self.logvar_q_theta_gene = nn.Linear(t_hidden_size_gene, num_topics, bias=True)
        self.logvar_q_theta_img = nn.Linear(t_hidden_size_img, num_topics, bias=True)

        ## Ensemble of word embedding matrices \rho
        self.rho_gene = nn.Parameter(torch.randn(vocab_size_gene, rho_size))
        self.rho_img = nn.Parameter(torch.randn(vocab_size_img, rho_size))

        ## Ensemble of topic embedding matrices
        self.alphas = nn.Linear(rho_size, num_topics, bias=False) 

        ## Batch correction
        self.n_batches = n_batches
        self.normalize_beta = normalize_beta
        self.enable_gene_batch_bias = enable_gene_batch_bias
        self.enable_img_batch_bias = enable_img_batch_bias
        self.enable_gene_global_bias = enable_gene_global_bias
        self.enable_img_global_bias = enable_img_global_bias
        self._init_batch_and_global_biases()

    def _init_batch_and_global_biases(self) -> None:
        """Initializes batch and global biases given the constant attributes."""
        if not self.normalize_beta:
            if self.enable_gene_batch_bias:
                self.gene_batch_bias = nn.Parameter(torch.randn(self.n_batches, self.vocab_size_gene))
            
            if self.enable_img_batch_bias:
                self.img_batch_bias = nn.Parameter(torch.randn(self.n_batches, self.vocab_size_img))

            if self.enable_gene_global_bias:
                self.gene_global_bias = nn.Parameter(torch.randn(1, self.vocab_size_gene))

            if self.enable_img_global_bias:
                self.img_global_bias = nn.Parameter(torch.randn(1, self.vocab_size_img))

    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def experts(self, mu, logvar, eps=1e-8):
        # precision of i-th Gaussian expert at point x
        T = 1. / (torch.exp(logvar) + eps) # (3, 2000, 100)
        sum_T = torch.sum(T, dim=0)
        pd_mu = torch.sum(mu * T, dim=0) / sum_T # (2000,100)
        pd_var = 1. / sum_T
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

    def encode(self, normalized_bows_gene, normalized_bows_img):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta_gene = self.gene_encoder(normalized_bows_gene)

        mu_theta_gene = self.mu_q_theta_gene(q_theta_gene)
        logvar_theta_gene = self.logvar_q_theta_gene(q_theta_gene)
        # logvar_theta_gene = logvar_theta_gene.clamp(-100,100)

        q_theta_img = self.img_encoder(normalized_bows_img)

        mu_theta_img = self.mu_q_theta_img(q_theta_img)
        logvar_theta_img = self.logvar_q_theta_img(q_theta_img)
        # logvar_theta_img = logvar_theta_img.clamp(-100,100)
        ck("q_theta_gene", q_theta_gene)
        ck("q_theta_img", q_theta_img)

        return mu_theta_gene, logvar_theta_gene, mu_theta_img, logvar_theta_img
    
    def product_of_Gaussian(self, mu_theta_gene, logvar_theta_gene, mu_theta_img, logvar_theta_img):

        B, K = mu_theta_img.shape # get batch and topic_number
        device = mu_theta_gene.device
        # mu_prior = torch.zeros((1, B, K)).to(device)
        # logvar_prior = torch.zeros((1, B, K)).to(device) # log 1 = 0
        # Mu = torch.cat((mu_prior, mu_theta_gene.unsqueeze(0), mu_theta_img.unsqueeze(0)), dim=0)
        # Logvar = torch.cat((logvar_prior, logvar_theta_gene.unsqueeze(0), logvar_theta_img.unsqueeze(0)), dim=0)
        
        Mu = torch.cat((mu_theta_gene.unsqueeze(0), mu_theta_img.unsqueeze(0)), dim=0)
        Logvar = torch.cat((logvar_theta_gene.unsqueeze(0), logvar_theta_img.unsqueeze(0)), dim=0)

        mu, logvar = self.experts(Mu, Logvar)
        return mu, logvar


    def get_beta(self):
        ## softmax over vocab dimension
        beta_gene = F.softmax(self.alphas(self.rho_gene), dim=0).transpose(1, 0)
        beta_img = F.softmax(self.alphas(self.rho_img), dim=0).transpose(1, 0)

        return beta_gene, beta_img


    def decode(self, mu, logvar, batch_indices):
        # get theta
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z , dim=-1)
        
        if self.normalize_beta:
            # get beta
            beta_gene, beta_img = self.get_beta()
            # p = theta @ beta
            res_gene = torch.mm(theta, beta_gene)
            preds_gene = torch.log(res_gene + 1e-6)
            res_img = torch.mm(theta, beta_img)
            preds_img = torch.log(res_img + 1e-6)
        else:
            # get beta
            beta_gene = self.alphas(self.rho_gene).transpose(1, 0) #[V,D]x[D,K]=[V,K], after transpose: [K,V]
            recon_logit_gene = torch.mm(theta, beta_gene)  # [batch_size, n_genes]
            if self.enable_gene_global_bias:
                recon_logit_gene += self.gene_global_bias
            if self.enable_gene_batch_bias:
                recon_logit_gene += self.gene_batch_bias[batch_indices]
            preds_gene = F.log_softmax(recon_logit_gene, dim=-1)
            
            beta_img = self.alphas(self.rho_img).transpose(1, 0)
            recon_logit_img = torch.mm(theta, beta_img)  # [batch_size, n_genes]
            if self.enable_img_global_bias:
                recon_logit_img += self.img_global_bias
            if self.enable_img_batch_bias:
                recon_logit_img += self.img_batch_bias[batch_indices]
            preds_img = F.log_softmax(recon_logit_img, dim=-1)

        return preds_gene, preds_img

    def forward(self, bows_gene, bows_img, normalized_bows_gene, normalized_bows_img, batch_indices):
        # encode
        mu_theta_gene, logvar_theta_gene, mu_theta_img, logvar_theta_img = self.encode(normalized_bows_gene, normalized_bows_img)
        mu, logvar = self.product_of_Gaussian(mu_theta_gene, logvar_theta_gene, mu_theta_img, logvar_theta_img)
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # get prediction loss
        preds_gene, preds_img = self.decode(mu, logvar, batch_indices)

        recon_loss_gene = -(preds_gene * bows_gene).sum(1)
        recon_loss_gene = recon_loss_gene.mean()

        recon_loss_img = -(preds_img * bows_img).sum(1)
        recon_loss_img = recon_loss_img.mean()

        return recon_loss_gene, recon_loss_img, kld_loss
    

class ETM(nn.Module):
    def __init__(self, num_topics, n_batches,
                 vocab_size, 
                 t_hidden_size=256, 
                 rho_size=256, 
                 enc_drop=0.05,
                 normalize_beta=False,
                 enable_batch_bias=True,
                 enable_global_bias=True):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.encoder = Encoder_Branch(vocab_size, t_hidden_size, t_hidden_size, enc_drop)

        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logvar_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        ## Ensemble of word embedding matrices \rho
        self.rho = nn.Parameter(torch.randn(vocab_size, rho_size)) 

        ## Ensemble of topic embedding matrices
        self.alphas = nn.Linear(rho_size, num_topics, bias=False) 

        ## Batch correction
        self.n_batches = n_batches
        self.normalize_beta = normalize_beta
        self.enable_batch_bias = enable_batch_bias
        self.enable_global_bias = enable_global_bias
        self._init_batch_and_global_biases()

    def _init_batch_and_global_biases(self) -> None:
        """Initializes batch and global biases given the constant attributes."""
        if not self.normalize_beta:
            if self.enable_batch_bias:
                self.batch_bias = nn.Parameter(torch.randn(self.n_batches, self.vocab_size))

            if self.enable_global_bias:
                self.global_bias = nn.Parameter(torch.randn(1, self.vocab_size))

    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    
    def encode(self, normalized_bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.encoder(normalized_bows)
        mu_theta = self.mu_q_theta(q_theta)
        logvar_theta = self.logvar_q_theta(q_theta)

        return mu_theta, logvar_theta
    

    def get_beta(self):

        beta = self.alphas(self.rho) #[V,D]x[D,K]=[V,K]
        beta = F.softmax(beta, dim=0).transpose(1, 0) # after transpose: [K,V]
        return beta


    def decode(self, mu, logvar, batch_indices):
        # get theta
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z , dim=-1)

        if self.normalize_beta:
            # get beta
            beta = self.get_beta()
            # p = theta @ beta
            res = torch.mm(theta, beta)
            preds = torch.log(res + 1e-6)
        else:
            # get beta
            beta = self.alphas(self.rho).transpose(1, 0) #[V,D]x[D,K]=[V,K], after transpose: [K,V]
            recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]
            if self.enable_global_bias:
                recon_logit += self.global_bias
            if self.enable_batch_bias:
                recon_logit += self.batch_bias[batch_indices]
            preds = F.log_softmax(recon_logit, dim=-1)

        return preds
    

    def forward(self, bows, normalized_bows, batch_indices):
        # encode
        mu, logvar = self.encode(normalized_bows)
        
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # get prediction loss
        preds = self.decode(mu, logvar, batch_indices)

        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()

        return recon_loss, kld_loss