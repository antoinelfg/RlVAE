import os
from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_VAE_MLP
from .vae_config import VAEConfig

from sklearn_extra.cluster import KMedoids


class VAE(BaseAE):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: VAEConfig,
        encoder: Optional[BaseEncoder] = None,
        #decoder: Optional[BaseDecoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior_var: Optional[int] = 1,
        prior_mean= None,
        beta = 1
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "VAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        self.warmup = 0
        self.prior_var = prior_var

        if prior_mean == None:
            self.prior_mean = torch.zeros(self.latent_dim).to(self.device)
        else:
            self.prior_mean = prior_mean
        
        self.beta = beta

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        if hasattr(inputs, "seq_mask"):
            seq_mask = inputs['seq_mask']
        else:
            seq_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
        
        if hasattr(inputs, "pix_mask"):
            pix_mask = inputs['pix_mask']
        else:
            pix_mask = torch.ones_like(x)
            
        epoch = kwargs.pop("epoch", 100)
        #x = x * pix_mask * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]
        loss, recon_loss, kld = self.loss_function(
            recon_x=recon_x,
            #x=x.reshape((x.shape[0]*x.shape[1],) + x.shape[2:]),
            x = x,
            mu=mu,
            log_var=log_var,
            z=z,
            seq_mask=seq_mask,
            pix_mask=pix_mask
        )
        output = ModelOutput(
            reconstruction_loss=recon_loss,
            x=x,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape_as(x),
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, seq_mask=None, pix_mask=None):


        if self.model_config.reconstruction_loss == "mse":
            recon_loss = 0.5 * (
                F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ) * pix_mask.reshape(x.shape[0], -1)
            ).sum(dim=-1)

        diff = mu - self.prior_mean.to(mu.device)
        KLD = -0.5 * torch.sum(1 - torch.log(torch.tensor(self.prior_var).to(mu.device)) + log_var - ((diff.pow(2)  + log_var.exp()) / self.prior_var), dim=-1)
        return (recon_loss + self.beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):

                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))))#.item())
            if i % 1000 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)
    
    def build_metrics(self, mu, log_var, idx=None, addStdNorm = False, T=0.3, lbd=0.0001, verbose = False):
        device = mu.device
        if idx is not None:
            mu = mu[idx]
            log_var = log_var[idx]

        with torch.no_grad():
            self.M_i = torch.diag_embed((-log_var).exp()).detach().to(device)
            self.M_i_flat = (-log_var).exp().detach().to(device)
            self.M_i_inverse_flat = (log_var).exp().detach().to(device)
            self.centroids = mu.detach().to(device)
            self.T = T
            self.lbd = lbd
            
            if addStdNorm:
                if verbose:
                    print('Adding std normal to centroids and var')
                self.centroids = torch.cat([self.centroids, torch.zeros_like(self.centroids[0]).unsqueeze(0)], dim=0)
                self.M_i = torch.cat([self.M_i, torch.eye(self.latent_dim).unsqueeze(0).to(device)], dim=0)
                self.M_i_flat = torch.cat([self.M_i_flat, torch.ones(self.latent_dim).unsqueeze(0).to(device)], dim=0)
                self.M_i_inverse_flat = torch.cat([self.M_i_inverse_flat, torch.ones(self.latent_dim).unsqueeze(0).to(device)], dim=0)

            def G_sampl(z):
                z = z.to(device)
                omega = (
                    -(
                        torch.transpose(
                                    (self.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1), 2, 3) 
                                    @ torch.diag_embed(self.M_i_flat).unsqueeze(0) 
                                    @ (self.centroids.unsqueeze(0) -z.unsqueeze(1)).unsqueeze(-1)
                                ) / self.T**2
                    ).exp()

                return (torch.diag_embed(self.M_i_flat).unsqueeze(0) * omega
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(device)
            
            def G_inv(z):
                return torch.inverse(G_sampl(z))

            self.G_sampl = G_sampl
            self.G_inv = G_inv
            
        #return model


    def retrieveG(self, train_data, num_centroids = 200, T_multiplier = 1, addStdNorm = False, device = 'cpu',  verbose = False):
        loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=False)
        mu = []
        log_var = []
        self.to(device)
        with torch.no_grad():
            for _ , x in enumerate(loader):

                data = x.data

                out = self.encoder(data.to(device))

                mu_data, log_var_data = out.embedding, out.log_covariance

                mu.append(mu_data)
                log_var.append(log_var_data)

        mu = torch.cat(mu)
        log_var = torch.cat(log_var)

        if verbose:
            print('Running Kmedoids')

        warnings.filterwarnings("ignore", category=UserWarning)

        kmedoids = KMedoids(n_clusters=num_centroids).fit(mu.detach().cpu().numpy())
        medoids = torch.tensor(kmedoids.cluster_centers_).to(device)
        centroids_idx = kmedoids.medoid_indices_ #

        if verbose:
            print("Finding temperature")
            
        eps_lf = 0.01
        lbd = 0.01
        T = 0
        T_is = []
        for i in range(len(medoids)-1):
            mask = torch.tensor([k for k in range(len(medoids)) if k != i])
            dist = torch.norm(medoids[i].unsqueeze(0) - medoids[mask], dim=-1)
            T_i =torch.min(dist, dim=0)[0]
            T_is.append(T_i.item())

        T = np.max(T_is)

        if verbose: 
            print('Best temperature found: ', T)
            print('Building metric')
            print('Increasing T by ', T_multiplier)
        T = T * T_multiplier
        self.build_metrics(mu, log_var, centroids_idx, addStdNorm=addStdNorm, T=T, lbd=lbd, verbose=verbose)
        self.centroids_tens = mu

        return self.G_sampl, mu, log_var
    
    def log_pi(self, z):
        return 0.5 * (torch.clamp(self.G_sampl(z).det(), 0, 1e32)).log()
