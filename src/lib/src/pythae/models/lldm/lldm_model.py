import os
from typing import Optional, Union

import sys
from tqdm import tqdm
sys.path.append("......")
sys.path.append(".....")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..normalizing_flows import IAF, IAFConfig
from ..vae import VAE, VAEConfig
from ..lvae_iaf.lvae_iaf_config import LVAE_IAF_Config
from diffusion.stable_diffusion.latent_diffusion import MyLatentDiffusion
from diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from geometric_perspective_on_vaes.sampling import hmc_sampling

class LLDM(VAE):
    """
    """

    def __init__(
        self,
        model_config: LVAE_IAF_Config,
        encoder: Optional[BaseEncoder],
        decoder: Optional[BaseDecoder] ,
        pretrained_vae: Optional[VAE], #do not forget to call retrieveG on the vae beforehand !
        pretrained_ldm: Optional[MyLatentDiffusion],
        ddim_sampler: Optional[DDIMSampler],
        precomputed_zT_samples = None,
        GM = None,
        temp: Optional[float] = 1.,
        verbose = False
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "LLDM_IAF"

        self.input_dim = model_config.input_dim

        self.n_obs = model_config.n_obs_per_ind
        self.warmup = model_config.warmup
        self.context_dim = model_config.context_dim
        self.beta = model_config.beta

        self.flows = nn.ModuleList()

        self.linear_scheduling = self.model_config.linear_scheduling_steps

        self.prior = model_config.prior
        self.posterior = model_config.posterior

        self.pretrained_vae = pretrained_vae
        self.pretrained_ldm = pretrained_ldm
        self.zT_samples = precomputed_zT_samples
        self.ddim_sampler = ddim_sampler
        self.diff_t_steps = np.flip(ddim_sampler.time_steps)
        print('Diffusion time steps ', self.diff_t_steps)
        self.device = self.pretrained_ldm.device
        if verbose:
            print('Running on ', self.device)

        self.temperature = temp
        if self.posterior == "iaf":
            self.posterior_iaf_config = IAFConfig(
                input_dim=(model_config.latent_dim,),
                n_blocks=3,
                n_hidden_in_made=2,
                hidden_size=model_config.hidden_size,
                context_dim=model_config.context_dim,
                include_batch_norm=False,
            )

            self.posterior_iaf_flow = IAF(self.posterior_iaf_config).to(self.device)


        if verbose:
            print('Freezing pre-trained VAE and pre-trained LDM...')
        
        for p in self.pretrained_vae.parameters():
            p.requires_grad = False
        for p in self.pretrained_ldm.parameters():
            p.requires_grad = False
        
        if verbose:
            print('Freezing done.')
            print('Number of trainable parameters: {:.1e}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
            print('Number of total parameters: {:.1e}'.format(sum(p.numel() for p in self.parameters())))


        self.GM = GM

        self.res_dict = {}
        for i in range(self.n_obs):
            self.res_dict[i] = {'rec_loss':0,
                                'reg_loss':0,
                                'count':0}

        

    def forward(self, inputs: Union[BaseDataset, torch.Tensor], vi_index=None, **kwargs):
        device = self.device
        # Expecting inputs to have a "data" key that is already [B, n_obs, C, H, W]
        if isinstance(inputs, dict):
            x = inputs["data"].to(device)
        else:
            x = inputs.to(device)
        # If x has an extra leading dimension (for instance, [1, 32, 10, 1, 64, 64]) squeeze it.
        if x.shape[0] == 1 and x.shape[1] > 1:
            x = x.squeeze(0)
        # Now x should have shape [B, n_obs, C, H, W].
        batch_size = x.shape[0]
        n_obs = x.shape[1]
        
        # Encode the input
        encoder_output = self.encoder(x.reshape(-1, *x.shape[2:]))
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        
        # Reshape to [B, n_obs, latent_dim]
        mu = mu.reshape(batch_size, n_obs, -1)
        log_var = log_var.reshape(batch_size, n_obs, -1)
        
        # Sample from the posterior
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Process through the decoder
        decoder_output = self.decoder(z.reshape(-1, z.shape[-1]))
        x_recon = decoder_output.reconstruction.reshape(batch_size, n_obs, *x.shape[2:])
        
        return ModelOutput(
            reconstruction=x_recon,
            z=z,
            mu=mu,
            log_var=log_var
        )

    def vae_loss_function(self, recon_x, x, mu, log_var, z0, epoch, zk=None, log_abs_det_jac_posterior=None, seq_mask=None, pix_mask=None):
        if self.prior == "standard" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vae_loss_function(recon_x, x, mu, log_var, z0, seq_mask, pix_mask)


        elif self.prior == "vamp" and self.posterior == "gaussian":
            loss, recon_loss, kld = self._vamp_loss_function(recon_x, x, mu, log_var, z0, epoch, seq_mask, pix_mask)


        elif self.posterior == "iaf":
            loss, recon_loss, kld = self._vae_iaf_loss_function(recon_x, x, mu, log_var, z0, zk, log_abs_det_jac_posterior, epoch, seq_mask, pix_mask)


        return loss, recon_loss, kld

    def _vae_loss_function(self, recon_x, x, mu, log_var, z, seq_mask=None, pix_mask=None):

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

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _vae_iaf_loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac, epoch, seq_mask=None, pix_mask=None):
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

        # starting gaussian log-density
        log_prob_z0 = (
            -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
        ).sum(dim=1)

        # prior log-density
        log_prob_zk = self._log_p_z(zk)

        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return ((recon_loss + KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


    def _vamp_loss_function(self, recon_x, x, mu, log_var, z, epoch, seq_mask=None, pix_mask=None):

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

        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
        KLD = -(log_p_z - log_q_z)

        if self.linear_scheduling > 0:
            beta = 1.0 * epoch / self.linear_scheduling
            if beta > 1 or not self.training:
                beta = 1.0

        else:
            beta = 1.0

        
        #print((recon_loss * mask.reshape_as(recon_loss)).mean(), (KLD* mask.reshape_as(recon_loss)).mean())


        return (
            ((recon_loss + beta * KLD) * seq_mask.reshape_as(recon_loss)).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )


    def loss_function(self, recon_x, x, mu, log_var, z_0_vi_index, z_seq, vi_index, z_vi_index, log_abs_det_jac_posterior, epoch, seq_mask=None, pix_mask=None):

        assert not torch.isnan(x).any()
        
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                (
                    F.mse_loss(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).mean(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                (
                    F.binary_cross_entropy(
                        recon_x.reshape(x.shape[0]*self.n_obs, -1),
                        x.reshape(x.shape[0]*self.n_obs, -1),
                        reduction="none"
                    ) * pix_mask.reshape(x.shape[0]*self.n_obs, -1)
                ).sum(dim=-1).reshape(x.shape[0], -1) * seq_mask
            ).mean(dim=-1)

        z0 = z_seq[:, 0]

        # starting gaussian log-density
        # it is q_\phi ( z_j | x_j) - same as LVAE
        log_prob_z_vi_index = (
            -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / (torch.exp(log_var) + 1e-7)) #adding small constant to avoid numerical instability in the denominator
        ).sum(dim=1) - log_abs_det_jac_posterior


        # prior log-density

        if vi_index == 0 or vi_index == self.n_obs - 1:
            log_prior_z_vi_index = self.log_p_j_hat(j= vi_index, z = z_vi_index)
            log_prior_z_vi_index = log_prior_z_vi_index.to(z_vi_index.device)
            KLD = log_prob_z_vi_index - log_prior_z_vi_index 
            KLD = torch.clamp(KLD, min = -2, max = 500)
        else:
            KLD = torch.zeros_like(log_prob_z_vi_index)

                
        # log_prior_z_vi_index = self.log_p_j_hat(j= vi_index, z = z_vi_index)
        # log_prior_z_vi_index = log_prior_z_vi_index.to(z_vi_index.device)
        # KLD = log_prob_z_vi_index - log_prior_z_vi_index
        # #########
        # KLD = - KLD #negative KL divergence
        # ###########
        KLD = torch.clamp(KLD, min = -2, max = 500)

        if self.linear_scheduling > 0:
            beta = self.beta * epoch / self.linear_scheduling
            if beta > self.beta or not self.training:
                beta = self.beta

        else:
            beta = self.beta

        return (recon_loss + beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _log_p_z(self, z):
        if self.prior == "standard":
            log_p_z = (-0.5 * torch.pow(z, 2)).sum(dim=1)
        
        elif self.prior == "vamp":
            C = self.vamp_number_components

            x = self.pseudo_inputs(self.idle_input.to(z.device)).reshape(
                (C,) + self.model_config.input_dim
            )

            # we bound log_var to avoid unbounded optim
            encoder_output = self.encoder(x)
            prior_mu, prior_log_var = (
                encoder_output.embedding,
                encoder_output.log_covariance,
            )

            z_expand = z.unsqueeze(1)
            prior_mu = prior_mu.unsqueeze(0)
            prior_log_var = prior_log_var.unsqueeze(0)

            log_p_z = (
                torch.sum(
                    -0.5
                    * (
                        prior_log_var
                        + (z_expand - prior_mu) ** 2 / torch.exp(prior_log_var)
                    ),
                    dim=2,
                )
                - torch.log(torch.tensor(C).type(torch.float))
            )

            log_p_z = torch.logsumexp(log_p_z, dim=1)

        return log_p_z
    
    def log_p_j_hat(self, j, z):
        """
        Prior on z_j

        Args:
            j (int, 0 <= j <= self.n_obs-1): index of the latent variable (within the sequence)
            z_j (torch.Tensor shape (batch_size, lat_dim)): latent variable
        
        Returns:
            torch.Tensor: prior log-density of z_j

        """

        #For these two special cases, we do not need the sampled z_T, as we know tractable priors

        assert j >= 0 and j < self.n_obs
        if j == 0:
            #z0 follows a standard normal prior
            return (-0.5 * torch.pow(z, 2)).sum(dim=1)
        
        #if j == self.n_obs-1:
        else:
            return self.pretrained_vae.log_pi(z) # log sqrt det G(z) = 0.5 log det G(z) with precompiled G !

            # z_np = z.clone().cpu().detach().numpy()
            # #GM prior
            # return torch.tensor(self.GM.score_samples(z_np)).to(z.device)
        
        # t_diff = self.diff_t_steps[j]
        
        # alpha_bar_t_diff = self.pretrained_ldm.alpha_bar[t_diff]
        # sqrt_alpha_bar_t_diff = self.pretrained_ldm.sqrt_alpha_bar[t_diff]
        # mean = sqrt_alpha_bar_t_diff * self.zT_samples
        # mean = mean.unsqueeze(0).repeat(z.shape[0], 1, 1)

        # mean = mean.to(z.device)

        # log_density = - torch.sum( (z.unsqueeze(1) - mean)**2 / (2 * (1 - alpha_bar_t_diff)), dim = -1)
        # log_density = log_density.mean(dim = -1) #Monte-Carlo average (over the 1000 zT samples)
        # log_density = log_density.mean() #average over the batch
        # return log_density

    def reconstruct(self, input, vi_index, z_vi_index = None):

        device = self.device

        x = input["data"].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 or len(x.shape) == 2  else x

        if hasattr(input, 'seq_mask'):
            seq_mask = input['seq_mask'].to(device)
        else:
            seq_mask = torch.ones(x.shape[0], self.n_obs).to(device)
        
        if len(x.shape) == 3:
            x = x * seq_mask.unsqueeze(-1)
            x = torch.nan_to_num(x)
        else:
            x = x * seq_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


        batch_size = x.shape[0]

        encoder_output = self.encoder(x[:, vi_index])
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance


        std = torch.exp(0.5 * log_var)
        #std = torch.zeros_like(log_var)
        z, _ = self._sample_gauss(mu, std) ######## À MODIFIER !!!! Début de la reconstruction
        z_0_vi_index = z

        log_abs_det_jac_posterior = 0
        if self.posterior == 'iaf':
            if self.posterior_iaf_config.context_dim is not None:
                try:
                    h = encoder_output.context

                except AttributeError as e:
                    raise AttributeError(
                        "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                        "something different from None please ensure that the encoder actually outputs "
                        f"the context vector 'h'. Exception caught: {e}."
                    )

                # Pass it through the Normalizing flows
                flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

            else:
                # Pass it through the Normalizing flows
                flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

            z = flow_output.out
            log_abs_det_jac_posterior += flow_output.log_abs_det_jac

        z_vi_index = z if z_vi_index is None else z_vi_index

                ##### FROM LVAE to LLDM ########

        ## propagate in past - Forward Diffusion (Noising)
        z_seq = []
        z_rev = z_vi_index
        for i in range(vi_index - 1, -1, -1): #noising in a sequential way

            #To keep the forward pass parallelisable, we repeat the same sampled vi_index
            t1 = self.diff_t_steps[(i+1)*np.ones(batch_size).astype(int)]
            t2 =  self.diff_t_steps[i*np.ones(batch_size).astype(int)]
            z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

            z_seq.append(z_rev)
##
        z_seq.reverse()
#
        z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

        #propagate in future - Backward Diffusion (Denoising)
        z_for = z_vi_index
        
        for i in range(vi_index, self.n_obs - 1):
            t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
            side = int(np.sqrt(z_for.shape[1] // 3))
            assert 3 * side * side == z_for.shape[1], f"Latent dim {z_for.shape[1]} incompatible with 3xHxW reshape"
            z_for = z_for.reshape(batch_size, 3, side, side).float().to(self.pretrained_ldm.device)
            noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
            z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                index = self.n_obs -1- i,
                                                                x = z_for,
                                                                temperature=self.temperature,
                                                                repeat_noise=False)
            
            z_for = z_for.reshape(batch_size, -1).to(self.pretrained_ldm.device)
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
        recon_x = self.decoder(z_seq)["reconstruction"].cpu()#.detach() #moooooooodif

        return z_seq.cpu(), recon_x#.detach(), recon_x #modddddiiffffff

    def oversample(self, x, vi_index = 0, z_vi_index = None, sampler = None, num_supp_steps = None, verbose = False):

        device = self.device
        x = x["data"].to(device)
        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        batch_size = x.shape[0]

        encoder_output = self.encoder(x[:, vi_index])
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance


        std = torch.exp(0.5 * log_var)
        #std = torch.zeros_like(log_var)
        z, _ = self._sample_gauss(mu, std) ######## SAMPLE z_0 !!!!
        z_0_vi_index = z

        log_abs_det_jac_posterior = 0
        if self.posterior == 'iaf':
            if self.posterior_iaf_config.context_dim is not None:
                try:
                    h = encoder_output.context

                except AttributeError as e:
                    raise AttributeError(
                        "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                        "something different from None please ensure that the encoder actually outputs "
                        f"the context vector 'h'. Exception caught: {e}."
                    )

                # Pass it through the Normalizing flows
                flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

            else:
                # Pass it through the Normalizing flows
                flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

            z = flow_output.out
            log_abs_det_jac_posterior += flow_output.log_abs_det_jac

        z_vi_index = z if z_vi_index is None else z_vi_index

        if sampler is None:
            assert num_supp_steps is not None
            sampler = DDIMSampler(self.pretrained_ldm, n_steps = self.n_obs-1+num_supp_steps, ddim_eta = 1)
        self.oversampling_diff_t_steps = np.flip(np.sort(sampler.time_steps))
        if verbose:
            print('Updating the DDIM sampler...')
            print('Previous diffusion timeline: ', self.diff_t_steps)
            print('New diffusion timeline: ', self.oversampling_diff_t_steps)

        matching_index = np.argmin(np.abs(self.oversampling_diff_t_steps - self.diff_t_steps[vi_index]))

        
        ## propagate in past - Forward Diffusion (Noising)
        z_seq = []
        z_rev = z_vi_index
        for i in range(matching_index - 1, -1, -1): #noising in a sequential way

            #To keep the forward pass parallelisable, we repeat the same sampled vi_index
            t1 = self.oversampling_diff_t_steps[(i+1)*np.ones(batch_size).astype(int)]
            t2 =  self.oversampling_diff_t_steps[i*np.ones(batch_size).astype(int)]
            z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

            z_seq.append(z_rev)
##
        z_seq.reverse()
#
        z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

        #propagate in future - Backward Diffusion (Denoising)
        z_for = z_vi_index
        
        for i in range(matching_index, len(self.oversampling_diff_t_steps) - 1):
            t = torch.tensor(self.oversampling_diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
            side = int(np.sqrt(z_for.shape[1] // 3))
            assert 3 * side * side == z_for.shape[1], f"Latent dim {z_for.shape[1]} incompatible with 3xHxW reshape"
            z_for = z_for.reshape(batch_size, 3, side, side).float().to(self.pretrained_ldm.device)
            noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
            z_for, _ = sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                index = len(self.oversampling_diff_t_steps) -1- i,
                                                                x = z_for,
                                                                temperature=self.temperature,
                                                                repeat_noise=False)
            
            z_for = z_for.reshape(batch_size, -1).to(self.pretrained_ldm.device)
            z_seq.append(z_for)

        z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim)
        recon_x = self.decoder(z_seq)["reconstruction"]

        return z_seq, recon_x
        

        









    def generate(self, train_data, num_gen_seq = 1, vi_index = 0, T_multiplier = 0.5, batch_size = 128, freeze = False, device = 'cpu', verbose = True):
        
        self = self.to(device)
        model_config = VAEConfig(input_dim=self.input_dim, latent_dim= self.latent_dim, uses_default_encoder= False, uses_default_decoder= False, reconstruction_loss= 'mse')
        final_vae = VAE(model_config = model_config, encoder = self.encoder, decoder = self.decoder)
        obs_data = train_data[:, vi_index]
        _, mu, log_var = final_vae.retrieveG(obs_data, verbose = verbose, T_multiplier=T_multiplier, device = device, addStdNorm=False)

        batch_size = num_gen_seq if num_gen_seq <= batch_size else batch_size
        all_z_vi = []

        final_vae = final_vae.to(device)
        mu = mu.to(device)

        if verbose:
            if freeze:
                print(f'Freezing the {vi_index}th/rd obs...')
                print(f'Sampling 1 point on the {vi_index}th/rd manifold...')
            
            else:
                print(f'Sampling {num_gen_seq} points on the {vi_index}th/rd manifold...')

        if not freeze:

            #the for loop and the if condition enables at the end to have a list of z_vi_index of size num_gen_seq
            for j in range(0, num_gen_seq // batch_size):
                z, p = hmc_sampling(final_vae, mu, n_samples=batch_size, mcmc_steps_nbr=100)
                all_z_vi.append(z)
            
            if num_gen_seq % batch_size != 0:
                z,p = hmc_sampling(final_vae, mu, n_samples=num_gen_seq % batch_size, mcmc_steps_nbr=100)
                all_z_vi.append(z)

        else:
            z,p = hmc_sampling(final_vae, mu, n_samples=1, mcmc_steps_nbr=100)
            all_z_vi = [z]*num_gen_seq #all_z_vi is size (num_gen_seq, latent_dim). As we freeze the vi_indes^th observation, we copy


        full_recon_x, full_z_seq = [], []
        for j in range(0, num_gen_seq // batch_size):
            if verbose:
                print(f'Batch {j+1}/{num_gen_seq // batch_size}')
            z_vi_index = torch.cat(all_z_vi[j*batch_size: (j+1)*batch_size], dim=0) if freeze else all_z_vi[j]
            z_seq = []
            z_rev = z_vi_index
            ## propagate in past - Forward Diffusion (Noising)
            z_seq = []
            z_rev = z_vi_index
            if verbose and vi_index > 0:
                print('Propagating in the past...')

            for i in range(vi_index - 1, -1, -1): #noising in a sequential way
                #To keep the forward pass parallelisable, we repeat the same sampled vi_index
                t1 = self.diff_t_steps[(i+1)*np.ones(z_rev.shape[0]).astype(int)]
                t2 =  self.diff_t_steps[i*np.ones(z_rev.shape[0]).astype(int)]
                z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

                z_seq.append(z_rev)
    ##
            z_seq.reverse()
    #
            z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

            #propagate in future - Backward Diffusion (Denoising)
            z_for = z_vi_index
            if verbose and vi_index < self.n_obs - 1:
                print('Propagating in the future...')
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
                z_for = z_for.reshape(-1, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                    index = self.n_obs -1- i,
                                                                    x = z_for,
                                                                    temperature=self.temperature,
                                                                    repeat_noise=False)
                
                z_for = z_for.reshape(-1, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim) # (batch_size * n_obs, latent_dim)
            if verbose:
                print('Decoding...')
            
            if len(self.input_dim) == 3:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            if len(self.input_dim) == 2:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[1])
            
            z_seq = z_seq.reshape(-1, self.n_obs, self.latent_dim)
            full_recon_x.append(recon_x.detach().cpu())
            full_z_seq.append(z_seq.detach().cpu())
        
        if num_gen_seq % batch_size != 0:
            if verbose:
                print(f'Remainder batch size...: size', num_gen_seq % batch_size)

            rem = num_gen_seq % batch_size
            z_vi_index = torch.cat(all_z_vi[-rem:], dim=0) if freeze else all_z_vi[-1]
            ## propagate in past - Forward Diffusion (Noising)
            z_seq = []
            z_rev = z_vi_index
            if verbose and vi_index > 0:
                print('Propagating in the past...')

            for i in range(vi_index - 1, -1, -1): #noising in a sequential way
                #To keep the forward pass parallelisable, we repeat the same sampled vi_index
                t1 = self.diff_t_steps[(i+1)*np.ones(z_rev.shape[0]).astype(int)]
                t2 =  self.diff_t_steps[i*np.ones(z_rev.shape[0]).astype(int)]
                z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

                z_seq.append(z_rev)
        ##
            z_seq.reverse()
        #
            z_seq.append(z_vi_index.to(self.pretrained_ldm.device))



            #propagate in future - Backward Diffusion (Denoising)
            z_for = z_vi_index
            if verbose and vi_index < self.n_obs - 1:
                print('Propagating in the future...')
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
                z_for = z_for.reshape(-1, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                    index = self.n_obs -1- i,
                                                                    x = z_for,
                                                                    temperature=self.temperature,
                                                                    repeat_noise=False)
                
                z_for = z_for.reshape(-1, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for)

            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim) # (batch_size * n_obs, latent_dim)

            if verbose:
                print('Decoding...')

            if len(self.input_dim) == 3:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            else:
                recon_x = self.decoder(z_seq)["reconstruction"].reshape(-1, self.n_obs, self.input_dim[1])
            
            z_seq = z_seq.reshape(-1, self.n_obs, self.latent_dim)
            full_recon_x.append(recon_x.detach().cpu())
            full_z_seq.append(z_seq.detach().cpu())
        
        full_recon_x = torch.cat(full_recon_x, dim=0)
        full_z_seq = torch.cat(full_z_seq, dim=0)

        return full_recon_x, full_z_seq

    def predict(self, x, vi_index, num_gen_seq = 1, batch_size = 100, device = 'cpu'):
        """
        Predict the latent variables and the reconstruction of the data at the vi_index-th time step

        Args:
            data (torch.Tensor): The input data from which the latent variables and the reconstruction should be predicted.
                Data must be of shape [Batch x n_channels x ...]
            vi_index (int): The index of the variable of interest
            device (str): The device on which the data should be loaded

        Returns:
            torch.Tensor: The predicted latent variables
            torch.Tensor: The predicted reconstruction
        """

        self = self.to(device)
        batch_size = num_gen_seq if num_gen_seq <= batch_size else batch_size


        x = x.unsqueeze(0) if len(x.shape) == 4 else x
        n_seq = x.shape[0]
        x_vi_index = x[:, vi_index].to(device)
        z_vi_index = self.encoder(x_vi_index.unsqueeze(0)).embedding
        z_vi_index = torch.cat(num_gen_seq*[z_vi_index], dim=1).reshape(n_seq, num_gen_seq, self.latent_dim) # (n_seq, num_gen_seq, latent_dim), with repetition on the axis 1 of the same z_vi_index (for each sequence)

        #Predict future with backward diffusion
        all_pred_x = []

        for j in range(0, num_gen_seq // batch_size):
            z_seq = []
            z_for = z_vi_index[:, j*batch_size: (j+1)*batch_size] # (n_seq, batch_size, latent_dim)
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
                z_for = z_for.reshape(-1, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                    index = self.n_obs -1- i,
                                                                    x = z_for,
                                                                    temperature=self.temperature,
                                                                    repeat_noise=False)
                
                z_for = z_for.reshape(-1, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for) 
            #At the end of the loop, z_seq contains the latent of vi_index + 1 to T

            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim) # (n_seq * batch_size * n_obs - vi_index, latent_dim)
            
            if len(self.input_dim) == 3:
                pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, batch_size, self.n_obs - vi_index - 1, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            else:
                pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, batch_size, self.n_obs - vi_index - 1, self.input_dim[1])
            all_pred_x.append(pred_x)

        if num_gen_seq % batch_size != 0:
            rem = num_gen_seq % batch_size
            z_seq = []
            z_for = z_vi_index[:, -rem:]
            for i in range(vi_index, self.n_obs - 1):
                t = torch.tensor(self.diff_t_steps[i]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
                z_for = z_for.reshape(n_seq, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
                z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                    index = self.n_obs -1- i,
                                                                    x = z_for,
                                                                    temperature=self.temperature,
                                                                    repeat_noise=False)
                
                z_for = z_for.reshape(n_seq, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                z_seq.append(z_for) 
            #At the end of the loop, z_seq contains the latent of vi_index + 1 to T

            z_seq = torch.cat(z_seq, dim=-1).reshape(-1, self.latent_dim) # (n_seq * batch_size * n_obs - vi_index, latent_dim)
            
            pred_x = self.decoder(z_seq)["reconstruction"].reshape(n_seq, batch_size, self.n_obs - vi_index - 1, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            all_pred_x.append(pred_x)

        all_pred_x = torch.cat(all_pred_x, dim=1).to(device) # (n_seq, num_gen_seq, n_obs - vi_index, input_dim)
        #prev_x = torch.repeat_interleave(x[:, :vi_index+1], num_gen_seq, dim=1).reshape((n_seq, num_gen_seq, vi_index + 1) + self.input_dim).to(device) # (n_seq, num_gen_seq, vi_index + 1, input_dim)

        return all_pred_x




    def get_nll(self, data, vi_index, n_samples=1, batch_size=100):
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
        for i in tqdm(range(len(data))):
            x = data[i].unsqueeze(0).to(self.device) # (1, 7, 3, 64, 64)

            log_p_x = []

            for _ in range(n_full_batch):

                x_rep = torch.cat(batch_size * [x]) # (100, 7, 3, 64, 64)

                encoder_output = self.encoder(x_rep[:, vi_index]) # (100, 12) 

                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std) ###idem pour la loss

                z_0_vi_index = z

                log_abs_det_jac_posterior = 0
                if self.posterior == 'iaf':

                    if self.posterior_iaf_config.context_dim is not None:
                        try:
                            h = encoder_output.context

                        except AttributeError as e:
                            raise AttributeError(
                                "Cannot get context from encoder outputs. If you set `context_dim` argument to "
                                "something different from None please ensure that the encoder actually outputs "
                                f"the context vector 'h'. Exception caught: {e}."
                            )

                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z, h=h)  # sampling

                    else:
                        # Pass it through the Normalizing flows
                        flow_output = self.posterior_iaf_flow.inverse(z)  # sampling

                    z = flow_output.out
                    log_abs_det_jac_posterior += flow_output.log_abs_det_jac

                z_vi_index = z

                ## propagate in past - Forward Diffusion (Noising)
                z_seq = []
                z_rev = z_vi_index
                for k in range(vi_index - 1, -1, -1): #noising in a sequential way

                    #To keep the forward pass parallelisable, we repeat the same sampled vi_index
                    t1 = self.diff_t_steps[(k+1)*np.ones(batch_size).astype(int)]
                    t2 =  self.diff_t_steps[k*np.ones(batch_size).astype(int)]
                    z_rev = self.pretrained_ldm.sequential_diffusion(x= z_rev, t1 = t1, t2 = t2).to(self.pretrained_ldm.device).float()

                    z_seq.append(z_rev)
    ##
                z_seq.reverse()
    #
                z_seq.append(z_vi_index.to(self.pretrained_ldm.device))

                #propagate in future - Backward Diffusion (Denoising)
                z_for = z_vi_index
                
                for k in range(vi_index, self.n_obs - 1):
                    t = torch.tensor(self.diff_t_steps[k]).reshape(1).to(self.pretrained_ldm.device).float() #diffusion time-step
                    z_for = z_for.reshape(batch_size, self.pretrained_ldm.c, self.pretrained_ldm.h, self.pretrained_ldm.w).float().to(self.pretrained_ldm.device)
                    noise_pred = self.pretrained_ldm(z_for, t) # \eps_\theta (z_t, t)
                    z_for, _ = self.ddim_sampler.get_x_prev_and_pred_x0(e_t = noise_pred,
                                                                    index = self.n_obs - 1- k,
                                                                    x = z_for,
                                                                    temperature=self.temperature,
                                                                    repeat_noise=False)
                    
                    z_for = z_for.reshape(batch_size, self.pretrained_ldm.c * self.pretrained_ldm.h * self.pretrained_ldm.w).to(self.pretrained_ldm.device)
                    z_seq.append(z_for)

                z_seq = torch.cat(z_seq, dim=-1)

                ###############################

                recon_x = self.decoder(z_seq.reshape(-1, self.latent_dim))["reconstruction"]# [B*n_obs x input_dim] # (700, 3, 64, 64)

                z_seq = z_seq.reshape(x_rep.shape[0], self.n_obs, self.latent_dim) # (100, 7, 12)

                
                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = (-0.5 * F.mse_loss(
                            recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                            x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                            reduction="none",
                        ).sum(dim=-1) - torch.tensor(
                            [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                        ).to(
                            x_rep.device
                        )
                    ).reshape(x_rep.shape[0], -1).mean(dim=-1) # decoding distribution is assumed unit variance  N(mu, I)

                    #
                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0]*self.n_obs, -1),
                        x_rep.reshape(x_rep.shape[0]*self.n_obs, -1),
                        reduction="none",
                    ).sum(dim=-1).reshape(x_rep.shape[0], -1).mean(dim=-1)

                z0 = z_seq[:, 0]

                # starting gaussian log-density
                log_prob_z_vi_index = (
                    -0.5 * (log_var + torch.pow(z_0_vi_index - mu, 2) / torch.exp(log_var))
                ).sum(dim=1) - log_abs_det_jac_posterior

                log_p_z = self._log_p_z(z0) 

                # prior log-density
                log_prior_z_vi_index = log_p_z

                # log_p_x.append(
                #     log_p_x_given_z.detach().cpu() + log_prior_z_vi_index.detach().cpu() - log_prob_z_vi_index.detach().cpu()
                # )  # log(2*pi) simplifies
                log_p_x.append(
                    log_p_x_given_z.cpu()#.detach().cpu()
                )

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

            if i % 100 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)

    def encode(self, x):
        """Encodes the input data into the latent space."""
        device = self.device
        x = x.to(device)
        if len(x.shape) == 4:  # If input is [B, C, H, W]
            x = x.unsqueeze(1)  # Add observation dimension: [B, 1, C, H, W]
        batch_size, n_obs, _, _, _ = x.shape
        
        # Flatten the observation dimension
        encoder_output = self.encoder(x.reshape(-1, *x.shape[2:]))
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        
        # Reshape to [B, n_obs, latent_dim]
        mu = mu.reshape(batch_size, n_obs, -1)
        log_var = log_var.reshape(batch_size, n_obs, -1)
        
        # Sample from the posterior
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

