import torch
from src.lib.src.pythae.data.datasets import DatasetOutput
from torch.utils.data import Dataset
from src.lib.src.pythae.models.nn import BaseEncoder, BaseDecoder
from src.lib.src.pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import numpy as np


def make_batched_masks(data, prob_missing_data, batch_size):
    mask = torch.zeros(data.shape[:2], requires_grad=False)
    prob = ((1 - prob_missing_data) - 2 / data.shape[1]) * data.shape[1] / (data.shape[1] - 2)

    for i in range(int(data.shape[0] / batch_size)):

        bern = torch.distributions.Bernoulli(probs=prob).sample((data.shape[1]-2,))
        
        _mask = torch.zeros(data.shape[1])
        _mask[:2] = 1
        _mask[2:] = bern

        idx = np.random.rand(*_mask.shape).argsort(axis=-1)
        
        _mask = np.take_along_axis(_mask, idx, axis=-1)
        mask[i*batch_size:(i+1)*batch_size] = _mask.repeat(batch_size, 1)

    if data.shape[0] % batch_size > 0:

        bern = torch.distributions.Bernoulli(probs=prob).sample((data.shape[1]-2,))
        
        _mask = torch.zeros(data.shape[1])
        _mask[:2] = 1
        _mask[2:] = bern

        idx = np.random.rand(*_mask.shape).argsort(axis=-1)
        _mask = np.take_along_axis(_mask, idx, axis=-1)

        mask[-(data.shape[0] % batch_size):] = _mask.repeat((data.shape[0] % batch_size), 1)


    return mask


class My_MaskedDataset(Dataset):
    def __init__(self, data, seq_mask, pix_mask):
        self.data = data.cpu().type(torch.float)
        self.sequence_mask = seq_mask.cpu().type(torch.float)
        self.pixel_mask = pix_mask.cpu().type(torch.float)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        seq_m = self.sequence_mask[index]
        pix_m = self.pixel_mask[index] 
        #x = (x > torch.distributions.Uniform(0, 1).sample(x.shape).to(x.device)).float()
        return DatasetOutput(data=x, seq_mask=seq_m, pix_mask=pix_m)


class My_Dataset(Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        #x = (x > torch.distributions.Uniform(0, 1).sample(x.shape).to(x.device)).float()
        return DatasetOutput(data=x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


### Define paper encoder network
class Encoder_ColorMNIST(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

class Encoder_ColorMNIST_GPVAE(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, 2 * self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class Decoder_ColorMNIST(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(self.input_dim)),
            nn.Sigmoid(),
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output


class Encoder_Chairs(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]

        layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, 4, 2, padding=1),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
        )

        self.layers = layers
        self.flattened_size = 128 * 4 * 4  # 2048
        self.embedding = nn.Linear(self.flattened_size, self.latent_dim)
        self.log_var = nn.Linear(self.flattened_size, self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(self.flattened_size, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()
        
        # Handle input shape
        if len(x.shape) == 5:  # [batch, seq, channel, height, width]
            batch_size, n_obs = x.shape[:2]
            x = x.reshape(-1, *x.shape[2:])  # [-1, channel, height, width]
        elif len(x.shape) == 4:  # [batch, channel, height, width]
            batch_size = x.shape[0]
            n_obs = 1
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Process through convolutional layers
        out = self.layers(x)  # [-1, 128, 4, 4]
        out = out.reshape(-1, self.flattened_size)  # [-1, 2048]
        
        # Project to latent space
        embedding = self.embedding(out)  # [-1, latent_dim]
        log_var = self.log_var(out)  # [-1, latent_dim]
        
        # Reshape back to include sequence dimension if needed
        if n_obs > 1:
            embedding = embedding.reshape(batch_size, n_obs, self.latent_dim)
            log_var = log_var.reshape(batch_size, n_obs, self.latent_dim)
        
        output["embedding"] = embedding
        output["log_covariance"] = log_var
        
        if self.context_dim is not None:
            context = self.context(out)
            if n_obs > 1:
                context = context.reshape(batch_size, n_obs, self.context_dim)
            output["context"] = context
        
        return output


class Encoder_Chairs_GPVAE(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]
        

        layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, 4, 2, padding=1),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),

        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, 2 * args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 4 * 4, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*4*4))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*4*4))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output


class Decoder_Chairs(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.fc = nn.Linear(args.latent_dim, 128 * 4 * 4)

        layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.n_channels, 4, 2, padding=1)
        )   

        self.layers = layers

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        # Handle sequence dimension if present
        if len(z.shape) == 3:  # [batch, seq, latent_dim]
            batch_size, seq_len, _ = z.shape
            z = z.reshape(-1, self.latent_dim)  # [batch*seq, latent_dim]
            has_sequence = True
        else:  # [batch, latent_dim]
            batch_size = z.shape[0]
            seq_len = 1
            has_sequence = False

        # Process through layers
        out = self.fc(z).reshape(-1, 128, 4, 4)  # [batch*seq, 128, 4, 4]
        out = self.layers(out)  # [batch*seq, channels, height, width]

        # Reshape output to include sequence dimension if needed
        if has_sequence:
            out = out.reshape(batch_size, seq_len, *out.shape[1:])  # [batch, seq, channels, height, width]
            # Permute to match input format [batch, seq, height, width, channels]
            out = out.permute(0, 1, 3, 4, 2)
        else:
            out = out.reshape(batch_size, *out.shape[1:])  # [batch, channels, height, width]
            # Permute to match input format [batch, height, width, channels]
            out = out.permute(0, 2, 3, 1)

        output["reconstruction"] = out
        return output

### Define paper encoder network
class Encoder_HMNIST(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        #self.conv = nn.Sequential(
        #    nn.Conv2d(1, 256, 3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(256, 1, 3, padding=1),
        #    nn.ReLU()
        #)

        #self.time_cnn = nn.Conv1d(np.prod(args.input_dim), args.out_channels_time_cnn, kernel_size=3, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.input_dim), 256),
            nn.ReLU(),
            #nn.Linear(256, 256),
            #nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, 2*self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()

        #x = torch.transpose(self.time_cnn(torch.transpose(x.reshape(x.shape[0], x.shape[1], -1), 2, 1)), 2, 1)
        #out = self.conv(x.reshape((-1,) + self.input_dim))
        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class Decoder_HMNIST(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(self.input_dim)),
            nn.Sigmoid()
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output


### Define paper encoder network
class Encoder_Sprites_Missing(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None

        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(256, self.latent_dim)
        self.log_var = nn.Linear(256, self.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(256, self.context_dim)

    def forward(self, x):
        output = ModelOutput()
        #out = self.conv(x.reshape((-1,) + self.input_dim))
        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

### Define paper decoder network
class Decoder_Sprites_Missing(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(self.input_dim)),
            nn.Sigmoid()
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output

class Encoder_Faces(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]

        layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2),
            nn.Conv2d(16, 32, 5, 2),
            nn.Conv2d(32, 64, 5, 2),
            nn.Conv2d(64, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 13 * 7, args.latent_dim)
        self.log_var = nn.Linear(128 * 13 * 7, args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 13 * 7, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*13*7))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*13*7))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output

class Encoder_Faces_GPVAE(BaseEncoder):

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        if hasattr(args, "context_dim"):
            self.context_dim = args.context_dim
        else:
            self.context_dim = None
        self.n_channels = args.input_dim[0]
        

        layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2),
            nn.Conv2d(16, 32, 5, 2),
            nn.Conv2d(32, 64, 5, 2),
            nn.Conv2d(64, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            nn.Conv2d(128, 128, 5, 2),
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
        )

        self.layers = layers

        self.embedding = nn.Linear(128 * 13 * 7, args.latent_dim)
        self.log_var = nn.Linear(128 * 13 * 7, 2 * args.latent_dim)
        if self.context_dim is not None:
            self.context = nn.Linear(128 * 13 * 7, self.context_dim)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()

        out = self.layers(x.reshape((-1,) + self.input_dim))
        output["embedding"] = self.embedding(out.reshape(-1, 128*4*4))
        output["log_covariance"] = self.log_var(out.reshape(-1, 128*4*4))
        if self.context_dim is not None:
            output["context"] = self.context(out)

        return output


class Decoder_Faces(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.fc = nn.Linear(args.latent_dim, 128 * 13 * 7)

        layers = nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32),
            nn.ConvTranspose2d(128, 128, 5, 2, output_padding=(0, 1)),
            nn.ConvTranspose2d(128, 128, 5, 2, output_padding=(0, 1)),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=(0, 0)),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=(0, 0), output_padding=(1, 0)),
            nn.ConvTranspose2d(32, 16, 5, 2, padding=(0, 1), output_padding=(0, 1)),
            nn.ConvTranspose2d(16, 3, (4, 5), 2, padding=(0, 1)), 
        )

        self.layers = layers

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        out = self.fc(z).reshape(z.shape[0], 128, 13, 7)
        out = self.layers(out)

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output
    

class Encoder_ADNI(BaseEncoder):
    def __init__(self, input_dim, latent_dim):
        BaseEncoder.__init__(self)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_dim), 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(15, self.latent_dim)
        self.log_var = nn.Linear(15, self.latent_dim)

    def forward(self, x):
        output = ModelOutput()
        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)

        return output
    
class Encoder_ADNI_GPVAE(BaseEncoder):
    def __init__(self, input_dim, latent_dim):
        BaseEncoder.__init__(self)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_dim), 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 15),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(15, self.latent_dim)
        self.log_var = nn.Linear(15, 2*self.latent_dim)

    def forward(self, x):
        output = ModelOutput()
        out = self.fc(x.reshape(-1, np.prod(self.input_dim)))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)

        return output
    
class Decoder_ADNI(BaseDecoder):
    def __init__(self, input_dim, latent_dim):
        BaseDecoder.__init__(self)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.ReLU(),
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Linear(60, np.prod(self.input_dim)),
            nn.ReLU(),
        )


    def forward(self, z: torch.Tensor):

        output = ModelOutput()

        out = self.fc(z)

        output["reconstruction"] = out.reshape((z.shape[0], np.prod(self.input_dim)))

        return output