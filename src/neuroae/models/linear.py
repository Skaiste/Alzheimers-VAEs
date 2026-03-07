import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    # a placeholder since the loss function doesn't have any parameters
    def set_loss_fn_params(self, params):
        self.params = params

    def _to_fc(self, x):
        fc = x.corrcoef()
        fc = fc.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        tri = torch.triu_indices(fc.shape[0], k=1)
        return fc[tri]
    
    def _vector_correlation(a, b):
        a = a.reshape(-1)
        b = a.reshape(-1)

        if a.size != b.size or a.size < 2:
            return torch.nan

        a_std = float(a.std())
        b_std = float(b.std())
        if a_std == 0.0 or b_std == 0.0:
            return torch.nan

        return float(torch.corrcoef([a, b])[0, 1])

    def loss(self, x, model_output):
        if self.params.get("fc_preservation", False):
            # calculate FC preservation and invert it to add to the loss function
            x_2d = x.reshape(x.shape[0], 400, 197)
            recon_2d = model_output[0].reshape(model_output[0].shape[0], 400, 197)
            # get fcs for both
            x_fc = self._to_fc(x_2d)
            recon_fc = self._to_fc(recon_2d)
            corr = self._vector_correlation(x_fc, recon_fc)
            # ???
            pass
        else:
            return {'loss': F.mse_loss(model_output[0], x)}
    
