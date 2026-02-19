from torch import nn
import torch.nn.functional as F

class LossFunction:
    def __init__(self, loss_name, loss_params):
        self.lfn_index = {
            "recon_kld_loss": self.recon_kld_loss,
            "recon_kld_loss_2d": self.recon_kld_loss_2d,
            "mse_loss": self.mse_loss
        }
        if loss_name not in self.lfn_index:
            raise ValueError(f"The loss function {loss_name} does not exist")
        self.loss_name = loss_name
        self.loss_params = {k:v for p in loss_params for k,v in p.items()}

        if self.loss_name == "mse_loss":
            self.criterion = nn.MSELoss()

    def run(self, *args, **kwargs):
        return self.lfn_index[self.loss_name](*args, **kwargs)

    def recon_kld_loss(self, x, model_output):
        x_hat, mu, log_var = model_output
        error_per_feature = self.loss_params.get("loss_per_feature", True)
        kld_weight = float(self.loss_params.get("kld_weight", 1.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

            # KL: mean over batch, then mean over latent dims
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

            # kld: sum over latent dims per sample, then mean over batch
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        return {
            'loss': recon + kld_weight * kld,
            'recon': recon, 
            'kld': kld
        }


    def recon_kld_loss_2d(self, x, model_output):
        x_hat, mu, log_var = model_output
        error_per_feature = self.loss_params.get("loss_per_feature", True)
        kld_weight = float(self.loss_params.get("kld_weight", 1.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

            # KL: mean over batch, then mean over latent dims
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.flatten(1).sum(dim=1).mean() / log_var.size(1)

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

            # kld: sum over latent dims per sample, then mean over batch
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.flatten(1).sum(dim=1).mean() / log_var.size(1)

        return {
            'loss': recon + kld_weight * kld,
            'recon': recon, 
            'kld': kld
        }

    def mse_loss(self, x, output):
        return {'loss': self.criterion(output[0], x)}