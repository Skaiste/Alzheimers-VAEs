import torch.nn.functional as F


def recon_kld_loss(x, model_output, error_per_feature=True, kld_weight=1.0):
    x_hat, mu, log_var = model_output
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


def recon_kld_loss_2d(x, model_output, error_per_feature=True, kld_weight=1.0):
    x_hat, mu, log_var = model_output
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


def get_loss_function(fn_name):
    lfn_index = {
        "recon_kld_loss": recon_kld_loss,
        "recon_kld_loss_2d": recon_kld_loss_2d
    }
    if fn_name not in lfn_index:
        raise ValueError(f"The loss function {fn_name} does not exist")
    return lfn_index[fn_name]