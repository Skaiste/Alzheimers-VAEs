import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _dataset_valid_last_dim(dataset):
    if not getattr(dataset, "pad_features", False):
        return None
    original_shape = getattr(dataset, "original_shape", None)
    if original_shape is None or len(original_shape) == 0:
        return None
    valid_last_dim = int(original_shape[-1])
    if valid_last_dim <= 0:
        return None
    return valid_last_dim


def _trim_last_dim(x, dataset):
    valid_last_dim = _dataset_valid_last_dim(dataset)
    if valid_last_dim is None or x.shape[-1] <= valid_last_dim:
        return x
    return x[..., :valid_last_dim]


def _build_valid_mask(x, dataset):
    valid_last_dim = _dataset_valid_last_dim(dataset)
    if valid_last_dim is None or x.shape[-1] <= valid_last_dim:
        return None
    mask = torch.zeros_like(x)
    mask[..., :valid_last_dim] = 1.0
    return mask


def _apply_recon_mask(x, model_output, mask):
    if mask is None:
        return model_output

    def _mask_recon(recon):
        return recon * mask + x * (1.0 - mask)

    if isinstance(model_output, dict):
        out = dict(model_output)
        for key in ("x_hat", "recon", "reconstruction"):
            if key in out and torch.is_tensor(out[key]):
                out[key] = _mask_recon(out[key])
                break
        return out

    if isinstance(model_output, tuple):
        if len(model_output) == 0:
            return model_output
        return (_mask_recon(model_output[0]), *model_output[1:])

    if isinstance(model_output, list):
        if len(model_output) == 0:
            return model_output
        out = list(model_output)
        out[0] = _mask_recon(out[0])
        return out

    if torch.is_tensor(model_output):
        return _mask_recon(model_output)

    return model_output


def _masked_mse_torch(x_hat, x, mask):
    if mask is None:
        return float(F.mse_loss(x_hat, x, reduction="mean").item())
    se = (x_hat - x).pow(2) * mask
    denom = mask.sum().clamp_min(1.0)
    return float((se.sum() / denom).item())


def _masked_mse_numpy(x_hat, x, mask):
    if mask is None:
        return float(np.mean((x_hat - x) ** 2))
    se = ((x_hat - x) ** 2) * mask
    denom = np.maximum(np.sum(mask), 1.0)
    return float(np.sum(se) / denom)


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data)


def _extract_model_outputs(model_out):
    """Return reconstruction and latent matrix from model outputs."""
    if isinstance(model_out, dict):
        recon_x = model_out.get("x_hat") or model_out.get("recon") or model_out.get("reconstruction")
        latent = model_out.get("z") or model_out.get("mu")
    elif isinstance(model_out, (tuple, list)):
        recon_x = model_out[0]
        if len(model_out) >= 4:
            latent = model_out[3]
        elif len(model_out) >= 2:
            latent = model_out[1]
        else:
            latent = None
    else:
        recon_x = model_out
        latent = None

    if recon_x is None:
        raise ValueError("Could not extract reconstruction tensor from model output.")

    if latent is None:
        raise ValueError("Could not extract latent tensor from model output.")

    return recon_x, latent


def _reshape_for_timeseries(sample, dataset):
    """Try to recover (R, T) for FC-preservation computation."""
    x = _to_numpy(sample)

    if getattr(dataset, "fc_input", False):
        return None

    if x.ndim == 2:
        return x

    if x.ndim == 1 and getattr(dataset, "flatten", False):
        original_shape = getattr(dataset, "original_shape", None)
        if original_shape is None:
            return None
        expected = int(np.prod(original_shape))
        if x.size != expected:
            return None
        return x.reshape(original_shape)

    return None


def _vector_correlation(a, b):
    a = _to_numpy(a).reshape(-1)
    b = _to_numpy(b).reshape(-1)

    if a.size != b.size or a.size < 2:
        return np.nan

    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _fc_upper_vector_from_timeseries(ts, roi_axis=0):
    ts = _to_numpy(ts)
    if ts.ndim != 2:
        return None
    if roi_axis == 1:
        ts = ts.T

    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    tri = np.triu_indices(fc.shape[0], k=1)
    return fc[tri]


def _fc_vector(sample, dataset):
    ts = _reshape_for_timeseries(sample, dataset)
    if ts is None:
        # fallback: treat sample as already FC-like feature vector
        return _to_numpy(sample).reshape(-1)

    # dataset.transpose=True means sample orientation is (T, R), so ROI axis is 1.
    roi_axis = 1 if getattr(dataset, "transpose", False) else 0
    fc_vec = _fc_upper_vector_from_timeseries(ts, roi_axis=roi_axis)
    if fc_vec is None:
        return _to_numpy(sample).reshape(-1)
    return fc_vec


def _fc_preservation_score(x, x_hat, dataset):
    x_np = _to_numpy(x)
    x_hat_np = _to_numpy(x_hat)

    if getattr(dataset, "timepoints_as_samples", False):
        subject_ids = np.asarray(getattr(dataset, "subject_ids", []))
        if subject_ids.size != x_np.shape[0]:
            return np.nan

        scores = []
        unique_subjects = pd.unique(subject_ids)
        for sid in unique_subjects:
            idx = np.where(subject_ids == sid)[0]
            if idx.size < 2:
                continue

            # In timepoints_as_samples mode each row is one timepoint vector of ROIs.
            ts_x = x_np[idx]
            ts_hat = x_hat_np[idx]
            v1 = _fc_upper_vector_from_timeseries(ts_x, roi_axis=1)
            v2 = _fc_upper_vector_from_timeseries(ts_hat, roi_axis=1)
            if v1 is None or v2 is None:
                continue

            corr = _vector_correlation(v1, v2)
            if np.isfinite(corr):
                scores.append(corr)
        return float(np.mean(scores)) if scores else np.nan

    scores = []
    for i in range(x_np.shape[0]):
        v1 = _fc_vector(x_np[i], dataset)
        v2 = _fc_vector(x_hat_np[i], dataset)
        corr = _vector_correlation(v1, v2)
        if np.isfinite(corr):
            scores.append(corr)

    return float(np.mean(scores)) if scores else np.nan


def _encode_labels(labels):
    labels = np.asarray(labels)
    valid_mask = pd.notna(labels)
    labels = labels[valid_mask]
    if labels.size == 0:
        return np.array([]), valid_mask

    classes, encoded = np.unique(labels.astype(str), return_inverse=True)
    return encoded.astype(int), valid_mask


def _silhouette(latents, labels):
    if len(latents.shape) > 2: # if the latent space is 2D
        latents = latents.reshape(latents.shape[0], -1)
    y, valid_mask = _encode_labels(labels)
    if y.size == 0:
        return np.nan
    z = latents[valid_mask]
    if len(np.unique(y)) < 2 or z.shape[0] <= len(np.unique(y)):
        return np.nan
    return float(silhouette_score(z, y))


def _logreg_accuracy_cv(latents, labels, random_seed=42):
    y, valid_mask = _encode_labels(labels)
    if y.size == 0:
        return np.nan

    z = latents[valid_mask]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.nan

    min_count = int(np.min(counts))
    if min_count < 2:
        return np.nan

    n_splits = min(5, min_count)
    clf = LogisticRegression(max_iter=5000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    if len(z.shape) > 2: # if the latent space is 2D
        z = z.reshape(z.shape[0], -1)
    scores = cross_val_score(clf, z, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def _flatten_batch(batch):
    batch_np = _to_numpy(batch)
    if batch_np.ndim == 1:
        return batch_np.reshape(1, -1)
    return batch_np.reshape(batch_np.shape[0], -1)


def _compute_pca_metrics(pca, inputs, latents, labels, dataset, valid_mask=None):
    mse = np.nan
    fc = np.nan
    if inputs.size > 0:
        recon = pca.inverse_transform(pca.transform(inputs))
        mse = _masked_mse_numpy(recon, inputs, valid_mask)
        fc = _fc_preservation_score(inputs, recon, dataset)

    sil = _silhouette(latents, labels)
    logreg_acc = _logreg_accuracy_cv(latents, labels)
    return {
        "mse": mse,
        "fc_preservation": fc,
        "silhouette": sil,
        "logreg_accuracy": logreg_acc,
    }


def eval_vae(
    model,
    data_loader,
    pca=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Run inference-time evaluation focused on reconstruction and latent-space metrics.

    Metrics:
    - MSE
    - FC preservation
    - Latent silhouette score
    - Latent logistic-regression accuracy (cross-validated)
    - PCA baseline comparison (if PCA object is provided)
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_inputs = []
    all_recons = []
    all_latents = []
    all_masks = []

    with torch.no_grad():
        for data, _ in data_loader:
            x = data.to(device)
            valid_mask = _build_valid_mask(x, data_loader.dataset)
            model_out = model(x)
            model_out = _apply_recon_mask(x, model_out, valid_mask)
            recon_x, latent = _extract_model_outputs(model_out)

            all_inputs.append(x.detach().cpu())
            all_recons.append(recon_x.detach().cpu())
            all_latents.append(latent.detach().cpu())
            if valid_mask is not None:
                all_masks.append(valid_mask.detach().cpu())

    x_all = torch.cat(all_inputs, dim=0)
    x_hat_all = torch.cat(all_recons, dim=0)
    z_all = torch.cat(all_latents, dim=0)
    valid_mask_all = torch.cat(all_masks, dim=0) if all_masks else None

    mse = _masked_mse_torch(x_hat_all, x_all, valid_mask_all)
    fc_preservation = _fc_preservation_score(x_all, x_hat_all, data_loader.dataset)

    z_np = _to_numpy(z_all)
    labels = np.asarray(getattr(data_loader.dataset, "labels", []))
    silhouette = _silhouette(z_np, labels)
    logreg_acc = _logreg_accuracy_cv(z_np, labels)

    metrics = {
        "model": {
            "mse": mse,
            "fc_preservation": fc_preservation,
            "silhouette": silhouette,
            "logreg_accuracy": logreg_acc,
        }
    }

    print("Inference metrics (model):")
    print(f"  MSE: {mse:.6f}")
    print(f"  FC preservation: {fc_preservation:.6f}" if np.isfinite(fc_preservation) else "  FC preservation: nan")
    print(f"  Silhouette: {silhouette:.6f}" if np.isfinite(silhouette) else "  Silhouette: nan")
    print(f"  Logistic regression accuracy (CV): {logreg_acc:.6f}" if np.isfinite(logreg_acc) else "  Logistic regression accuracy (CV): nan")

    if pca is not None:
        x_all = x_all.detach().cpu().numpy()
        valid_mask_np = _to_numpy(valid_mask_all) if valid_mask_all is not None else None
        z_pca = pca.transform(x_all)

        pca_metrics = _compute_pca_metrics(
            pca=pca,
            inputs=x_all,
            latents=z_pca,
            labels=labels,
            dataset=data_loader.dataset,
            valid_mask=valid_mask_np,
        )

        metrics["pca"] = pca_metrics
        metrics["comparison"] = {
            "mse_delta_model_minus_pca": metrics["model"]["mse"] - pca_metrics["mse"],
            "fc_delta_model_minus_pca": metrics["model"]["fc_preservation"] - pca_metrics["fc_preservation"],
            "silhouette_delta_model_minus_pca": metrics["model"]["silhouette"] - pca_metrics["silhouette"],
            "logreg_delta_model_minus_pca": metrics["model"]["logreg_accuracy"] - pca_metrics["logreg_accuracy"],
        }

        print("Inference metrics (PCA baseline):")
        print(f"  MSE: {pca_metrics['mse']:.6f}" if np.isfinite(pca_metrics['mse']) else "  MSE: nan")
        print(
            f"  FC preservation: {pca_metrics['fc_preservation']:.6f}"
            if np.isfinite(pca_metrics['fc_preservation'])
            else "  FC preservation: nan"
        )
        print(f"  Silhouette: {pca_metrics['silhouette']:.6f}" if np.isfinite(pca_metrics['silhouette']) else "  Silhouette: nan")
        print(
            f"  Logistic regression accuracy (CV): {pca_metrics['logreg_accuracy']:.6f}"
            if np.isfinite(pca_metrics['logreg_accuracy'])
            else "  Logistic regression accuracy (CV): nan"
        )

        print("Model vs PCA deltas (model - PCA):")
        print(f"  MSE delta: {metrics['comparison']['mse_delta_model_minus_pca']:.6f}")
        print(
            f"  FC preservation delta: {metrics['comparison']['fc_delta_model_minus_pca']:.6f}"
            if np.isfinite(metrics['comparison']['fc_delta_model_minus_pca'])
            else "  FC preservation delta: nan"
        )
        print(
            f"  Silhouette delta: {metrics['comparison']['silhouette_delta_model_minus_pca']:.6f}"
            if np.isfinite(metrics['comparison']['silhouette_delta_model_minus_pca'])
            else "  Silhouette delta: nan"
        )
        print(
            f"  Logistic regression accuracy delta: {metrics['comparison']['logreg_delta_model_minus_pca']:.6f}"
            if np.isfinite(metrics['comparison']['logreg_delta_model_minus_pca'])
            else "  Logistic regression accuracy delta: nan"
        )

    return metrics
