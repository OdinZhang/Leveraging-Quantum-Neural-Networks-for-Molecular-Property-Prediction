import copy
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pandas
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader as PyGDataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def create_output_dir(path: str) -> None:
    if path != "":
        os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    create_output_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_qm9_split(
    dataset,
    num_samples: int,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict:
    total_samples = min(num_samples, len(dataset))

    generator = torch.Generator().manual_seed(seed)
    sampled_indices = torch.randperm(len(dataset), generator=generator)[:total_samples].tolist()

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_indices = sampled_indices[:train_size]
    val_indices = sampled_indices[train_size:train_size + val_size]
    test_indices = sampled_indices[train_size + val_size:]

    split_data = {
        "seed": seed,
        "num_samples": total_samples,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }
    return split_data


def get_or_create_qm9_split(
    dataset,
    split_path: str,
    num_samples: int,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict:
    if os.path.exists(split_path):
        split_data = load_json(split_path)

        expected_num_samples = min(num_samples, len(dataset))
        loaded_num_samples = (
            len(split_data["train_indices"])
            + len(split_data["val_indices"])
            + len(split_data["test_indices"])
        )
        if loaded_num_samples != expected_num_samples:
            raise ValueError(
                f"Split file {split_path} contains {loaded_num_samples} samples, "
                f"but current num_samples is {expected_num_samples}. "
                f"Please use the same num_samples or create another split file."
            )

        return split_data

    split_data = create_qm9_split(
        dataset=dataset,
        num_samples=num_samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    save_json(split_data, split_path)
    print(f"Created and saved fixed split to {split_path}")
    return split_data


def compute_target_statistics(
    dataset,
    train_indices: List[int],
    target_index: int,
) -> Tuple[float, float]:
    train_targets = []
    for index in train_indices:
        target_value = dataset[index].y[0, target_index].item()
        train_targets.append(target_value)

    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(train_targets) + 1e-8)
    return target_mean, target_std


def load_qm9_splits(
    root: str,
    num_samples: int,
    target_index: int,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_path: Optional[str] = None,
):
    dataset = QM9(root=root)

    if split_path is None:
        split_data = create_qm9_split(
            dataset=dataset,
            num_samples=num_samples,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    else:
        split_data = get_or_create_qm9_split(
            dataset=dataset,
            split_path=split_path,
            num_samples=num_samples,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    test_indices = split_data["test_indices"]

    target_mean, target_std = compute_target_statistics(
        dataset=dataset,
        train_indices=train_indices,
        target_index=target_index,
    )

    return dataset, train_indices, val_indices, test_indices, target_mean, target_std


def extract_pooled_features(
    dataset,
    indices: List[int],
    target_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_list = []
    target_list = []

    for index in indices:
        data = dataset[index]
        pooled_features = data.x.float().mean(dim=0).cpu().numpy()
        target_value = data.y[0, target_index].item()

        feature_list.append(pooled_features)
        target_list.append([target_value])

    features = np.asarray(feature_list, dtype=np.float32)
    targets = np.asarray(target_list, dtype=np.float32)
    return features, targets


def build_pooled_loaders(
    dataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    target_index: int,
    batch_size: int,
    target_mean: float,
    target_std: float,
):
    train_features, train_targets = extract_pooled_features(dataset, train_indices, target_index)
    val_features, val_targets = extract_pooled_features(dataset, val_indices, target_index)
    test_features, test_targets = extract_pooled_features(dataset, test_indices, target_index)

    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_features)
    val_features = feature_scaler.transform(val_features)
    test_features = feature_scaler.transform(test_features)

    train_targets = (train_targets - target_mean) / target_std
    val_targets = (val_targets - target_mean) / target_std
    test_targets = (test_targets - target_mean) / target_std

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_features, dtype=torch.float32),
        torch.tensor(test_targets, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = train_features.shape[1]
    return train_loader, val_loader, test_loader, input_dim


class QM9GraphSubset(Dataset):
    """
    Wrap QM9 subset and standardize only the selected target.
    """
    def __init__(
        self,
        dataset,
        indices: List[int],
        target_index: int,
        target_mean: float,
        target_std: float,
    ):
        self.dataset = dataset
        self.indices = indices
        self.target_index = target_index
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item_index):
        data = self.dataset[self.indices[item_index]].clone()
        scaled_target = (data.y[0, self.target_index] - self.target_mean) / self.target_std
        data.y = scaled_target.view(1, 1).float()
        return data


def build_graph_loaders(
    dataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    target_index: int,
    batch_size: int,
    target_mean: float,
    target_std: float,
):
    train_dataset = QM9GraphSubset(dataset, train_indices, target_index, target_mean, target_std)
    val_dataset = QM9GraphSubset(dataset, val_indices, target_index, target_mean, target_std)
    test_dataset = QM9GraphSubset(dataset, test_indices, target_index, target_mean, target_std)

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = dataset[train_indices[0]].x.shape[1]
    return train_loader, val_loader, test_loader, input_dim


def run_one_epoch(model, data_loader, optimizer, device):
    model.train()
    loss_function = torch.nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        optimizer.zero_grad()

        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            batch_size = inputs.shape[0]
        else:
            batch = batch.to(device)
            targets = batch.y
            predictions = model(batch)
            batch_size = batch.num_graphs

        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_model(model, data_loader, device, target_mean, target_std):
    model.eval()
    loss_function = torch.nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    prediction_list = []
    target_list = []

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            batch_size = inputs.shape[0]
        else:
            batch = batch.to(device)
            targets = batch.y
            predictions = model(batch)
            batch_size = batch.num_graphs

        loss = loss_function(predictions, targets)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        prediction_list.append(predictions.detach().cpu())
        target_list.append(targets.detach().cpu())

    predictions = torch.cat(prediction_list, dim=0).numpy()
    targets = torch.cat(target_list, dim=0).numpy()

    predictions_original = predictions * target_std + target_mean
    targets_original = targets * target_std + target_mean

    mse = mean_squared_error(targets_original, predictions_original)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(targets_original, predictions_original)

    metrics = {
        "scaled_loss": total_loss / total_samples,
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
    }
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    target_mean: float,
    target_std: float,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    bad_epoch_count = 0

    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = run_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device, target_mean, target_std)

        scheduler.step(val_metrics["scaled_loss"])

        current_learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_scaled_loss": float(val_metrics["scaled_loss"]),
                "val_rmse": float(val_metrics["rmse"]),
                "val_mae": float(val_metrics["mae"]),
                "learning_rate": float(current_learning_rate),
            }
        )

        if val_metrics["scaled_loss"] < best_val_loss - 1e-8:
            best_val_loss = val_metrics["scaled_loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            bad_epoch_count = 0
        else:
            bad_epoch_count += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['scaled_loss']:.6f} | "
            f"val_rmse={val_metrics['rmse']:.6f} | "
            f"lr={current_learning_rate:.2e}"
        )

        if bad_epoch_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate_model(model, test_loader, device, target_mean, target_std)
    return history, test_metrics


def save_history(history, output_path: str):
    create_output_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def save_results(rows: List[Dict], output_path: str):
    create_output_dir(os.path.dirname(output_path))
    results_frame = pandas.DataFrame(rows)
    results_frame.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")