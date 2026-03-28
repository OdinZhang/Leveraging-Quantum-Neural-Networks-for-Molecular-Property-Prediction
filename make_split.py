from utils import load_qm9_splits

dataset, train_indices, val_indices, test_indices, target_mean, target_std = load_qm9_splits(
    root="./data/QM9",
    num_samples=20000,
    target_index=0,
    seed=42,
    split_path="./splits/qm9_20000_seed42.json",
)

print("Split file created.")
print(len(train_indices), len(val_indices), len(test_indices))