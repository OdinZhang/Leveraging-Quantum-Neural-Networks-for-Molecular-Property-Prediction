import argparse

from model import MLPRegressor, MeanPoolQNNRegressor
from utils import (
    build_pooled_loaders,
    get_device,
    load_qm9_splits,
    save_history,
    save_results,
    set_seed,
    train_model,
)


def main(arguments):
    set_seed(arguments.seed)
    device = get_device()

    dataset, train_indices, val_indices, test_indices, target_mean, target_std = load_qm9_splits(
        root=arguments.root,
        num_samples=arguments.num_samples,
        target_index=arguments.target_index,
        seed=arguments.seed,
        split_path=arguments.split_path,
    )

    train_loader, val_loader, test_loader, input_dim = build_pooled_loaders(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        target_index=arguments.target_index,
        batch_size=arguments.batch_size,
        target_mean=target_mean,
        target_std=target_std,
    )

    model_builders = {
        "mlp_pooled": lambda: MLPRegressor(
            input_dim=input_dim,
            hidden_dim=arguments.hidden_dim,
            dropout=arguments.dropout,
        ),
        "qnn_pooled": lambda: MeanPoolQNNRegressor(
            input_dim=input_dim,
            n_qubits=arguments.n_qubits,
            hidden_dim=arguments.hidden_dim,
            dropout=arguments.dropout,
        ),
    }

    result_rows = []

    for model_name, builder in model_builders.items():
        print(f"\n===== Training {model_name} =====")
        model = builder().to(device)

        history, test_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            max_epochs=arguments.max_epochs,
            early_stopping_patience=arguments.patience,
        )

        save_history(history, f"{arguments.output_dir}/{model_name}_history.json")

        result_rows.append(
            {
                "experiment": "classical_vs_qnn",
                "model": model_name,
                "num_samples": arguments.num_samples,
                "target_index": arguments.target_index,
                "n_qubits": arguments.n_qubits if "qnn" in model_name else 0,
                **test_metrics,
            }
        )

    save_results(result_rows, f"{arguments.output_dir}/classical_vs_qnn_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/QM9")
    parser.add_argument("--output_dir", type=str, default="./outputs/representation")
    parser.add_argument("--split_path", type=str, default="./splits/qm9_20000_seed42.json")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--target_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--n_qubits", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)

    arguments = parser.parse_args()
    main(arguments)