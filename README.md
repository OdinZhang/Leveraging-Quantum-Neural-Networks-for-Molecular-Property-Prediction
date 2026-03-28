# QNN Project: Molecular Property Prediction on QM9

This project studies molecular property prediction on the QM9 dataset using both classical neural networks and hybrid quantum neural networks.

The codebase is organized around four experiments:

1. **Full benchmark**: compare all implemented models on the same fixed QM9 split.
2. **Classical vs QNN**: compare a classical MLP and a pooled-feature QNN under the same input representation.
3. **Representation study**: compare a QNN using mean-pooled atom features with a graph-aware QNN.
4. **Qubit scaling study**: evaluate how performance changes as the number of qubits increases.



## Recommended Workflow

### 0. Build the environment from `qnn_mol.yml`
If you already have `qnn_mol.yml`, create the Conda environment with:

```bash
# CUDA 12.4
conda env create -f qnn_mol.yml
conda activate qnn_mol
```

### 1. Create the fixed split
Run this once:

```bash
python make_split.py
```

This creates a fixed split file, for example:

```text
./splits/qm9_20000_seed42.json
```

All experiments should reuse the same split file.

### 2. Run the experiments

```bash
python exp_classical_vs_qnn.py
python exp_full_benchmark.py
python exp_representation.py
python exp_qubit_scaling.py
```

### 3. Check outputs
Each experiment writes:
- training history files (`*.json`)
- summarized result tables (`*.csv`)

Outputs are saved under the corresponding subdirectory in `./outputs/`.

---

## Default Experimental Design

- Dataset: QM9
- Subset size: 20,000 molecules
- Split: fixed train/validation/test split
- Training: AdamW + learning rate scheduler + early stopping
- Metrics: MSE, RMSE, MAE

---

## Notes

1. All comparisons are intended to be run on the **same fixed split** for fairness.
2. The quantum models may be significantly slower than the classical baselines, especially in graph-aware and larger-qubit settings.
3. If runtime becomes too high, it is recommended to first test the scripts on a smaller subset before scaling back to the full 20,000-molecule split.

---

## Example Multi-GPU Submission

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nohup python -u exp_classical_vs_qnn.py > logs/exp_classical_vs_qnn_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u exp_full_benchmark.py > logs/exp_full_benchmark_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u exp_qubit_scaling.py > logs/exp_qubit_scaling_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u exp_representation.py > logs/exp_representation_gpu3.log 2>&1 &
```



