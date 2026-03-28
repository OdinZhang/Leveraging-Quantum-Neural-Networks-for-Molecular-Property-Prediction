import math
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_functional

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

def build_quantum_layer(n_qubits: int) -> TorchConnector:
    input_params = ParameterVector("x", n_qubits)
    weight_params = ParameterVector("theta", 2 * n_qubits)

    quantum_circuit = QuantumCircuit(n_qubits)

    for qubit_index in range(n_qubits):
        quantum_circuit.ry(input_params[qubit_index], qubit_index)

    for qubit_index in range(n_qubits):
        quantum_circuit.rz(weight_params[qubit_index], qubit_index)

    if n_qubits > 1:
        for qubit_index in range(n_qubits - 1):
            quantum_circuit.cx(qubit_index, qubit_index + 1)
        quantum_circuit.cx(n_qubits - 1, 0)

    for qubit_index in range(n_qubits):
        quantum_circuit.ry(weight_params[n_qubits + qubit_index], qubit_index)

    observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])

    estimator = Estimator(
        backend_options={
            "device": "GPU",
            "method": "statevector",
        },
        approximation=True,
    )

    quantum_neural_network = EstimatorQNN(
        circuit=quantum_circuit,
        estimator=estimator,
        observables=observable,
        input_params=list(input_params),
        weight_params=list(weight_params),
        input_gradients=True,
    )

    return TorchConnector(quantum_neural_network)


class MLPRegressor(torch_nn.Module):
    """
    Baseline for pooled molecular features.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class GCNRegressor(torch_nn.Module):
    """
    Classical graph baseline.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.convolution_1 = GCNConv(input_dim, hidden_dim)
        self.convolution_2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch_nn.Dropout(dropout)
        self.regression_head = torch_nn.Sequential(
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        node_features = batch.x.float()
        edge_index = batch.edge_index
        batch_index = batch.batch

        node_features = self.convolution_1(node_features, edge_index)
        node_features = torch_functional.relu(node_features)
        node_features = self.dropout(node_features)

        node_features = self.convolution_2(node_features, edge_index)
        node_features = torch_functional.relu(node_features)

        graph_features = global_mean_pool(node_features, batch_index)
        return self.regression_head(graph_features)


class GATRegressor(torch_nn.Module):
    """
    Classical graph baseline with attention.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convolution_1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.convolution_2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=True,
            dropout=dropout,
        )
        self.dropout = torch_nn.Dropout(dropout)
        self.regression_head = torch_nn.Sequential(
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        node_features = batch.x.float()
        edge_index = batch.edge_index
        batch_index = batch.batch

        node_features = self.convolution_1(node_features, edge_index)
        node_features = torch_functional.elu(node_features)
        node_features = self.dropout(node_features)

        node_features = self.convolution_2(node_features, edge_index)
        node_features = torch_functional.elu(node_features)

        graph_features = global_mean_pool(node_features, batch_index)
        return self.regression_head(graph_features)


class MeanPoolQNNRegressor(torch_nn.Module):
    """
    QNN on mean-pooled atom features.
    Input is already a fixed-dimensional pooled vector.
    """
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.classical_encoder = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, n_qubits),
        )
        self.quantum_layer = build_quantum_layer(n_qubits)
        self.regression_head = torch_nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_features = self.classical_encoder(inputs)
        encoded_features = math.pi * torch.tanh(encoded_features)
        quantum_output = self.quantum_layer(encoded_features)
        return self.regression_head(quantum_output)


class GraphQNNRegressor(torch_nn.Module):
    """
    Graph-aware QNN:
    graph -> graph encoder -> latent vector -> QNN -> regression
    """
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_qubits = n_qubits

        self.convolution_1 = GCNConv(input_dim, hidden_dim)
        self.convolution_2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch_nn.Dropout(dropout)

        self.projector = torch_nn.Sequential(
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout),
            torch_nn.Linear(hidden_dim, n_qubits),
        )

        self.quantum_layer = build_quantum_layer(n_qubits)
        self.regression_head = torch_nn.Linear(1, 1)

    def forward(self, batch) -> torch.Tensor:
        node_features = batch.x.float()
        edge_index = batch.edge_index
        batch_index = batch.batch

        node_features = self.convolution_1(node_features, edge_index)
        node_features = torch_functional.relu(node_features)
        node_features = self.dropout(node_features)

        node_features = self.convolution_2(node_features, edge_index)
        node_features = torch_functional.relu(node_features)

        graph_features = global_mean_pool(node_features, batch_index)
        projected_features = self.projector(graph_features)
        projected_features = math.pi * torch.tanh(projected_features)

        quantum_output = self.quantum_layer(projected_features)
        return self.regression_head(quantum_output)