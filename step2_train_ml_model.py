import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import json
import math
import time
import os

from util import NodeFeatureIndices, get_directory_path
import step1_generate_dataset

class TransformerGraphNN(nn.Module):
    """Transformer-based graph neural network with graph-restricted attention."""

    def __init__(
        self,
        in_features=NodeFeatureIndices.counter,
        embedding_dim=256,
        num_heads=8,
        num_layers=8,
        ff_dim=None,
        readout_dim=2,
        dropout=0.1,
        device=None,
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embedding_dim

        # Project input node features into the Transformer embedding space.
        self.input_proj = nn.Linear(
            in_features,
            embedding_dim,
            bias=False,
        )

        # Transformer encoder used for graph-restricted self-attention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            device=device,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Map the pooled graph representation to an output.
        self.readout = nn.Linear(embedding_dim, readout_dim)

    def forward(self, x, edges):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features with shape (B, N, in_features).

        edges : torch.Tensor
            Graph edges with shape (E, 2), where each row contains
            [source, destination].

        Returns
        -------
        torch.Tensor
            Scalar output for each graph with shape (B,T).
        """

        num_batch, num_nodes, _ = x.shape

        # Project node features into the Transformer embedding space.
        node_embeddings = self.input_proj(x) # (B, N, D)

        # Create a mask that initially blocks all node-to-node attention.
        # In PyTorch Transformer semantics:
        #   True  -> attention is blocked
        #   False -> attention is allowed.
        attention_mask = torch.ones(
            num_nodes,
            num_nodes,
            dtype=torch.bool,
            device=x.device,
        )

        # Allow each node to attend to itself.
        attention_mask.fill_diagonal_(False)

        # Allow attention between connected nodes.
        if edges.numel() > 0:
            edges = edges.to(
                device=x.device,
                dtype=torch.long,
            )

            src, dst = edges.T

            # Treat the graph as undirected.
            attention_mask[src, dst] = False
            attention_mask[dst, src] = False

        # Apply graph-restricted self-attention.
        node_embeddings = self.transformer(
            node_embeddings,
            mask=attention_mask,
        ) # (B, N, D)

        # Obtain a graph-level representation by mean pooling for each batch.
        graph_embedding = node_embeddings.mean(dim=1)  # (B, D)

        # Produce one prediction per graph.
        output = self.readout(graph_embedding) # (B, readout_dim)

        return output # (B, readout_dim)


def model_minibatch(
        minibatch_node_features: List[torch.Tensor], 
        minibatch_edges: List[torch.Tensor], 
        minibatch_targets: List[torch.Tensor], 
        optimizer, 
        criterion, 
        model: TransformerGraphNN, 
        update_model=True
    ):
    sample_count = len(minibatch_node_features)
    if update_model:
        model.train()
        optimizer.zero_grad() # Reset gradients
    else:
        model.eval()

    mean_loss = 0

    for sample in range(sample_count):
        node_features = minibatch_node_features[sample]
        edges = minibatch_edges[sample]
        target = minibatch_targets[sample]
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0) # Make sure node features are passed in batch format

        prediction: torch.Tensor = model.forward(node_features, edges)
        prediction = prediction.squeeze(0)

        # Compute the loss
        loss = criterion(
            prediction,
            target,
        )
        loss_for_backward = loss / sample_count
        mean_loss += loss_for_backward.item()

        # Compute gradients, which will be stored in the optimizer
        if update_model:
            loss_for_backward.backward()

    if update_model:
        # Prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        # Update model parameters
        optimizer.step()

    return mean_loss


# Pre-Process the samples by adding a virtual node, applying log1p to features, and normalizing features
def data_graph_pre_processing(samples_node_features: List[torch.Tensor], samples_edges: List[torch.Tensor], normalization):
    device = samples_node_features[0].device
    virtual_node_features = torch.zeros(
        NodeFeatureIndices.counter,
        dtype=samples_node_features[0].dtype,
        device=device,
    )
    virtual_node_features[NodeFeatureIndices.is_virtual_node] = 1

    node_feature_mean = torch.tensor(normalization["node_feature_mean"], device=device)
    node_feature_std = torch.tensor(normalization["node_feature_std"], device=device)

    sample_count = len(samples_node_features)

    for sample in range(sample_count):
        # Add virtual node
        virtual_node = virtual_node_features.unsqueeze(0)  # [1, num_features]

        samples_node_features[sample] = torch.cat(
            [samples_node_features[sample], virtual_node],
            dim=0,
        )
        node_count = samples_node_features[sample].shape[0]
        virtual_node_edges = torch.tensor([[node_count-1, i] for i in range(node_count-1)], device=device)
        samples_edges[sample] = torch.cat(
            [samples_edges[sample], virtual_node_edges],
            dim=0,
        )

        # Apply log(1 + x)
        samples_node_features[sample] = torch.log1p(
            samples_node_features[sample]
        )

        # Apply normalization
        samples_node_features[sample] = (
            samples_node_features[sample] - node_feature_mean
        ) / node_feature_std


def data_target_pre_processing(samples_targets: List[torch.Tensor], normalization):
    sample_count = len(samples_targets)

    target_mean = torch.tensor(normalization["target_mean"], device=device)
    target_std = torch.tensor(normalization["target_std"], device=device)

    for sample in range(sample_count):
        samples_targets[sample] = torch.log1p(
            samples_targets[sample]
        )

        samples_targets[sample] = (
            samples_targets[sample] - target_mean
        ) / target_std


def data_pre_processing(samples_node_features: List[torch.Tensor], samples_edges: List[torch.Tensor], samples_targets: List[torch.Tensor], normalization):
    data_graph_pre_processing(samples_node_features, samples_edges, normalization)
    data_target_pre_processing(samples_targets, normalization)


class DataSampler:
    def __init__(self, file_path: str):
        self.file_path = file_path

        # Store byte offsets where each line starts
        self.newline_index = []

        with open(self.file_path, "rb") as f:
            while True:
                position = f.tell()
                line = f.readline()

                if not line:
                    break

                self.newline_index.append(position)

    def sample_data(self, count, device):
        samples_node_features = []
        samples_edges = []
        samples_targets = []

        # Sample line indices without replacement
        indices = np.random.choice(
            len(self.newline_index),
            size=min(count, len(self.newline_index)),
            replace=False
        )

        with open(self.file_path, "rb") as f:
            for index in indices:
                f.seek(self.newline_index[index])
                random_file_line = f.readline().decode("utf-8").strip()
                data = json.loads(random_file_line)

                target_time = 0.1 * 10**(np.random.random() * math.log10(50))
                node_features = torch.tensor(data["node_features"], device=device)
                node_features[:, NodeFeatureIndices.time] = target_time
                samples_node_features.append(node_features)

                edges = torch.tensor(data["edges"], device=device)
                samples_edges.append(edges)

                time_data: List[float] = data["time_data"]
                lift_data: List[float] = data["lift_data"]
                air_cons_data: List[float] = data["cumulative_air_cons"]

                # Interpolate values at target_time
                lift = min(max(0, np.interp(target_time, time_data, lift_data)), 10000)
                air_cons = min(max(0, np.interp(target_time, time_data, air_cons_data)), 10000)
                target = torch.tensor([lift, air_cons], device=device)
                samples_targets.append(target)
                
        return samples_node_features, samples_edges, samples_targets


def get_normalization(model_directory_path: str, training_file_path: str):
    normalization_path = model_directory_path + "/norm.json"
    if not os.path.exists(normalization_path):
        train_sampler = DataSampler(training_file_path)
        
        print("Generating normalization parameters")
        samples_node_features, samples_edges, samples_targets = train_sampler.sample_data(count=1_000, device=device)
        normalization = {
            "node_feature_mean": [0 for i in range(NodeFeatureIndices.counter)],
            "node_feature_std": [1 for i in range(NodeFeatureIndices.counter)],
            "target_mean": [0, 0],
            "target_std": [1, 1],
        }
        data_pre_processing(samples_node_features, samples_edges, samples_targets, normalization)
        node_features = torch.cat(samples_node_features, dim=0)
        print(node_features.shape)
        normalization["node_feature_mean"] = (
            node_features.mean(dim=0).cpu().tolist()
        )
        normalization["node_feature_std"] = (
            node_features.std(dim=0, unbiased=False).cpu().tolist()
        )
        targets = torch.stack(samples_targets)
        print(targets.shape)
        normalization["target_mean"] = targets.mean(dim=0).cpu().tolist()
        normalization["target_std"] = targets.std(dim=0, unbiased=False).cpu().tolist()

        normalization["node_feature_std"] = [
            max(x, 1e-8) for x in normalization["node_feature_std"]
        ]
        normalization["target_std"] = [
            max(x, 1e-8) for x in normalization["target_std"]
        ]

        with open(normalization_path, "w") as f:
            json.dump(normalization, f, indent=2)

    else:
        with open(normalization_path, "r") as f:
            normalization = json.loads(f.read())

    return normalization


def get_model_data(checkpoint_path: str, device):
    criterion = nn.L1Loss()
    model: TransformerGraphNN = TransformerGraphNN().to(device)
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )
    interval_train_loss = list()
    interval_eval_loss = list()
    best_eval_loss = float("inf")

    if os.path.exists(checkpoint_path):
        data = torch.load(
            checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        interval_train_loss = data.get("interval_train_loss")
        interval_eval_loss = data.get("interval_eval_loss")
        best_eval_loss = data.get("best_eval_loss")
        
    
    return model, optimizer, criterion, interval_train_loss, interval_eval_loss, best_eval_loss

def save_model_data(
        checkpoint_path: str, 
        model: TransformerGraphNN, 
        optimizer: torch.optim.AdamW, 
        criterion: nn.L1Loss, 
        interval_train_loss: list, 
        interval_eval_loss: list, 
        best_eval_loss: float):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "interval_train_loss": interval_train_loss,
        "interval_eval_loss": interval_eval_loss,
        "best_eval_loss": best_eval_loss
    }

    torch.save(
        data,
        checkpoint_path
    )


class LossPlotter:
    def __init__(self, window_size=10):
        self.window_size = window_size

        # Interactive mode: don't block the training program.
        plt.ion()

        self.figure, self.ax = plt.subplots(figsize=(10, 6))

        # Create the lines once. We only update their data later.
        self.train_loss_line, = self.ax.plot(
            [], [],
            label="Train Loss",
            alpha=0.3,
        )

        self.eval_loss_line, = self.ax.plot(
            [],
            [],
            label="Eval Loss",
            alpha=0.3,
        )

        self.train_avg_line, = self.ax.plot(
            [],
            [],
            label=f"Train Loss ({window_size}-point avg)",
        )

        self.eval_avg_line, = self.ax.plot(
            [],
            [],
            label=f"Eval Loss ({window_size}-point avg)",
        )

        self.ax.set_xlabel("Evaluation steps")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Eval Loss")
        self.ax.legend()

        self.ax.minorticks_on()

        self.ax.grid(
            True,
            axis="y",
            which="major",
            alpha=0.7,
        )

        self.ax.grid(
            True,
            axis="y",
            which="minor",
            alpha=0.3,
        )

        self.figure.tight_layout()

    def _rolling_average(self, data):
        result = np.empty(len(data))

        for i in range(len(data)):
            start = max(0, i - self.window_size + 1)
            result[i] = np.mean(data[start:i + 1])

        return result

    def update(self, train_loss, eval_loss):
        """
        Update the plot without blocking the program.

        Parameters
        ----------
        train_loss : sequence
            Training loss history.
        eval_loss : sequence
            Evaluation/eval loss history.
        """
        train_loss = np.asarray(train_loss)
        eval_loss = np.asarray(eval_loss)

        train_loss_avg = self._rolling_average(train_loss)
        eval_loss_avg = self._rolling_average(eval_loss)

        x_train = np.arange(len(train_loss))
        x_eval = np.arange(len(eval_loss))

        self.train_loss_line.set_data(x_train, train_loss)
        self.eval_loss_line.set_data(x_eval, eval_loss)

        self.train_avg_line.set_data(x_train, train_loss_avg)
        self.eval_avg_line.set_data(x_eval, eval_loss_avg)

        # Make sure the axes fit the new data.
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw without blocking the training loop.
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def save(self, filename, dpi=150):
        """Save the current plot as an image."""
        self.figure.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )

    def close(self):
        """Close the plot window."""
        plt.close(self.figure)


def format_duration(seconds):
    days = seconds // (24 * 60 * 60)
    hours = (seconds % (24 * 60 * 60)) // (60 * 60)
    minutes = (seconds % (60 * 60)) // 60
    seconds = seconds % 60

    parts = []

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"


def train_model(model_directory_path: str, target_sample_count: int, training_file_path: str, eval_file_path: str, image_directory_path: str, device):
    os.makedirs(model_directory_path, exist_ok=True)

    train_sampler = DataSampler(training_file_path)
    eval_sampler = DataSampler(eval_file_path)

    normalization = get_normalization(model_directory_path, training_file_path)
    print(normalization)

    checkpoint_path = model_directory_path + "/model.checkpoint"
    best_checkpoint_path = model_directory_path + "/best_model.checkpoint"
    model, optimizer, criterion, interval_train_loss, interval_eval_loss, best_eval_loss = get_model_data(checkpoint_path, device)

    training_interval_sample_size = 2 ** 14
    training_interval_eval_size = 2 ** 12
    minibatch_sample_size = 16
    target_interval_count = int(target_sample_count / training_interval_sample_size)

    loss_plotter = LossPlotter()
    loss_plotter.update(interval_train_loss, interval_eval_loss)
    training_interval_counter = len(interval_train_loss)

    while(training_interval_counter < target_interval_count):
        start_time = time.time()
        np.random.seed(training_interval_counter)
        train_node_features, train_edges, train_targets = train_sampler.sample_data(count=training_interval_sample_size, device=device)
        data_pre_processing(train_node_features, train_edges, train_targets, normalization)

        assert training_interval_sample_size % minibatch_sample_size == 0
        interval_count = int(training_interval_sample_size / minibatch_sample_size)
        mean_train_loss = 0
        for i in range(interval_count):
            if i % 128 == 0:
                print(training_interval_counter, "/", target_interval_count, ": ", i, "/", interval_count)
            train_loss = model_minibatch(
                minibatch_node_features=train_node_features[i*minibatch_sample_size: (i+1)*minibatch_sample_size],
                minibatch_edges=train_edges[i*minibatch_sample_size: (i+1)*minibatch_sample_size],
                minibatch_targets=train_targets[i*minibatch_sample_size: (i+1)*minibatch_sample_size],
                optimizer=optimizer,
                criterion=criterion,
                model=model,
                update_model=True
            )
            mean_train_loss += train_loss / interval_count

        eval_node_features, eval_edges, eval_targets = eval_sampler.sample_data(count=training_interval_eval_size, device=device)
        data_pre_processing(eval_node_features, eval_edges, eval_targets, normalization)
        mean_eval_loss = model_minibatch(
            minibatch_node_features=eval_node_features,
            minibatch_edges=eval_edges,
            minibatch_targets=eval_targets,
            optimizer=optimizer,
            criterion=criterion,
            model=model,
            update_model=False
        )

        interval_train_loss.append(mean_train_loss)
        interval_eval_loss.append(mean_eval_loss)
        if best_eval_loss > mean_eval_loss:
            best_eval_loss = mean_eval_loss
            save_model_data(
                best_checkpoint_path, 
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                interval_train_loss=interval_train_loss,
                interval_eval_loss=interval_eval_loss,
                best_eval_loss=best_eval_loss
            )

        save_model_data(
            checkpoint_path, 
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            interval_train_loss=interval_train_loss,
            interval_eval_loss=interval_eval_loss,
            best_eval_loss=best_eval_loss
        )

        loss_plotter.update(train_loss=interval_train_loss, eval_loss=interval_eval_loss)
        training_interval_counter = len(interval_train_loss)
        interval_time = time.time() - start_time
        remaining_seconds = interval_time * (target_interval_count- training_interval_counter)
        print("interval_time_s", interval_time)
        print("estimated time left: ", format_duration(remaining_seconds))

    os.makedirs(image_directory_path, exist_ok=True)
    loss_plotter.save(image_directory_path + "/loss_plot")
    best_model, best_model_optimizer, best_model_criterion, best_model_interval_train_loss, best_model_interval_eval_loss, best_eval_loss = get_model_data(best_checkpoint_path, device)
    print("best model index: ", len(best_model_interval_eval_loss))


model_directory_path = get_directory_path(__file__) + "/model"
image_directory_path = get_directory_path(__file__) + "/images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_step2_train_ml_model():
    train_model(
        model_directory_path,
        target_sample_count=2**24,
        training_file_path=step1_generate_dataset.training_file_path,
        eval_file_path=step1_generate_dataset.eval_file_path,
        image_directory_path=image_directory_path,
        device=device
    )

if __name__ == "__main__":
    run_step2_train_ml_model()