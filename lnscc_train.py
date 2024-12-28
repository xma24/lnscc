import os
import pickle
from dataclasses import dataclass, field
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
import umap
from umap.umap_ import find_ab_params
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


@dataclass
class Config:
    """Configuration parameters for the training process."""

    expr_index: int = 3
    continuous: bool = True
    dim: int = 2
    num_samples: Optional[int] = None
    batch_size: int = 500
    negative_sample_rate: int = 10
    negative_force: float = 1.0
    min_dist: float = 0.1
    lr: float = 0.008  # Adjusted learning rate for neural network
    knn_neighbors: int = 30
    num_clusters: int = 10  # Number of clusters
    clustering_iter: int = 10000  # Iteration at which to perform K-Means clustering
    max_kmeans_attempts: int = 20  # Maximum attempts for desired K-Means
    kmeans_tolerance: float = 0.1  # Tolerance for cluster size
    desired_cluster_distribution: Optional[List[float]] = (
        None  # Desired cluster proportions
    )
    lambda_desired: float = 0.1  # Weighting factor
    lambda_: float = 0.1  # Weighting factor for  loss
    a: float = field(init=False)
    b: float = field(init=False)

    def __post_init__(self):
        self.a, self.b = find_ab_params(1.0, self.min_dist)
        self.validate_cluster_distribution()

    def validate_cluster_distribution(self):
        """Validates and normalizes the desired cluster distribution."""
        if self.desired_cluster_distribution is not None:
            if len(self.desired_cluster_distribution) != self.num_clusters:
                raise ValueError(
                    f"Length of desired_cluster_distribution ({len(self.desired_cluster_distribution)}) "
                    f"does not match num_clusters ({self.num_clusters})."
                )
            total = sum(self.desired_cluster_distribution)
            if not np.isclose(total, 1.0):
                # Normalize the distribution to sum to 1
                self.desired_cluster_distribution = [
                    float(x) / total for x in self.desired_cluster_distribution
                ]
                print("Desired cluster distribution normalized to sum to 1.")
        else:
            # Default to uniform distribution
            self.desired_cluster_distribution = [
                1.0 / self.num_clusters
            ] * self.num_clusters
            print(
                "No desired_cluster_distribution provided. Using uniform distribution."
            )


class DirectoryManager:
    """Handles the creation of necessary directories for data and work outputs."""

    DATA_FOLDER = "./data/"
    WORK_DIRS_FOLDER = "./work_dirs/"

    @staticmethod
    def create_directories():
        """Creates data and work directories if they do not exist."""
        os.makedirs(DirectoryManager.DATA_FOLDER, exist_ok=True)
        os.makedirs(DirectoryManager.WORK_DIRS_FOLDER, exist_ok=True)


class ResidualBlock(nn.Module):
    """A residual block with layer normalization and dropout."""

    def __init__(
        self, in_features: int, hidden_features: int, dropout_rate: float = 0.3
    ):
        """
        Initializes the ResidualBlock.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_features, in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (N, in_features).
        """
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.norm1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        out += residual
        out = self.activation(out)
        return out


class NeuralNetworkModel(nn.Module):
    """Enhanced Neural Network Model with residual blocks and a clustering layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        num_clusters: int,
        dropout_rate: float = 0.3,
    ):
        """
        Initializes the NeuralNetworkModel.

        Args:
            input_dim (int): Dimensionality of input features (e.g., 384).
            hidden_dims (list): List of hidden layer dimensions.
            output_dim (int): Dimensionality of the output features.
            num_clusters (int): Number of clusters for .
            dropout_rate (float): Dropout rate for regularization.
        """
        super(NeuralNetworkModel, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(ResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate))
            current_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Clustering layer: Initialize cluster centers (mu)
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, output_dim))
        nn.init.xavier_uniform_(self.cluster_layer.data)

        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NeuralNetworkModel.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 384).

        Returns:
            torch.Tensor: Output tensor of shape (N, output_dim).
        """
        # Feature extraction
        features = self.feature_extractor(x)  # Shape: (N, hidden_dim)

        # Output layer
        output = self.output_layer(features)  # Shape: (N, output_dim)

        return output


class TorchTrainer:
    """Handles the training of the PyTorch model with existing and  losses."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        graph,
        edges_to_exp,
        edges_from_exp,
        config: Config,
        data_test: np.ndarray,
        labels_test: np.ndarray,
    ):
        """
        Initializes the TorchTrainer.

        Args:
            data (np.ndarray): Input data of shape (num_samples, 384).
            labels (np.ndarray): Corresponding labels.
            graph: Weighted KNN graph.
            edges_to_exp (np.ndarray): Expanded edge sources.
            edges_from_exp (np.ndarray): Expanded edge targets.
            config (Config): Configuration parameters.
        """
        self.data = data.astype(np.float32)  # Ensure data is float32
        self.labels = labels
        self.graph = graph
        self.edges_to_exp = edges_to_exp
        self.edges_from_exp = edges_from_exp
        self.config = config
        self.data_test = data_test.astype(np.float32)  # Ensure data is float32
        self.labels_test = labels_test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.loss_fn_ = nn.KLDivLoss(reduction="batchmean")  #  loss
        self.loss_fn_existing = nn.LogSigmoid()  # Existing attraction & repellant loss

    def _initialize_model(self) -> NeuralNetworkModel:
        """
        Initializes the NeuralNetworkModel with proper initialization.

        Returns:
            NeuralNetworkModel: Initialized model.
        """
        input_dim = self.data.shape[1]  # 384
        hidden_dims = [512, 256]  # Example hidden layer sizes
        output_dim = self.config.dim  # e.g., 2 for 2D embeddings
        num_clusters = self.config.num_clusters  # e.g., 10

        model = NeuralNetworkModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_clusters=num_clusters,
        ).to(self.device)

        # Initialize weights using He initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return model

    def perform_desired_kmeans(self, features: np.ndarray) -> np.ndarray:
        """
        Performs K-Means clustering multiple times and selects the run
        whose cluster assignment distribution is closest to the desired distribution.

        Args:
            features (np.ndarray): Feature matrix of shape (num_samples, feature_dim).

        Returns:
            np.ndarray: Best cluster centers of shape (num_clusters, feature_dim).
        """
        best_score = float("inf")
        best_centers = None
        desired_distribution = np.array(self.config.desired_cluster_distribution)

        print(f"Desired Cluster Distribution: {desired_distribution}")

        for attempt in range(1, self.config.max_kmeans_attempts + 1):
            kmeans = KMeans(
                n_clusters=self.config.num_clusters, n_init=1, random_state=None
            )
            y_pred = kmeans.fit_predict(features)
            counts = np.bincount(y_pred, minlength=self.config.num_clusters)

            # Normalize counts to get distribution
            current_distribution = counts / counts.sum()

            # Compute distance between current and desired distribution
            # Using Kullback-Leibler Divergence
            # Adding a small epsilon to avoid log(0)
            epsilon = 1e-10
            kl_div = np.sum(
                desired_distribution
                * np.log(
                    (desired_distribution + epsilon) / (current_distribution + epsilon)
                )
            )

            if kl_div < best_score:
                best_score = kl_div
                best_centers = kmeans.cluster_centers_

            # Optionally, break early if a perfect match is found
            if kl_div == 0:
                break
        return best_centers

    def initialize_cluster_centers(self, data_tensor: torch.Tensor):
        """
        Initializes the cluster centers using K-Means aligned with the desired distribution.

        Args:
            data_tensor (torch.Tensor): Tensor containing all data points.
        """
        self.model.eval()
        with torch.no_grad():
            features = self.model(data_tensor).cpu().numpy()

        # Perform K-Means aligned with desired distribution
        best_centers = self.perform_desired_kmeans(features)

        # Update cluster centers
        self.model.cluster_layer.data = (
            torch.tensor(best_centers).float().to(self.device)
        )
        print("Cluster centers updated with the best desired K-Means run.")

    def convert_distance_to_log_probability(
        self, distances: torch.Tensor, a: float, b: float
    ) -> torch.Tensor:
        """
        Converts distances to log probabilities.

        Args:
            distances (torch.Tensor): Euclidean distances.
            a (float): Parameter based on min_dist.
            b (float): Parameter based on min_dist.

        Returns:
            torch.Tensor: Log probabilities.
        """
        return -torch.log1p(a * distances ** (2 * b))

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the target distribution P based on soft assignments Q.

        Args:
            q (torch.Tensor): Soft cluster assignments.

        Returns:
            torch.Tensor: Target distribution P.
        """
        weight = q**2 / q.sum(dim=0)
        p = (weight.t() / weight.sum(dim=1)).t()
        return p

    def compute_soft_assignments(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the soft assignments Q using Student's t-distribution.

        Args:
            features (torch.Tensor): Latent features of shape (N, output_dim).

        Returns:
            torch.Tensor: Soft assignments Q of shape (N, num_clusters).
        """
        # Compute squared Euclidean distance between features and cluster centers
        # features: (N, D), cluster_centers: (K, D)
        # Compute distance: (N, K)
        distance = torch.cdist(features, self.model.cluster_layer, p=2) ** 2  # (N, K)

        # Compute Student's t-distribution, degrees of freedom alpha=1
        q = 1.0 / (1.0 + distance)
        q = q ** ((1 + 1) / 2)  # alpha=1
        q = q / q.sum(dim=1, keepdim=True)  # Normalize to get soft assignments

        return q

    def desired_cluster_loss(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes a loss term that encourages cluster assignments to follow the desired distribution.

        Args:
            q (torch.Tensor): Soft cluster assignments of shape (N, K).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute the mean of q across all samples for each cluster
        q_mean = q.mean(dim=0)  # Shape: (K,)

        # Define desired distribution
        desired = (
            torch.tensor(self.config.desired_cluster_distribution)
            .float()
            .to(self.device)
        )  # Shape: (K,)

        # Ensure desired sums to 1
        desired = desired / desired.sum()

        # Compute Mean Squared Error between current mean and desired mean
        loss_desired = nn.MSELoss()(q_mean, desired)

        return loss_desired

    def train(self, epochs: int = 1, limit_iterations: int = 100000):
        """
        Trains the model using both existing and  losses, with a constraint on cluster distribution.

        Args:
            epochs (int): Number of training epochs.
        """
        num_iterations = len(self.edges_to_exp) // self.config.batch_size
        print(f"Total iterations per epoch: {num_iterations}")

        # Convert data to torch tensor and ensure it's on the correct device
        data_tensor = torch.from_numpy(self.data).float().to(self.device)

        # Initialize cluster centers using desired K-Means on initial embeddings
        # self.initialize_cluster_centers(data_tensor)

        # Begin training
        for epoch in range(epochs):
            print(f"\nStarting epoch {epoch + 1}/{epochs}")
            # Shuffle the edges at the start of each epoch
            permutation = np.random.permutation(len(self.edges_to_exp))
            edges_to_exp_shuffled = self.edges_to_exp[permutation]
            edges_from_exp_shuffled = self.edges_from_exp[permutation]

            self.model.train()
            for i in tqdm(range(num_iterations), desc=f"Epoch {epoch + 1}"):
                if i == limit_iterations:
                    print(
                        f"Limiting iterations to {limit_iterations}. Finishing training."
                    )
                    break
                if i == self.config.clustering_iter:
                    print(f"Performing desired K-Means clustering at iteration {i}.")

                    self.initialize_cluster_centers(data_tensor)

                    # Extract features from the entire dataset
                    self.model.eval()
                    with torch.no_grad():
                        features = self.model(data_tensor).cpu().numpy()

                    # Perform desired K-Means
                    desired_centers = self.perform_desired_kmeans(features)

                    # Update cluster centers
                    self.model.cluster_layer.data = (
                        torch.tensor(desired_centers).float().to(self.device)
                    )
                    print("Cluster centers updated with desired K-Means.")

                if i % 1000 == 0 and i > 0:
                    # Optionally reshuffle within the epoch
                    permutation_inner = np.random.permutation(len(self.edges_to_exp))
                    edges_to_exp_shuffled = edges_to_exp_shuffled[permutation_inner]
                    edges_from_exp_shuffled = edges_from_exp_shuffled[permutation_inner]

                self.optimizer.zero_grad()

                start = self.config.batch_size * i
                end = self.config.batch_size * (i + 1)
                batch_to_indices = edges_to_exp_shuffled[start:end]
                batch_from_indices = edges_from_exp_shuffled[start:end]

                # Fetch the corresponding data vectors
                batch_to = data_tensor[batch_to_indices]  # Shape: (batch_size, 384)
                batch_from = data_tensor[batch_from_indices]  # Shape: (batch_size, 384)

                # Forward pass
                embedding_to = self.model(batch_to)  # Shape: (batch_size, output_dim)
                embedding_from = self.model(
                    batch_from
                )  # Shape: (batch_size, output_dim)

                # Negative Sampling
                embedding_neg_to = embedding_to.repeat_interleave(
                    self.config.negative_sample_rate, dim=0
                )  # Shape: (batch_size * negative_sample_rate, output_dim)
                embedding_neg_from = embedding_from.repeat_interleave(
                    self.config.negative_sample_rate, dim=0
                )  # Shape: (batch_size * negative_sample_rate, output_dim)

                # Shuffle negative samples
                permutation_neg = torch.randperm(embedding_neg_from.size(0))
                embedding_neg_from = embedding_neg_from[permutation_neg]

                # Compute distances
                distance_positive = torch.norm(
                    embedding_to - embedding_from, dim=1
                )  # Shape: (batch_size,)
                distance_negative = torch.norm(
                    embedding_neg_to - embedding_neg_from, dim=1
                )  # Shape: (batch_size * negative_sample_rate,)

                distance = torch.cat([distance_positive, distance_negative])

                # Convert distances to log probabilities
                log_probs = self.convert_distance_to_log_probability(
                    distance, self.config.a, self.config.b
                )

                # Create target labels
                target = torch.cat(
                    [
                        torch.ones_like(distance_positive),
                        torch.zeros_like(distance_negative),
                    ]
                ).to(self.device)

                # Compute existing loss (Attraction + Repellant)
                attraction_repellant = -target * self.loss_fn_existing(log_probs)
                disconnected = (
                    -(1.0 - target)
                    * (self.loss_fn_existing(log_probs) - log_probs)
                    * self.config.negative_force
                )
                loss_existing = (attraction_repellant + disconnected).mean()

                # Compute  Loss
                # Concatenate embeddings from both batches
                embeddings = torch.cat(
                    [embedding_to, embedding_from], dim=0
                )  # Shape: (2*batch_size, output_dim)

                # Compute soft assignments Q
                q = self.compute_soft_assignments(
                    embeddings
                )  # Shape: (2*batch_size, num_clusters)

                # Compute target distribution P
                p = self.target_distribution(
                    q
                ).detach()  # Shape: (2*batch_size, num_clusters)

                # Compute log(Q)
                log_q = torch.log(q + 1e-10)  # Add epsilon to prevent log(0)

                # Compute KL divergence loss
                clustering_loss_ = self.loss_fn_(log_q, p)

                # Compute desired Cluster Loss
                loss_desired = self.desired_cluster_loss(q)

                # Combine all losses
                # You can adjust the weighting factors as needed
                lambda_ = self.config.lambda_
                lambda_desired = self.config.lambda_desired  # From Config

                if i >= int(Config.clustering_iter * 1.5):
                    total_loss = (
                        loss_existing
                        + lambda_ * clustering_loss_
                        + lambda_desired * loss_desired
                    )
                else:
                    total_loss = loss_existing

                # Backpropagation
                total_loss.backward()
                self.optimizer.step()

                # Optionally print loss
                if i % 1000 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], Iteration [{i}/{num_iterations}], "
                        f"Attraction-Repellant-Disconnected Loss: {loss_existing.item():.4f},  Clustering Loss: {clustering_loss_.item():.4f}, "
                        f"Clustering Distribution Loss: {loss_desired.item():.4f}, Total Loss: {total_loss.item():.4f}"
                    )

                # Save visualization periodically
                if i % 2500 == 0:
                    self._save_visualization_train(epoch, i)
                    self._save_visualization_test(epoch, i)

    def _save_visualization_train(self, epoch: int, iteration: int):
        """
        Saves a scatter plot of the current embeddings, evaluates clustering performance,
        and annotates the plot with NMI and ARI scores.

        Args:
            epoch (int): Current epoch.
            iteration (int): Current iteration within the epoch.
        """
        save_folder = os.path.join(
            DirectoryManager.WORK_DIRS_FOLDER,
            f"{self.config.expr_index}_lr_{self.config.lr}",
        )
        os.makedirs(save_folder, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            # Generate embeddings for all data points
            data_tensor = torch.from_numpy(self.data).float().to(self.device)
            embeddings = self.model(data_tensor).cpu().numpy()

            # Compute soft assignments Q
            embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
            q = (
                self.compute_soft_assignments(embeddings_tensor).cpu().numpy()
            )  # Shape: (N, K)

            # Assign clusters based on highest soft assignment
            cluster_assignments = np.argmax(q, axis=1)  # Shape: (N,)

            # Compute NMI and ARI
            nmi = normalized_mutual_info_score(self.labels, cluster_assignments)
            ari = adjusted_rand_score(self.labels, cluster_assignments)

            print(
                f"Train: Visualization at Epoch {epoch + 1}, Iteration {iteration}: NMI={nmi:.4f}, ARI={ari:.4f}"
            )

            # Apply UMAP for dimensionality reduction if needed
            if self.config.dim > 2:
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings  # Already 2D

            # Create scatter plot colored by cluster assignments
            plt.figure(figsize=(11.7, 8.27))
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=cluster_assignments,
                s=5,
                cmap="tab10",
                alpha=0.6,
            )
            plt.colorbar(scatter, ticks=range(self.config.num_clusters))
            plt.title(
                f"Epoch {epoch + 1}, Iteration {iteration}\nNMI: {nmi:.4f}, ARI: {ari:.4f}"
            )

            # Save the figure
            plt.savefig(
                os.path.join(save_folder, f"train_epoch_{epoch}_iter_{iteration}.png")
            )
            plt.close()

    def _save_visualization_test(self, epoch: int, iteration: int):
        """
        Saves a scatter plot of the current embeddings, evaluates clustering performance,
        and annotates the plot with NMI and ARI scores.

        Args:
            epoch (int): Current epoch.
            iteration (int): Current iteration within the epoch.
        """
        save_folder = os.path.join(
            DirectoryManager.WORK_DIRS_FOLDER,
            f"{self.config.expr_index}_lr_{self.config.lr}",
        )
        os.makedirs(save_folder, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            # Generate embeddings for all data points
            data_tensor = torch.from_numpy(self.data_test).float().to(self.device)
            embeddings = self.model(data_tensor).cpu().numpy()

            # Compute soft assignments Q
            embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
            q = (
                self.compute_soft_assignments(embeddings_tensor).cpu().numpy()
            )  # Shape: (N, K)

            # Assign clusters based on highest soft assignment
            cluster_assignments = np.argmax(q, axis=1)  # Shape: (N,)

            # Compute NMI and ARI
            nmi = normalized_mutual_info_score(self.labels_test, cluster_assignments)
            ari = adjusted_rand_score(self.labels_test, cluster_assignments)

            print(
                f"Test: Visualization at Epoch {epoch + 1}, Iteration {iteration}: NMI={nmi:.4f}, ARI={ari:.4f}"
            )

            # Apply UMAP for dimensionality reduction if needed
            if self.config.dim > 2:
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings  # Already 2D

            # Create scatter plot colored by cluster assignments
            plt.figure(figsize=(11.7, 8.27))
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=cluster_assignments,
                s=5,
                cmap="tab10",
                alpha=0.6,
            )
            plt.colorbar(scatter, ticks=range(self.config.num_clusters))
            plt.title(
                f"Epoch {epoch + 1}, Iteration {iteration}\nNMI: {nmi:.4f}, ARI: {ari:.4f}"
            )

            # Save the figure
            plt.savefig(
                os.path.join(save_folder, f"test_epoch_{epoch}_iter_{iteration}.png")
            )
            plt.close()


def main():
    """Main function to execute the training pipeline."""
    # Initialize configuration
    config = Config(
        desired_cluster_distribution=[
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]  # Example for 10 clusters
        # If desired_cluster_distribution is None, uniform distribution will be used
    )

    # Create necessary directories
    DirectoryManager.create_directories()

    # Load preprocessed data
    data_path = os.path.join(
        DirectoryManager.DATA_FOLDER,
        f"cifar10_umap_dinov2_nn_{config.knn_neighbors}train_test.pkl",
    )
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        labels = pickle.load(f)
        weighted_knn = pickle.load(f)
        edges_to_exp = pickle.load(f)
        edges_from_exp = pickle.load(f)
        graph = pickle.load(f)
        data_test = pickle.load(f)
        labels_test = pickle.load(f)

    if config.num_samples is None:
        config.num_samples = data.shape[0]

    print(f"Data type: {type(data)}, Shape: {data.shape}")

    # Initialize and train the model
    trainer = TorchTrainer(
        data=data,
        labels=labels,
        graph=graph,
        edges_to_exp=edges_to_exp,
        edges_from_exp=edges_from_exp,
        config=config,
        data_test=data_test,
        labels_test=labels_test,
    )
    trainer.train(epochs=1, limit_iterations=50000)


if __name__ == "__main__":
    main()
