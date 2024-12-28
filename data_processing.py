import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pynndescent import NNDescent
from umap.umap_ import find_ab_params, fuzzy_simplicial_set


class Config:
    min_dist = 0.01
    _a, _b = find_ab_params(1.0, min_dist)
    knn_neighbors = 30


class DataPreprocessing:
    @staticmethod
    def get_weighted_knn(data, labels, metric="euclidean", n_neighbors=10):
        input_data_tensor = data
        input_labels_tensor = labels
        print("==>> input_data_tensor.shape: ", input_data_tensor.shape)
        print("==>> input_labels_tensor.shape: ", input_labels_tensor.shape)

        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))
        nnd = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True,
        )
        knn_index, knn_dist = nnd.neighbor_graph
        weighted_knn, _, _ = fuzzy_simplicial_set(
            data,
            n_neighbors=n_neighbors,
            random_state=None,
            metric="euclidean",
            metric_kwds={},
            knn_indices=knn_index,
            knn_dists=knn_dist,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            apply_set_operations=True,
            verbose=False,
            return_dists=None,
        )
        print("weighted_knn: {}".format(weighted_knn.shape))

        return weighted_knn

    @staticmethod
    def get_graph_elements(graph_, n_epochs):
        graph = graph_.tocoo()
        graph.sum_duplicates()
        n_vertices = graph.shape[1]
        if n_epochs is None:
            if graph.shape[0] <= 10000:
                n_epochs = 500
            else:
                n_epochs = 200
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()
        epochs_per_sample = n_epochs * graph.data

        head = graph.row
        tail = graph.col
        weight = graph.data

        return graph, epochs_per_sample, head, tail, weight, n_vertices

    @staticmethod
    def get_edges_with_negative_sampling(weighted_knn):
        (
            graph,
            epochs_per_sample,
            head,
            tail,
            weight,
            n_vertices,
        ) = DataPreprocessing.get_graph_elements(weighted_knn, 200)
        print("==>> epochs_per_sample: ", epochs_per_sample)
        print("==>> epochs_per_sample.shape: ", epochs_per_sample.shape)
        print("==>> graph.shape: ", graph.shape)
        print("==>> head: ", head)
        print("==>> head.shape: ", head.shape)
        print("==>> tail: ", tail)
        print("==>> tail.shape: ", tail.shape)
        print("==>> weight: ", weight)
        print("==>> weight.shape: ", weight.shape)
        print("==>> n_vertices: ", n_vertices)
        edges_to_exp, edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
        edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
        print("==>> edges_to_exp: ", edges_to_exp)
        print("==>> edges_to_exp.shape: ", edges_to_exp.shape)
        edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
        print("==>> edges_from_exp.shape: ", edges_from_exp.shape)

        return edges_to_exp, edges_from_exp, graph


def load_model(device):
    return (
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        .to(device)
        .eval()
    )


def prepare_data(expected_image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((expected_image_size, expected_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return cifar10_train, cifar10_test


def extract_features(model, dataloader, device):
    embeddings, labels = [], []
    total_processed = 0
    with torch.no_grad():
        for batch_idx, (images, lbls) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            try:
                features = model.forward_features(images)["x_norm_clstoken"]
                embeddings.append(features.cpu().numpy())
                labels.append(lbls.numpy())
                total_processed += len(lbls)
                print(
                    f"Batch {batch_idx + 1}: Processed {total_processed} samples so far."
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"CUDA out of memory: Skipping batch {batch_idx + 1}.")
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


# def save_embeddings(embeddings, labels, output_path):
#     with open(output_path, "wb") as f:
#         pickle.dump({"embeddings": embeddings, "labels": labels}, f)


# def apply_umap(embeddings):
#     return umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)


# def plot_embeddings(embedding_2d, labels, output_path):
#     plt.figure(figsize=(12, 10))
#     scatter = plt.scatter(
#         embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.6
#     )
#     plt.colorbar(scatter, ticks=range(10), label="CIFAR10 Labels")
#     plt.title("UMAP Projection of CIFAR10 Embeddings using DINOv2 (ViT-B/14)")
#     plt.xlabel("UMAP Dimension 1")
#     plt.ylabel("UMAP Dimension 2")
#     plt.savefig(output_path, dpi=300)
#     plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(device)
    expected_image_size = 224
    dataset_train, dataset_test = prepare_data(expected_image_size)
    dataloader_train = DataLoader(
        dataset_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    embeddings_train, labels_train = extract_features(model, dataloader_train, device)
    print(f"==>> embeddings_train.shape: {embeddings_train.shape}")

    dataloader_test = DataLoader(
        dataset_test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    embeddings_test, labels_test = extract_features(model, dataloader_test, device)
    print(f"==>> embeddings_test.shape: {embeddings_test.shape}")

    data = embeddings_train
    labels = labels_train

    if len(data.shape) > 2:
        data = data.reshape((data.shape[0], -1))

    weighted_knn = DataPreprocessing.get_weighted_knn(
        data, labels, n_neighbors=Config.knn_neighbors
    )

    (
        edges_to_exp,
        edges_from_exp,
        graph,
    ) = DataPreprocessing.get_edges_with_negative_sampling(weighted_knn)

    data_folder = (
        "./data/cifar10_umap_dinov2_nn_"
        + str(int(Config.knn_neighbors))
        + "train_test.pkl"
    )
    with open(data_folder, "wb") as f:
        pickle.dump(data, f)
        pickle.dump(labels, f)
        pickle.dump(weighted_knn, f)
        pickle.dump(edges_to_exp, f)
        pickle.dump(edges_from_exp, f)
        pickle.dump(graph, f)
        pickle.dump(embeddings_test, f)
        pickle.dump(labels_test, f)


if __name__ == "__main__":
    main()
