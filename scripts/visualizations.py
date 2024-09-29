import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def create_heatmap(topic_counts, metric, num_topics):
    # Prepare data for heatmap
    categories = list(next(iter(topic_counts.values()))[metric].keys())
    data = np.zeros((len(categories), num_topics))

    for topic in range(num_topics):
        for i, category in enumerate(categories):
            data[i, topic] = topic_counts[topic][metric][category]

    # Normalize data
    data = data / data.sum(axis=0)[np.newaxis, :]

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=range(num_topics),
        yticklabels=categories,
    )
    plt.title(f"Heatmap of {metric} Distribution Across Topics")
    plt.xlabel("Topics")
    plt.ylabel("Categories")
    plt.tight_layout()
    plt.savefig(f"./temp_plots/heatmap_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()


def prepare_data_for_tsne(topic_counts, metric):
    X = []
    y = []
    for topic, data in topic_counts.items():
        topic_distribution = [data[metric][cat] for cat in data[metric]]
        X.append(topic_distribution)
        y.append(
            max(data[metric], key=data[metric].get)
        )  # Use the most common category as the label
    return np.array(X), np.array(y)


def create_tsne_visualization(topic_counts, metric, num_topics):
    X, y = prepare_data_for_tsne(topic_counts, metric)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    categories = list(set(y))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    for cat, color in zip(categories, colors):
        mask = y == cat
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], label=cat, alpha=0.7)

    # Add topic numbers as annotations
    for i, (x, y) in enumerate(X_tsne):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")

    plt.title(f"t-SNE Visualization of Topics for {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./temp_plots/tsne_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_topics(topic_counts, metrics, num_topics):
    for metric in metrics:
        create_heatmap(topic_counts, metric, num_topics)
        create_tsne_visualization(topic_counts, metric, num_topics)


# Call this function in your main script
# visualize_topics(topic_counts, ['confidentialityImpact', 'integrityImpact', 'availabilityImpact'], num_topics)
