import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(features, labels, folder_path, title, flag):
    if flag:
        labels = labels.detach().cpu().numpy()
    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # Add title and labels
    plt.title("2D Visualization of Node Classes")
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=70, c=labels, cmap='Set2')

    # Save the plot in the dated subfolder
    plt.savefig(os.path.join(folder_path, title), bbox_inches='tight')
    plt.close()  # Close the plot to free up memory