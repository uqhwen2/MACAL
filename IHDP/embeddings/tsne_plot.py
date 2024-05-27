import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='sim_2.5')
args = parser.parse_args()

for query_step in range(47):
    # To reload the embedding later
    # print("Reloading the embedding from file")
    data = np.load('true{}/embedding_and_labels_{}.npz'.format(args.method, query_step))

    X_embedded, labels = data['X_embedded'], data['labels']
    # print("Embedding reloaded successfully")

    # Plot the t-SNE visualization with different colors for different datasets
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[labels == 1, 0],
                X_embedded[labels == 1, 1],
                c='b',
                label='Treated',
                alpha=0.5)
    plt.scatter(X_embedded[labels == 0, 0],
                X_embedded[labels == 0, 1],
                c='r',
                label='Control',
                alpha=0.5)

    #    plt.title('t-SNE Visualization')
    #    plt.xlabel('Latent Dimension 1')
    #    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to a PDF file
    plt.savefig('tsne/true{}/true{}_{}.pdf'.format(args.method, args.method, query_step), bbox_inches='tight')
    # plt.clf()  # or plt.cla() if you only want to clear the axes
    plt.close()
