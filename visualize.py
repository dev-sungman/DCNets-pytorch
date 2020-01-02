import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Visualizer:
    def __init__(self, lr):
        self.model = TSNE(learning_rate=lr)

    def visualize(self, feature, labels):
        transformed = self.model.fit_transform(feature)
        
        print(transformed.shape)
        xs = transformed[:,0]
        ys = transformed[:,1]
        
        print(xs.shape, ys.shape)

        plt.scatter(xs, ys, c=labels)

        plt.show()
