import matplotlib.pyplot as plt
import numpy as np
import random

""" 
Util class for data vizualization
"""
class DataVizualizator():
    
    markers = ['.', ',', 'o', 'x', 'X', 'D', '*', 's', 'v', '^', '<', '>']
    colors = ['red', 'green', 'blue', 'gray', 'orange', 'purple']

    """ Public method for plotting data in scatter plot chart """
    def plot_data(X, Y, xlim = None, ylim = None):
        X_dims = np.size(X, axis = 1)
        if X_dims == 1:
            DataVizualizator.__plot_1d_data(X, Y)
        elif X_dims == 2:
            DataVizualizator.__plot_2d_data(X, Y)

    """ Private method for plotting 1D data"""
    def __plot_1d_data(X, Y, xlim = None):
        plt.scatter(X, Y, marker = 'x', color = 'blue')
        
        # Set chart axis
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        else:
            plt.xlim(np.min(X) - np.std(X), np.max(X) + np.std(X))

        # Draw plot
        plt.show()

    """ Private method for plotting 2D data (data of multiple classes) """
    def __plot_2d_data(X, Y, random_marker = 'color', xlim = None, ylim = None):
        marker_map = dict()
        if (random_marker == 'color'):
            marker_list = DataVizualizator.colors
        elif (random_marker == 'shape'):
            marker_list = DataVizualizator.markers

        # Choose unique marker for each class
        Y_classes = np.unique(Y)
        if len(Y_classes) > len(marker_list):
            raise Exception("Too many classes in Y")
        for i in Y_classes:
            marker = random.choice(marker_list)
            while marker in list(marker_map.values()):
                marker = random.choice(marker_list)
            marker_map[i] = marker
        
        # Draw each marker, i.e. Y class
        for i in Y_classes:
            x = np.squeeze(X[np.squeeze(Y == i), 0])
            y = np.squeeze(X[np.squeeze(Y == i), 1])
            plt.scatter(x, y, **{ random_marker : marker_map[i] })

        # Set chart axis
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        else:
            x = X[:, 0]
            plt.xlim(np.min(x) - np.std(x), np.max(x) + np.std(x))
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        else:
            y = X[:, 1]
            plt.ylim(np.min(y) - np.std(y), np.max(y) + np.std(y))

        # Draw plot
        plt.show()


def main():
    #x1 = np.array([[1], [2], [3]])
    #y1 = np.array([[4], [5], [6]])
    #DataVizualizator.plot_data(x1, y1)

    x2 = np.array([[1,2], [2,3], [3,4]])
    y2 = np.array([[1], [0], [1]])
    DataVizualizator.plot_data(x2, y2)

if __name__ == '__main__':
    main()