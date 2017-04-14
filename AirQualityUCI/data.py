from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

headers = ["station_id","PM25_AQI_value","PM10_AQI_value","NO2_AQI_value","temperature","pressure","humidity","wind","weather"]

x_col = 1
y_col = 3
z_col = 2

steps= 5

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def scale_linear_bycolumn(rawpoints, high=1.0, low=-0.5):
    mins = np.min(rawpoints[:,1:], axis=0)
    maxs = np.max(rawpoints[:,1:], axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints[:,1:])) / rng)

def main():
    data = genfromtxt('acd\Beijing\CrawledData.txt', delimiter=',', dtype=float, skip_header=1,usecols=(0,2,3,4,5,6,7,8,9))
    print(data.shape)
    data = data[~np.isnan(data).any(axis=1)]
    data_norm = scale_linear_bycolumn(data)
    ids = np.vstack(data[:,0])
    data_norm = np.column_stack((ids,data_norm))

    x = data_norm[:, x_col]
    y = data_norm[:, y_col]
    z = data_norm[:, z_col]

    plot3d(data_norm,x,y,z)
    plot2d(data_norm,x,y)


def plot2d(data_norm,x,y):
    colors = []
    i = 0
    for station in range(1001, 1037, steps):
        plt.scatter(x[data_norm[:, 0] == station], y[data_norm[:, 0] == station], label=str(station), s=10,
                    color=generate_new_color(colors))
        i += 1
    plt.xlabel(headers[x_col])
    plt.ylabel(headers[y_col])
    plt.legend()
    plt.show()


def plot3d(data_norm,x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = []
    i = 0
    for station in range(1001, 1037, steps):
        ax.scatter(x[data_norm[:, 0] == station], y[data_norm[:, 0] == station], z[data_norm[:, 0] == station], c=generate_new_color(colors), label=str(station), marker='o')
        i += 1
    ax.set_xlabel(headers[x_col])
    ax.set_ylabel(headers[y_col])
    ax.set_zlabel(headers[z_col])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()