import gc
from sklearn.cluster import HDBSCAN
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from numpy.typing import NDArray
from copy import deepcopy
from collections import defaultdict
from numba import njit

from TDMS_Read import TdmsReader
from TDMS_Utilities import get_data
from filters import filter_waterfall
from SSTA import ssta



# class UnionFind:
#     def __init__(self, n):
#         self.parent = list(range(n))
#         self.rank = [0] * n

#     def find(self, x):
#         if self.parent[x] != x:
#             self.parent[x] = self.find(self.parent[x])  # path compression
#         return self.parent[x]

#     def union(self, a, b):
#         root_a = self.find(a)
#         root_b = self.find(b)

#         if root_a == root_b:
#             return

#         if self.rank[root_a] < self.rank[root_b]:
#             self.parent[root_a] = root_b

#         elif self.rank[root_a] > self.rank[root_b]:
#             self.parent[root_b] = root_a

#         else:
#             self.parent[root_b] = root_a
#             self.rank[root_a] += 1


def plot_clusters(data, cluster_info=None, x=None, min_width=-1, max_width=-1, num_points=-1, save=None):
    """Plots

    Keyword arguments:
        data -- A 2D array containing DAS data
        cluster_info -- A 2D array containing the properties of the clusters within the parsed data
        x -- A 2D array containing the locations of event points within the window
        min_width -- An int indicating the minimum width a cluster must have to be shown.
        -1 indicates there is no minimum width
        max_width -- An int indicating the maximum width a cluster must have to be shown.
        -1 indicates there is no maximum width
        num_points -- An int indicating the minimum number of points a cluster must be made of.
        -1 indicates there is no minimum number of points
        save -- A string indicating where the plot generated should be saved.
        No string indicated the plot should be shown and not saved.
    """

    bounds = 1000
    fig, ax = plt.subplots()
    img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
    # ax.set_xlim(1200, 1500)
    plt.ylabel('Time (seconds)')
    img1.set_cmap(cm.get_cmap('bwr'))

    if cluster_info is not None:
        rects = []
        for c in cluster_info:

            # filter out the different sizes of clusters based on what we know about our event
            # spacial presence
            if c[3] - c[2] > max_width != -1:
                continue

            if c[3] - c[2] < min_width != -1:
                continue

            # Number of points flagged
            if c[4] < num_points != -1:
                continue

            rect = patches.Rectangle((c[2], c[0]), c[3] - c[2], (c[1] - c[0]), linewidth=2, edgecolor="r", facecolor='none')
            #print(f"Width: {c[3] - c[2]}, Height: {c[1] - c[0]}, Num of Points: {c[4]}")
            rects.append(rect)

        for rect in rects:
            ax.add_patch(rect)

    if x is not None:
        ax.plot(x[1], x[0], color='black', linewidth=0.5, marker='o', markerfacecolor='red', markersize=0.5,
                linestyle='None', label="Anomaly")

    if save is not None:
        plt.savefig(save)
        plt.clf()
        plt.close("all")
        gc.collect()
    else:
        plt.show(block=False)

def extract_clusters(x):
    """Method takes in an input of a set of points and clusters them using HDBScan. The resulting clusters are then
    processed to get information about their width height and the number of points that make up the cluster.

    Keyword arguments:
        x -- An 2D array containing a set of points to be clustered
    """

    point_data = list(zip(x[1], x[0]))

    hdb = HDBSCAN(min_cluster_size= 4, min_samples=None)
    clusters = hdb.fit_predict(point_data)

    #extract cluster window information from the point dataset
    cluster_vals = []
    for i in range(0, (max(clusters) + 1)):
        cluster_vals.append([])
    tx = np.transpose(x)

    for point, l in zip(tx, hdb.labels_):
        #print(point, l)
        cluster_vals[l].append(point)

    #cluster_vals[0] = np.transpose(cluster_vals[0])
    #cluster_vals[1] = np.transpose(cluster_vals[1])
    #print(cluster_vals)
    cluster_info = []
    for c in cluster_vals:
        c = np.transpose(c)
        info = []

        #time
        info.append(np.min(c[0]))
        info.append(np.max(c[0]))

        #channel
        info.append(np.min(c[1]))
        info.append(np.max(c[1]))
        info.append(np.shape(c)[1])

        cluster_info.append(info)

    return cluster_info


@njit(inline="always")
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(inline="always")
def union(parent, rank, a, b):

    pa = find(parent, a)
    pb = find(parent, b)

    if pa == pb:
        return

    if rank[pa] < rank[pb]:
        parent[pa] = pb

    elif rank[pb] < rank[pa]:
        parent[pb] = pa

    else:
        parent[pb] = pa
        rank[pa] += 1


@njit(cache=True)
def overlap_union_find(
    clusters,
    xOff,
    yOff
):
    n = len(clusters)

    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    for i in range(n):

        a0 = clusters[i, 0]
        a1 = clusters[i, 1]
        a2 = clusters[i, 2]
        a3 = clusters[i, 3]

        for j in range(i + 1, n):

            b0 = clusters[j, 0]

            #
            # IMPORTANT:
            # since sorted by start sample,
            # once this fails we can stop
            #
            if b0 > a1 + yOff:
                break

            b1 = clusters[j, 1]
            b2 = clusters[j, 2]
            b3 = clusters[j, 3]

            #
            # channel overlap
            #
            b_mp = 0.5 * (b2 + b3)

            bx_min = min(b2, b_mp - xOff)
            bx_max = max(b3, b_mp + xOff)

            if a3 < bx_min or bx_max < a2:
                continue

            #
            # time overlap
            #
            if a1 < (b0 - yOff):
                continue

            if (b1 + yOff) < a0:
                continue

            union(parent, rank, i, j)

    return parent


def cluster_association(
    cluster_data,
    sample_overlap,
    channel_overlap,
    fs,
    min_width,
    max_width,
    min_length,
    max_length,
):

    clusters = np.asarray(
        cluster_data,
        dtype=np.float32
    )

    if len(clusters) == 0:
        return []

    #
    # width/length filtering
    #
    widths = clusters[:, 3] - clusters[:, 2]
    lengths = clusters[:, 1] - clusters[:, 0]

    keep = (
        (widths >= min_width)
        & (widths < max_width)
        & (lengths > min_length)
        & (lengths < max_length)
    )

    clusters = clusters[keep]

    if len(clusters) == 0:
        return []

    #
    # sort by start sample
    #
    order = np.argsort(clusters[:, 0])

    clusters = clusters[order]

    xOff = channel_overlap * 0.5
    yOff = sample_overlap * fs * 0.5

    parent = overlap_union_find(
        clusters,
        xOff,
        yOff
    )

    #
    # path compression
    #
    for i in range(len(parent)):
        parent[i] = find(parent, i)

    #
    # merge clusters
    #
    roots = np.unique(parent)

    merged = []

    for root in roots:

        idx = np.where(parent == root)[0]

        group = clusters[idx]

        merged.append([
            float(np.min(group[:, 0])),
            float(np.max(group[:, 1])),
            float(np.min(group[:, 2])),
            float(np.max(group[:, 3])),
            float(np.sum(group[:, 4])),
        ])

    return merged

# def cluster_association(cluster_data, sample_overlap: int, channel_overlap: int, fs: int, min_width: int, max_width: int, min_length:int, max_length:int):
#     """Groups clusters within a specific range of each other

#     Keyword arguments:
#         cluster_data -- 
#         sample_overlap -- 
#         channel_overlap -- 
#         fs -- 
#         min_width -- 
#         max_width -- 
#         min_length --
#         max_length
#     """

#     #massive clusters need filtering or they may skew the association
#     cluster_data = np.asarray(cluster_data)

#     widths = cluster_data[:,3] - cluster_data[:,2]
#     lengths = cluster_data[:,1] - cluster_data[:,0]

#     mask = (
#         (widths >= min_width) &
#         (widths < max_width) &
#         (lengths > min_length) &
#         (lengths < max_length)
# )

#     cluster_data = cluster_data[mask].tolist()

#     stable = False

#     #averaging distance for a signal
#     xOff = channel_overlap / 2

#     #fixed time diff
#     yOff = (sample_overlap * fs) / 2

#     while not stable:
#         stable = True

#         for i, a in enumerate(cluster_data):
#             for j in range(i + 1, len(cluster_data)):
#                 b = cluster_data[j]
#                 if i != j:
#                     x_overlap = False
#                     y_overlap = False

#                     #find the midpoint
#                     b_mp = b[2] + int((b[3] - b[2])/2)

#                     #midpoint extended overlap

#                     x_overlap = not (
#                         a[3] < min(b[2], b_mp - xOff) or 
#                         max(b[3], b_mp + xOff) < a[2]
#                     )

#                     #adaptive time difference
#                     #yOff = b[1] - b[0]

#                     #boundry extended overlap
#                     y_overlap = not (
#                         a[1] < (b[0] - yOff) or
#                         (b[1] + yOff) < a[0]
#                     )

#                     if x_overlap and y_overlap:
#                         stable = False

#                         cluster_data[i][0] = min(a[0], b[0])
#                         cluster_data[i][1] = max(a[1], b[1])

#                         cluster_data[i][2] = min(a[2], b[2])
#                         cluster_data[i][3] = max(a[3], b[3])

#                         #print(b)
#                         #print(copy[i+j+1])
#                         try:
#                             del cluster_data[j]
#                         except IndexError:
#                             print("Index Error")
#                         break

#             if not stable:
#                 break
#     return cluster_data

def labelled_cluster_association(cluster_data, label_data, sample_overlap, channel_overlap, fs, min_width, max_width, min_length, max_length):

    label_copy = deepcopy(label_data)
    label_associations = []

    #massive clusters need filtering or they may skew the association
    copy = []
    for c in cluster_data:
        if min_width <= c[3] - c[2] < max_width and min_length < c[1] - c[0] < max_length:
            copy.append(c)

    stable = False

    #averaging distance for a signal
    xOff = channel_overlap / 2

    #fixed time diff
    yOff = (sample_overlap * fs) / 2

    while not stable:
        stable = True

        for i, a in enumerate(copy):
            if i == len(copy):
                print("end")
            else:
                for j, b in enumerate(copy):
                    if i != j:
                        x_overlap = False
                        y_overlap = False
                        c = [0, 0, 0, 0]

                        #find the midpoint
                        b_mp = b[2] + int((b[3] - b[2])/2)

                        #midpoint extended overlap
                        if ((b_mp - xOff if b_mp - xOff < b[2] else b[2]) <= a[2] <= (b_mp + xOff if b_mp + xOff > b[3] else b[3]) <= a[3] or
                                a[2] <= (b_mp - xOff if b_mp - xOff < b[2] else b[2]) <= a[3] <= (b_mp + xOff if b_mp + xOff > b[3] else b[3]) or
                                (b_mp - xOff if b_mp - xOff < b[2] else b[2]) <= a[2] <= a[3] <= (b_mp + xOff if b_mp + xOff > b[3] else b[3]) or
                                a[2] <= (b_mp - xOff if b_mp - xOff < b[2] else b[2]) <= (b_mp + xOff if b_mp + xOff > b[3] else b[3]) <= a[3]):
                            c[2] = min([a[2], b[2]])
                            c[3] = max([a[3], b[3]])
                            x_overlap = True

                        #is it within the length of a window?
                        #yOff = b[1] - b[0]

                        #boundry extended overlap
                        if b[0] - yOff <= a[0] <= b[1] + yOff <= a[1] or a[0] <= b[0] - yOff <= a[1] <= b[1] + yOff or b[0] - yOff <= a[0] <= a[1] <= b[1] + yOff or a[0] <= b[0] - yOff <= b[1] + yOff <= a[1]:
                            c[0] = min([a[0], b[0]])
                            c[1] = max([a[1], b[1]])
                            y_overlap = True

                        if x_overlap and y_overlap:
                            stable = False

                            copy[i][0] = copy[i][0] if c[0] == 0 else c[0]
                            copy[i][1] = copy[i][1] if c[1] == 0 else c[1]
                            copy[i][2] = copy[i][2] if c[2] == 0 else c[2]
                            copy[i][3] = copy[i][3] if c[3] == 0 else c[3]
                            copy[i][4] = copy[i][4] + copy[j][4]

                            val = label_copy[j]
                            val.append(i)
                            label_associations.append(val)

                            #print(b)
                            #print(copy[i+j+1])
                            try:
                                del copy[j]
                                del label_copy[j]
                            except IndexError:
                                print("Index Error")
                            break

            if not stable:
                break

    #find the majority label of a cluster and set it to that
    label_majority=[]
    for i, l in enumerate(label_copy):
        # print(l)
        labels = {l[2]: 1}
        for la in label_associations:
            if la[3] == i:
                if la[2] in labels:
                    labels[la[2]] = labels[la[2]] + 1
                else:
                    labels[la[2]] = 1

        # print(max(labels, key=labels.get))
        label_majority.append(max(labels, key=labels.get))

    return copy, label_majority

if __name__ == '__main__':
    file_path = "Example Windows/November_Window_UTC_20231109_134947.573.tdms"

    tdms = TdmsReader(file_path)
    data = get_data(tdms)

    highcut = -1
    lowcut = 100
    filtered_data = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)

    mask = ssta(filtered_data, thresh=8, num=2)
    x = np.where(mask == 1)

    cluster_info = extract_clusters(x)

    plot_clusters(data, cluster_info, x)
    plot_clusters(data, cluster_info, x, max_width=100, num_points=100)
