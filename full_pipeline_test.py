import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import colormaps as cm
from copy import deepcopy
from sklearn.cluster import HDBSCAN
import fast_hdbscan
from time import perf_counter


from TDMS_Utilities import get_data
from TDMS_Read import TdmsReader
from filters import filter_waterfall
from SSTA import ssta
from clustering import cluster_association

from clustering import plot_clusters

def processing_pipeline(data, fs, highcut, lowcut, threshold, num):

    pipeline_start = perf_counter()

    # -------------------------
    # Filtering
    # -------------------------
    t0 = perf_counter()

    filtered_data = filter_waterfall(
        data,
        fs,
        lowcut=lowcut,
        highcut=highcut
    )

    print(f"Filtering: {perf_counter() - t0:.3f} s")

    # -------------------------
    # SSTA
    # -------------------------
    t0 = perf_counter()

    mask = ssta(
        filtered_data,
        thresh=threshold,
        num=num
    )

    SSTA_Points = np.where(mask == 1)

    print(f"SSTA: {perf_counter() - t0:.3f} s")

    # -------------------------
    # HDBSCAN Setup
    # -------------------------
    t0 = perf_counter()

    point_data = list(zip(SSTA_Points[1], SSTA_Points[0]))
    point_data = np.array(point_data)

    print(f"Point Preparation: {perf_counter() - t0:.3f} s")
    print(f"Detection Points: {len(point_data)}")

    # -------------------------
    # HDBSCAN
    # -------------------------
    t0 = perf_counter()

    hdb = fast_hdbscan.HDBSCAN(
        min_cluster_size=4,
        min_samples=None
        # copy=False,
        # n_jobs=-1
    )

    clusters = hdb.fit_predict(point_data)

    print(f"HDBSCAN: {perf_counter() - t0:.3f} s")

    # -------------------------
    # Cluster Extraction
    # -------------------------
    t0 = perf_counter()

    cluster_vals = []

    for i in range(0, max(clusters) + 1):
        cluster_vals.append([])

    tx = np.transpose(SSTA_Points)

    for point, l in zip(tx, hdb.labels_):
        cluster_vals[l].append(point)

    cluster_info = []

    for c in cluster_vals:

        c = np.transpose(c)

        info = []

        info.append(np.min(c[0]))
        info.append(np.max(c[0]))

        info.append(np.min(c[1]))
        info.append(np.max(c[1]))

        info.append(np.shape(c)[1])

        cluster_info.append(info)

    print(f"Cluster Extraction: {perf_counter() - t0:.3f} s")

    # -------------------------
    # Association
    # -------------------------
    t0 = perf_counter()

    events = cluster_association(
        deepcopy(cluster_info),
        0.25,
        40,
        1000,
        -1,
        100,
        -1,
        1000
    )

    print(f"Association: {perf_counter() - t0:.3f} s")

    print(f"Events Found: {len(events)}")

    # -------------------------
    # Total
    # -------------------------
    print(f"TOTAL PIPELINE: {perf_counter() - pipeline_start:.3f} s")
    print("-" * 50)


def pipeline_loader(folder_location,
    duration,
    native_channel_spacing,
    fs,
    start_channel,
    end_channel,
    channel_spacing,
    processing_duration,
    highcut,
    lowcut,
    num,
    threshold):

    # Data Loader - Best Time: 

    file_array = sorted(os.listdir(folder_location))[0:3]

    # 50% overlap between the sampled durations
    interval = int(processing_duration * 0.5)

    file_length = None
    data = None
    for i, file in enumerate(file_array):
        tdms = TdmsReader(f'{folder_location}{file}')
        if i == 0 and processing_duration == duration:
                # get_data(tdms, first_chan=-1, last_chan=-1, first_sample=-1, last_sample=-1, spacing=-1)

                t0 = perf_counter()
                data = get_data(tdms, start_channel, end_channel, native_spacing = native_channel_spacing, spacing = channel_spacing)
                print(f"TDMS Read: {perf_counter() - t0:.3f} s")
                
                processing_pipeline(data, fs, highcut, lowcut, threshold, num)

        elif i == 0 and processing_duration != duration:

            t0 = perf_counter()
            data = get_data(tdms, start_channel, end_channel, 0, (processing_duration * fs) - 1, native_spacing = native_channel_spacing, spacing = channel_spacing)
            print(f"TDMS Read: {perf_counter() - t0:.3f} s")
            file_read_length = int(np.shape(np.array(data))[0]/fs)

            processing_pipeline(data, fs, highcut, lowcut, threshold, num)

            while file_read_length != duration:
                
                new_data = get_data(tdms, 
                                start_channel, 
                                end_channel, 
                                file_read_length * fs, 
                                (file_read_length + interval) * fs, 
                                native_spacing= native_channel_spacing, 
                                spacing = channel_spacing
                                )

                t0 = perf_counter()
                data = np.append(data[((interval * fs) + 1)::], new_data, axis=0)
                print(f"Window Update: {perf_counter() - t0:.3f} s")

                file_read_length += interval

                processing_pipeline(data, fs, highcut, lowcut, threshold, num)

        else:
            file_read_length = 0
            while file_read_length != duration:
                
                new_data = get_data(tdms, 
                                start_channel, 
                                end_channel, 
                                file_read_length * fs, 
                                (file_read_length + interval) * fs, 
                                native_spacing= native_channel_spacing, 
                                spacing = channel_spacing
                                )

                t0 = perf_counter()
                data = np.append(data[((interval * fs) + 1)::], new_data, axis=0)
                print(f"Window Update: {perf_counter() - t0:.3f} s")

                file_read_length += interval

                processing_pipeline(data, fs, highcut, lowcut, threshold, num)


if __name__ == "__main__":
    # important parameters
    # folder_location = 'D:/1000Hz Data/FDay/'
    folder_location = 'D:/Full Pipeline Test Data/'

    # recording propeties
    duration = 10
    native_channel_spacing = 0.25
    fs = 1000

    # muting and spacing
    start_channel = int(1650)
    end_channel = int(8650)
    channel_spacing = 1

    # processing duration 
    processing_duration = 10

    # filtering
    highcut = -1
    lowcut = 100

    # SSTA
    num = 5
    threshold = 8

    pipeline_loader(folder_location, duration, native_channel_spacing, fs, start_channel, end_channel, channel_spacing, processing_duration, highcut, lowcut, num, threshold)