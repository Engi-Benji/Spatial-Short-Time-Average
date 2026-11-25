import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import colormaps as cm
from copy import deepcopy
from sklearn.cluster import HDBSCAN
from time import perf_counter


from TDMS_Utilities import get_data
from TDMS_Read import TdmsReader
from filters import filter_waterfall
from SSTA import ssta
from clustering import cluster_association

from clustering import plot_clusters

def processing_pipeline(data, fs, highcut, lowcut, threshold, num):

    # Data Filtering  - Best Time:

    filtered_data = filter_waterfall(data, fs, lowcut=lowcut, highcut=highcut)

    # SSTA  - Best Time:
    mask = ssta(data, thresh=threshold, num=num)
    SSTA_Points = np.where(mask == 1)

    # Clustering, Association  - Best Time:

    point_data = list(zip(SSTA_Points[1], SSTA_Points[0]))
    x = SSTA_Points

    point_data = np.array(point_data)

    hdb = HDBSCAN(min_cluster_size= 4, min_samples=None)
    clusters = hdb.fit_predict(point_data)

    #extract cluster window information from the point dataset
    cluster_vals = []
    for i in range(0, (max(clusters) + 1)):
        cluster_vals.append([])
    tx = np.transpose(x)

    for point, l in zip(tx, hdb.labels_):
        cluster_vals[l].append(point)

    cluster_info = []
    for c in cluster_vals:
        c = np.transpose(c)
        info = []
        #time
        #print(np.size(c))
        info.append(np.min(c[0]))
        info.append(np.max(c[0]))

        #channel
        info.append(np.min(c[1]))
        info.append(np.max(c[1]))

        #number of points
        info.append(np.shape(c)[1])
        info.append(c)

        cluster_info.append(info)

    # association 

    events = cluster_association(deepcopy(cluster_info), 0.25, 40, 1000, -1, 100, -1, 1000)
    print(len(events))




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
        
        if i == 0 and processing_duration == duration:
                # get_data(tdms, first_chan=-1, last_chan=-1, first_sample=-1, last_sample=-1, spacing=-1)
                data = get_data(TdmsReader(f'{folder_location}{file}'), start_channel, end_channel, native_spacing = native_channel_spacing, spacing = channel_spacing)
                
                processing_pipeline(data, fs, highcut, lowcut, threshold, num)

        elif i == 0 and processing_duration != duration:

            data = get_data(TdmsReader(f'{folder_location}{file}'), start_channel, end_channel, 0, (processing_duration * fs) - 1, native_spacing = native_channel_spacing, spacing = channel_spacing)
            file_read_length = int(np.shape(np.array(data))[0]/fs)

            processing_pipeline(data)

            while file_read_length != duration:
                
                new_data = get_data(TdmsReader(f'{folder_location}{file}'), 
                                start_channel, 
                                end_channel, 
                                file_read_length * fs, 
                                (file_read_length + interval) * fs, 
                                native_spacing= native_channel_spacing, 
                                spacing = channel_spacing
                                )

                data = np.append(data[((interval * fs) + 1)::], new_data, axis=0)

                file_read_length += interval

                processing_pipeline(data, fs, highcut, lowcut, threshold, num)

        else:
            file_read_length = 0
            while file_read_length != duration:
                
                new_data = get_data(TdmsReader(f'{folder_location}{file}'), 
                                start_channel, 
                                end_channel, 
                                file_read_length * fs, 
                                (file_read_length + interval) * fs, 
                                native_spacing= native_channel_spacing, 
                                spacing = channel_spacing
                                )

                data = np.append(data[((interval * fs) + 1)::], new_data, axis=0)

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