import gc
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from numba import njit

from filters import filter_waterfall
from TDMS_Read import TdmsReader
from TDMS_Utilities import get_data


def plot_mask(data, x, save=None):
    """Plots an array of DAS data and the mask produced by the SSTA method

    Keyword arguments:
        data -- A 2D array of data pulled from a DAS tdms array.
        x -- A 2D array containing the mask to plot on the graph
        save -- A string for the directory to save the graph produced, if left blank the graph will not be saved
    """

    bounds = 1000

    fig, ax = plt.subplots()
    plt.set_cmap(plt.colormaps.get_cmap('bwr'))
    img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
    plt.ylabel('Time (seconds)')
    plt.xlabel('Distance (Channels)')
    plt.title('2023/11/09 13:49:47 - Unprocessed')
    # ax.plot(x[1], x[0], color='black', linewidth = 1, marker='o', markerfacecolor='red', markersize=1, linestyle='None', label="Anomaly")

    bounds = 1000

    fig, ax = plt.subplots()
    plt.set_cmap(plt.colormaps.get_cmap('bwr'))
    img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
    plt.ylabel('Time (seconds)')
    plt.xlabel('Distance (Channels)')
    plt.title('2023/11/09 13:57:07 - Filtered and Processed')
    ax.plot(x[1], x[0], color='black', linewidth=1, marker='o', markerfacecolor='black', markersize=1, linestyle='None',
            label="Anomaly")

    if save is not None:
        plt.savefig(save)
        plt.clf()
        plt.close("all")
        gc.collect()
    else:
        plt.show(block=False)

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def ssta(
    data,
    start_channel=0,
    end_channel=-1,
    num=50,
    thresh=2.0,
):
    """
    Fast rolling-window SSTA.

    Returns:
        -1 = not processed
         0 = no event
         1 = event

    Output dtype = int8
    """

    n_samples, n_channels = data.shape

    if end_channel == -1:
        end_channel = n_channels

    n_used = end_channel - start_channel

    # output
    result = np.full(
        (n_samples, n_channels),
        -1,
        dtype=np.int8,
    )

    # rolling channel sums
    channel_sums = np.zeros(
        n_used,
        dtype=np.float32,
    )

    global_sum = np.float32(0.0)

    # initialise first window
    for row in range(num):

        row_sum = 0.0

        for ch in range(n_used):

            val = abs(
                np.float32(
                    data[row, start_channel + ch]
                )
            )

            channel_sums[ch] += val
            row_sum += val

        global_sum += row_sum

    n_values = np.float32(num * n_used)

    # rolling window
    for i in range(num + 1, n_samples):

        new_row = i - 1
        old_row = i - num - 1

        row_add = 0.0
        row_sub = 0.0

        # update rolling sums
        for ch in range(n_used):

            add_val = abs(
                np.float32(
                    data[new_row, start_channel + ch]
                )
            )

            sub_val = abs(
                np.float32(
                    data[old_row, start_channel + ch]
                )
            )

            channel_sums[ch] += add_val - sub_val

            row_add += add_val
            row_sub += sub_val

        global_sum += row_add - row_sub

        mean = global_sum / n_values

        if mean <= 0:
            continue

        for ch in range(n_used):

            ratio = (
                channel_sums[ch] / num
            ) / mean

            if ratio > thresh:
                result[i, start_channel + ch] = 1
            else:
                result[i, start_channel + ch] = 0

    return result


def fast_ssta(
    data: NDArray[np.floating],
    start_channel: int = 0,
    end_channel: int = -1,
    num: int = 50,
    thresh: float = 2.0,
) -> NDArray[np.int8]:
    """
    Fast vectorized SSTA implementation.

    Returns:
        -1 = Not processed
         0 = No Event
         1 = Event
    """

    n_samples, n_channels = data.shape

    if end_channel == -1:
        end_channel = n_channels

    n_used_channels = end_channel - start_channel

    if n_used_channels <= 0:
        raise ValueError("end_channel must be greater than start_channel")

    if num >= n_samples:
        raise ValueError("num must be less than number of samples")

    # Absolute values once
    abs_data = np.abs(data[:, start_channel:end_channel])

    # Pad a zero row so the rolling sum includes the first window
    cumsum = np.vstack(
        (
            np.zeros((1, n_used_channels), dtype=abs_data.dtype),
            np.cumsum(abs_data, axis=0),
        )
    )

    # Rolling window sums
    window_sums = cumsum[num:] - cumsum[:-num]

    # Channel means within each window
    channel_means = window_sums / num

    # Mean across all channels
    global_means = channel_means.mean(axis=1)

    # Prevent divide-by-zero
    global_means[global_means == 0] = np.finfo(global_means.dtype).eps

    # Threshold test
    hits = (
        channel_means /
        global_means[:, None]
    ) > thresh

    # Convert to int8
    output = hits.astype(np.int8)

    # Pad back to original shape
    result = np.pad(
        output,
        (
            (num, 0),
            (
                start_channel,
                n_channels - end_channel,
            ),
        ),
        mode="constant",
        constant_values=-1,
    )

    return result

# def ssta(
#     data: NDArray[np.floating],
#     start_channel: int = 0,
#     end_channel: int = -1,
#     num: int = 50,
#     thresh: int = 2,
#     get_mask: bool = True,
# ) -> NDArray[np.floating | np.int64]:
#     """Performs the spatial short term average on an array of DAS data

#     Keyword arguments:
#         data -- A 2D array of DAS data
#         start_channel -- An int. First column (channel) of data the method will use
#         end_channel -- An int. Last column (channel of data the method will use)
#         num -- An int. Number of rows (time samples) used to perform the SSTA
#         thresh -- An int. The threshold ratio above which indicates an event has occured
#         get_mask -- A boolean flag indicating the returned arrays format:
#             True - -1 (Not processed), 0 (No Event), 1 (Event)
#             False - -1 (Not processed), ratio value
#     """

#     n_channels = data.shape[1]

#     if end_channel == -1:
#         end_channel = n_channels

#     mask: list[NDArray[np.int64] | NDArray[np.floating]] = []
#     # with tqdm(total=len(data)) as pbar:
#     for i in range(0, len(data)):
#         if not (i - num <= 0):
#             temp = np.abs(data[i - num : i, start_channel:end_channel])
#             mean = temp.mean()

#             channel_means = np.ascontiguousarray(temp.T).mean(axis=1)
#             hits = channel_means / mean > thresh

#             if get_mask:
#                 mask.append(hits.astype(np.int64))
#             else:
#                 amplitudes = np.abs(data[i, : hits.size])

#                 if not hits.any():
#                     mask.append(np.zeros(hits.size, np.int64))
#                 elif hits.all():
#                     mask.append(amplitudes)
#                 else:
#                     mask.append(
#                         np.where(hits, amplitudes, 0).astype(
#                             np.result_type(data.dtype, np.int64)
#                         )
#                     )

#             # pbar.update()

#     return np.pad(
#         np.array(mask),
#         [(num, 0), (start_channel, (n_channels - end_channel))],
#         mode="constant",
#         constant_values=-1,
#     )

if __name__ == '__main__':
    file_path = "Example Windows/November_Window_UTC_20231109_134947.573.tdms"

    tdms = TdmsReader(file_path)
    data = get_data(tdms)
    print("Data Loaded")

    highcut = -1
    lowcut = 100
    filtered_data = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)
    print("Data Filtered")

    mask = ssta(filtered_data, thresh=8, num=2, get_mask=False)
    x = np.where(mask > 0)

    plot_mask(data, x)
