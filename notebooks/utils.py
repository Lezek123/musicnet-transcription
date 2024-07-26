from matplotlib import pyplot as plt
import numpy as np
from musicnet.utils import notes_vocab, note_frequency
from PIL import Image

def y_vs_y_pred_vis(y, y_pred, size=(20, 8), freq_limit=1000):
    note_frequencies = np.array([note_frequency(i) for i in range(len(notes_vocab))])
    note_frequencies = note_frequencies[note_frequencies <= freq_limit]
    y = y.T.reshape((*y.T.shape, 1))
    y_pred = y_pred.T.reshape((*y_pred.T.shape, 1))
    pixels = np.where(
        y_pred == True,
        # True positive: green, False positive: red
        np.where(y == True, np.array([0, 255, 0]), np.array([255, 0, 0])),
        # False negative: greyish-green, True negative: black
        np.where(y == True, np.array([30, 60, 30]), np.array([0, 0, 0]))
    )
    pixels = pixels.astype(np.uint8)

    Z = pixels[:len(note_frequencies), :]
    X = np.arange(Z.shape[1])
    Y = note_frequencies
    X, Y = np.meshgrid(X, Y)
    print(X.shape, Y.shape, Z.shape)
    assert(X.shape[:2] == Y.shape[:2])
    assert(Y.shape[:2] == Z.shape[:2])
    
    plt.figure(figsize=size)
    plt.gca().yaxis.tick_right()
    plt.pcolormesh(X, Y, Z)
    plt.yticks(note_frequencies)
    plt.vlines(
        np.arange(0, X.shape[1], 100),
        plt.ylim()[0],
        plt.ylim()[-1],
        linewidth=3,
        color="white",
        linestyles="dashed"
    )
    plt.show()

def spectogram_vis(spectogram, n_fft, target_sr, min_hz, size=(20, 8), freq_limit=1000):
    hz_frequencies = np.fft.rfftfreq(n_fft, 1.0 / target_sr)
    start_index = np.argmax(hz_frequencies >= min_hz)
    end_index = np.argmin(hz_frequencies <= freq_limit)

    Z = spectogram.T[:end_index-start_index, :]
    X = np.arange(Z.shape[1])
    y_ticks = hz_frequencies[start_index:end_index]
    Y = y_ticks
    X, Y = np.meshgrid(X, Y)
    assert(X.shape == Y.shape)
    assert(Y.shape == Z.shape)
    plt.figure(figsize=size)
    plt.gca().yaxis.tick_right()
    plt.pcolormesh(X, Y, Z)
    plt.vlines(
        np.arange(0, X.shape[1], 100),
        plt.ylim()[0],
        plt.ylim()[-1],
        linewidth=3,
        color="white",
        linestyles="dashed"
    )
    plt.show()