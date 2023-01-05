import matplotlib.pyplot as plt

from dataset import MusicDataset

import pretty_midi
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':

    dataset_path = os.getenv('DATASET_PATH')
    csv_file = 'maestro-v3.0.0.csv'
    csv_path = os.path.join(dataset_path, csv_file)

    df = pd.read_csv(csv_path)
    df = df['midi_filename'].T.to_dict()

    measures_per_img = 6

    for path in df.values():
        midi_path = os.path.join(dataset_path, path)
        midi = pretty_midi.PrettyMIDI(midi_path)

        piano_roll = midi.get_piano_roll(fs=50)

        num_measures = piano_roll.shape[1] // (measures_per_img * midi.get_downbeats().shape[0])
        measure_len = int(midi.get_downbeats()[1]) * 50 * measures_per_img

        for i in range(num_measures):
            plt.imshow(piano_roll[:, i * measure_len:(i + 1) * measure_len])
            plt.show()

    print('done')