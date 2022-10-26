import os
import pandas as pd
import pretty_midi
import numpy as np
import yaml
from yaml.loader import SafeLoader

from torch.utils.data import Dataset

from collections import defaultdict

class MusicDataset(Dataset):
    def __init__(self, dataset: str = 'Maestro', type: str = 'train', max_samples: int = None):
        # DATASET CHOICE
        dataset_path = os.getenv('DATASET_PATH')
        csv_file, csv_path = None, None
        if dataset == 'Maestro':
            csv_file = 'maestro-v3.0.0.csv'
            csv_path = os.path.join(dataset_path, csv_file)

        # READ CONFIG FILE
        self.read_config()
        self.type = type

        # GET DATAFRAME
        df = pd.read_csv(csv_path)
        if max_samples is not None:
            df = df[df['split'] == type][:max_samples].T.to_dict()
        else:
            df = df[df['split'] == type].T.to_dict()

        # READ MIDIS, GET PIANOROLLS
        i = 0
        self.data = defaultdict()
        for key in df:
            midi_path = os.path.join(dataset_path, df[key]['midi_filename'])
            midi = pretty_midi.PrettyMIDI(midi_path)

            piano_roll = midi.get_piano_roll(fs=self.fs)
            piano_roll = piano_roll / piano_roll.max()

            num_measures = piano_roll.shape[1] // midi.get_downbeats().shape[0]
            measure_len = int(midi.get_downbeats()[1]) * self.fs

            if measure_len != 100:
                continue

            for j in range(num_measures//self.measures_per_img):
                measure_img = piano_roll[:,
                                  self.measures_per_img * measure_len * j:
                                  self.measures_per_img * measure_len * (j + 1)]
                if measure_img.shape != (128, self.measures_per_img * measure_len):
                    continue

                self.data[i] = df[key].copy()
                self.data[i]['measure_img'] = measure_img
                self.data[i]['key-order'] = key, j

                # mask_idx = np.random.randint(0, self.measures_per_img)
                # mask = np.zeros((128, self.measures_per_img * measure_len))
                # mask[:, measure_len * mask_idx:measure_len * (mask_idx + 1)] = \
                #     np.ones((128, measure_len))
                # self.data[i]['mask'] = mask

                i += 1

        print(f'finished loading {self.type} dataset')

    def read_config(self):
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']

        self.fs = dataset_params['fs']
        self.measures_per_img = dataset_params['measures_per_img']

    def __getitem__(self, item):
        sample = self.data[item]

        mask_idx = np.random.randint(0, self.measures_per_img)
        mask = np.zeros((128, self.measures_per_img * 100))
        mask[:, 100 * mask_idx:100 * (mask_idx + 1)] = np.ones((128, 100))
        sample['mask'] = mask

        return sample

    def __len__(self):
        return len(self.data)