import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.preprocessed_path = preprocess_config['path']['preprocessed_path']
        self.dataset_name = preprocess_config["dataset"]
        self.ema_path = preprocess_config['path']['ema_path']
        self.energy_path = preprocess_config['path']['energy_path']
        self.pitch_path = preprocess_config['path']['pitch_path']
        self.periodicity_path = preprocess_config['path']['periodicity_path']
        self.duration_path = preprocess_config['path']['duration_path']
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.text, self.raw_text = self.process_meta(
            filename
        )
        
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        ema_path = os.path.join(
            self.ema_path,
            f"{basename}.npy",
        )
        ema = np.load(ema_path).T
        assert ema.shape[1] == 12
        pitch_path = os.path.join(
            self.pitch_path,
            f"{basename}.npy",
        )
        pitch = np.load(pitch_path)[:ema.shape[0]]
        periodicity_path = os.path.join(
            self.periodicity_path,
            f"{basename}.npy",
        )
        periodicity = np.load(periodicity_path)[:ema.shape[0]]
        energy_path = os.path.join(
            self.energy_path,
            f"{basename}.npy",
        )
        energy = np.load(energy_path)[:ema.shape[0]]
        duration_path = os.path.join(
            self.duration_path,
            f"{basename}.npy",
        )
        duration = np.load(duration_path)
        assert duration.shape[0] == phone.shape[0]

        sample = {
            "id": basename,
            "speaker": self.speaker_map[basename.split('_')[0]],
            "text": phone,
            "raw_text": raw_text,
            "ema": ema,
            "pitch": pitch,
            "periodicity": periodicity,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, t, r = line.strip("\n").split("|")
                name.append(n)
                text.append(t)
                raw_text.append(r)
            return name, text, raw_text

    def reprocess(self, data):
        idxs = range(len(data))
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        emas = [data[idx]["ema"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        periodicities = [data[idx]["periodicity"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        speakers = np.array(speakers)
        text_lens = np.array([text.shape[0] for text in texts])
        bn_lens = np.array([ema.shape[0] for ema in emas])

        texts = np.stack([np.pad(text, (0, max(text_lens) - text.shape[0])) for text in texts])
        emas = np.stack([np.pad(ema, ((0, max(bn_lens) - ema.shape[0]), (0, 0))) for ema in emas])
        pitches = np.stack([np.pad(pitch, (0, max(bn_lens) - pitch.shape[0])) for pitch in pitches])
        periodicities = np.stack([np.pad(periodicity, (0, max(bn_lens) - periodicity.shape[0])) for periodicity in periodicities])
        energies = np.stack([np.pad(energy, (0, max(bn_lens) - energy.shape[0])) for energy in energies])
        durations = np.stack([np.pad(duration, (0, max(text_lens) - duration.shape[0])) for duration in durations])
        return (
            ids,
            speakers,
            raw_texts,
            texts,
            text_lens,
            max(text_lens),
            emas,
            bn_lens,
            max(bn_lens),
            pitches,
            periodicities,
            energies,
            durations,
        )

    def collate_fn(self, data):
        # data_size = len(data)

        # if self.sort:
        #     len_arr = np.array([d["text"].shape[0] for d in data])
        #     idx_arr = np.argsort(-len_arr)
        # else:
        #     idx_arr = np.arange(data_size)

        # tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        # idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        # if not self.drop_last and len(tail) > 0:
        #     idx_arr += [tail.tolist()]

        # output = list()
        # for idx in idx_arr:
        #     output.append()

        return self.reprocess(data)


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )