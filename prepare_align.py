import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts
import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

# def main(config):
#     if "LJSpeech" in config["dataset"]:
#         ljspeech.prepare_align(config)
#     if "AISHELL3" in config["dataset"]:
#         aishell3.prepare_align(config)
#     if "LibriTTS" in config["dataset"]:
#         libritts.prepare_align(config)

def prepare_align():
    wav_dir = '/data/ddsp_data/LibriTTS_R/wavs'
    raw_text_dir = '/data/ddsp_data/LibriTTS_R/txts'
    out_dir = '/data/nzxyin/LibriTTS_R_preprocessed'
    sampling_rate = 16000
    max_wav_value = 32768.0
    cleaners = ["english_cleaners"]
    all_filenames = {filename[:-4] for filename in os.listdir(wav_dir) if filename[-4:] == '.wav'} & {filename[:-4] for filename in os.listdir(raw_text_dir) if filename[-4:] == '.txt'}
    for filename in tqdm(all_filenames):
        text_path = os.path.join(raw_text_dir, f"{filename}.txt")
        wav_path = os.path.join(wav_dir, f"{filename}.wav")
        speaker = filename.split('_')[0]
        with open(text_path) as f:
            text = f.readline().strip("\n")
        text = _clean_text(text, cleaners)

        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        wav, _ = librosa.load(wav_path, sr=sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        wavfile.write(
            os.path.join(out_dir, speaker, f"{filename}.wav"),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(
            os.path.join(out_dir, speaker, f"{filename}.lab"),
            "w",
        ) as f1:
            f1.write(text)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, help="path to preprocess.yaml")
    # args = parser.parse_args()

    # config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    # main(config)
    prepare_align()
