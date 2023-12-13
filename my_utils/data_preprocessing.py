import joblib

import torch
import librosa
import numpy as np
import torch.nn.functional as F


MEMORY = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=0)
NUM_CHANNELS = 1
IMG_HEIGHT = NUM_FREQ_BINS = 195


def get_spectrogram_from_file(path: str) -> np.ndarray:
    y, fs = librosa.load(path, sr=22050)
    stft_fmax = 2093
    stft_frequency_filter_max = librosa.fft_frequencies(sr=fs, n_fft=2048) <= stft_fmax

    stft = librosa.stft(y, hop_length=512, win_length=2048, window="hann")
    stft = stft[stft_frequency_filter_max]

    stft_db = librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)
    log_stft = ((1.0 / 80.0) * stft_db) + 1.0

    return log_stft


@MEMORY.cache
def preprocess_audio(path: str) -> torch.Tensor:
    # Get spectrogram (already normalized)
    x = get_spectrogram_from_file(path)
    # Convert to PyTorch tensor
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x)  # [1, freq_bins, time_frames]
    return x


################################# CTC PREPROCESSING:


def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x, dtype="int32"):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(dtype=dtype)
    return x


def ctc_batch_preparation(batch):
    x, xl, y, yl = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl


################################# AR PREPROCESSING:


def ar_batch_preparation(batch):
    x, y = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    # Decoder input: transcript[:-1]
    y_in = [i[:-1] for i in y]
    y_in = pad_batch_transcripts(y_in, dtype="int64")
    # Decoder target: transcript[1:]
    y_out = [i[1:] for i in y]
    y_out = pad_batch_transcripts(y_out, dtype="int64")
    return x, y_in, y_out
