import os
import json
import math
import random
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from my_utils.encoding_convertions import krnParser
from my_utils.data_preprocessing import (
    preprocess_audio,
    ctc_batch_preparation,
    set_pad_index,
)

DATASETS = ["quartets", "beethoven", "mozart", "haydn"]


class CTCDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
        width_reduction: int = 2,
        training_data_percentage_to_use: float = 1.,
        val_data_percentage_to_use: float = 1.,
        random_state: int = 42,
    ):
        super(CTCDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width_reduction = (
            width_reduction  # Must be overrided with that of the model!
        )
        self.training_data_percentage_to_use = training_data_percentage_to_use
        self.val_data_percentage_to_use = val_data_percentage_to_use
        self.random_state = random_state

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = CTCDataset(
                ds_name=self.ds_name,
                partition_type="train",
                width_reduction=self.width_reduction,
                use_voice_change_token=self.use_voice_change_token,
                data_percentage_to_use=self.training_data_percentage_to_use,
                random_state=self.random_state,
            )
            self.val_ds = CTCDataset(
                ds_name=self.ds_name,
                partition_type="val",
                width_reduction=self.width_reduction,
                use_voice_change_token=self.use_voice_change_token,
                data_percentage_to_use=self.val_data_percentage_to_use,
                random_state=self.random_state,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = CTCDataset(
                ds_name=self.ds_name,
                partition_type="test",
                width_reduction=self.width_reduction,
                use_voice_change_token=self.use_voice_change_token,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ctc_batch_preparation,
        )  # prefetch_factor=2

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader(self)

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_audio_len(self):
        try:
            return self.train_ds.max_audio_len
        except AttributeError:
            return self.test_ds.max_audio_len

    def get_frame_multiplier_factor(self):
        try:
            return self.train_ds.frame_multiplier_factor
        except AttributeError:
            return self.test_ds.frame_multiplier_factor


####################################################################################################


class CTCDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        width_reduction: int = 2,
        use_voice_change_token: bool = False,
        data_percentage_to_use: float = 1.,
        random_state: int = 42,
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.width_reduction = width_reduction
        self.use_voice_change_token = use_voice_change_token
        self.data_percentage_to_use = data_percentage_to_use
        self.random_state = random_state
        self.init(vocab_name="ctc_w2i")

    def init(self, vocab_name: str = "w2i"):
        # Initialize krn parser
        self.krn_parser = krnParser(use_voice_change_token=self.use_voice_change_token)

        # Check dataset name
        assert self.ds_name in DATASETS, f"Invalid dataset name: {self.ds_name}"

        # Check partition type
        assert self.partition_type in [
            "train",
            "val",
            "test",
        ], f"Invalid partition type: {self.partition_type}"

        # Get audios and transcripts files
        self.X, self.Y = self.get_audios_and_transcripts_files()

        # Reduce the size of the dataset if data_percentage_to_use != 1.
        self.max_samples = int(len(self.X) * self.data_percentage_to_use)
        np.random.seed(self.random_state)
        self.idxs_to_keep = np.random.choice(len(self.X), self.max_samples, replace=False)
        if self.data_percentage_to_use < 1:
            print(f"Reducing size of the {self.partition_type} dataset by a factor of {self.data_percentage_to_use}")
            print(f"\tOriginal {self.partition_type} dataset size: {len(self.X)}")
            print(f"\tNew {self.partition_type} dataset size: {len(self.idxs_to_keep)}")
            #TODO: log idxs_to_keep somewhere, or even better, the actual names of the files

        # Check and retrieve vocabulary
        vocab_folder = os.path.join("Quartets", "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}"
        vocab_name += "_withvc" if self.use_voice_change_token else ""
        vocab_name += ".json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()
        # Modify the global PAD_INDEX to match w2i["<PAD>"]
        set_pad_index(self.w2i["<PAD>"])

        # self.set_max_lens()  # now done in the LightningDataModule

    def __len__(self):
        return self.max_samples  # len(self.X)

    def __getitem__(self, idx):
        idx = self.idxs_to_keep[idx]  # to make it work with reduced datasets as well
        x = preprocess_audio(path=self.X[idx])
        y = self.preprocess_transcript(path=self.Y[idx])
        if self.partition_type == "train":
            # x.shape = [channels, height, width]
            return (
                x,
                (x.shape[2] // self.width_reduction)
                * self.width_reduction
                * self.frame_multiplier_factor,
                y,
                len(y),
            )
        return x, y

    def preprocess_transcript(self, path: str):
        y = self.krn_parser.convert(src_file=path)
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int32)

    def get_audios_and_transcripts_files(self):
        partition_file = f"Quartets/partitions/{self.ds_name}/{self.partition_type}.txt"

        audios = []
        transcripts = []
        with open(partition_file, "r") as file:
            for s in file.read().splitlines():
                s = s.strip()
                audios.append(f"Quartets/flac/{s}.flac")
                transcripts.append(f"Quartets/krn/{s}.krn")
        return audios, transcripts

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self):
        vocab = []
        for partition_type in ["train", "val", "test"]:
            partition_file = f"Quartets/partitions/{self.ds_name}/{partition_type}.txt"
            with open(partition_file, "r") as file:
                for s in file.read().splitlines():
                    s = s.strip()
                    transcript = self.krn_parser.convert(
                        src_file=f"Quartets/krn/{s}.krn"
                    )
                    vocab.extend(transcript)
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def set_max_lens(self):
        # Set the maximum lengths for the whole QUARTETS collection:
        # 1) Get the maximum transcript length
        # 2) Get the maximum audio length
        # 3) Get the frame multiplier factor so that
        # the frames input to the RNN are equal to the
        # length of the transcript, ensuring the CTC condition
        self.max_seq_len = 0
        self.max_audio_len = 0
        self.max_frame_multiplier_factor = 0

        # # for debugging only
        # print('\n\n\nREMOVE THIS! ctc_dataset.py (275)\n\n\n')
        # self.max_seq_len = 1009
        # self.max_audio_len = 700
        # self.max_frame_multiplier_factor = 5
        # return

        for t in tqdm(os.listdir("Quartets/krn"), desc=f"Setting max lengths ({self.ds_name})"):
            if t.endswith(".krn") and not t.startswith("."):
                # Max transcript length
                transcript = self.krn_parser.convert(
                    src_file=os.path.join("Quartets/krn", t)
                )
                self.max_seq_len = max(self.max_seq_len, len(transcript))
                # Max audio length
                audio = preprocess_audio(
                    path=os.path.join("Quartets/flac", t[:-4] + ".flac")
                )
                self.max_audio_len = max(self.max_audio_len, audio.shape[2])
                # Max frame multiplier factor
                self.max_frame_multiplier_factor = max(
                    self.max_frame_multiplier_factor,
                    math.ceil(((2 * len(transcript)) + 1) / audio.shape[2]),
                )
