import math

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from my_utils.ctc_dataset import CTCDataset
from my_utils.data_preprocessing import preprocess_audio, ar_batch_preparation
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION

SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token
PADDING_TOKEN = "<PAD>"  # Padding token


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
        training_data_percentage_to_use: float = 1.,
        val_data_percentage_to_use: float = 1.,
    ):
        super(ARDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_data_percentage_to_use = training_data_percentage_to_use
        self.val_data_percentage_to_use = val_data_percentage_to_use

        # to prevent executing setup() twice, once when defining the model
        #  arch and another when running trainer.fit()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    use_voice_change_token=self.use_voice_change_token,
                    data_percentage_to_use=self.training_data_percentage_to_use,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="val",
                    use_voice_change_token=self.use_voice_change_token,
                    data_percentage_to_use=self.val_data_percentage_to_use,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="test",
                    use_voice_change_token=self.use_voice_change_token,
                )
                # NOT SURE IF THE FOLLOWING IS NEEDED, CHECK CAREFULLY!
                # set max_seq_len, max_audio_len and max_frame_multiplier_factor for test_ds
                # for train_ds and val_ds it is done in LightningDataModule->set_max_lens()
                # this way we prevent computing max_lens several times
                if hasattr(self.train_ds, 'max_seq_length'):
                    self.test_ds.max_seq_length = self.train_ds.max_seq_length
                    self.test_ds.max_audio_len = self.train_ds.max_audio_len
                    self.test_ds.max_frame_multiplier_factor = self.train_ds.max_frame_multiplier_factor
                else:
                    self.test_ds.set_max_lens()
                    self.test_ds.max_seq_len += 1  # Add 1 for EOS_TOKEN

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ar_batch_preparation,
        )  # prefetch_factor=2

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,  #self.batch_size,  # 1,
            #TODO: modify following IF with self.batch_size_val
            # collate_fn=ar_batch_preparation if self.batch_size > 1 else None,  # to allow bs > 1
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,  #self.batch_size,  # 1,
            # TODO: modify following IF with self.batch_size_val
            # collate_fn=ar_batch_preparation if self.batch_size > 1 else None,  # to allow bs > 1
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader()

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

    def set_max_lens(self):
        self.train_ds.set_max_lens()
        self.train_ds.max_seq_len += 1  # Add 1 for EOS_TOKEN
        self.val_ds.max_seq_len = self.train_ds.max_seq_len
        self.val_ds.max_audio_len = self.train_ds.max_audio_len
        self.val_ds.max_frame_multiplier_factor = self.train_ds.max_frame_multiplier_factor


####################################################################################################


class ARDataset(CTCDataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        use_voice_change_token: bool = False,
        data_percentage_to_use: float = 1.,
        random_state: int = 42,
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.use_voice_change_token = use_voice_change_token
        self.data_percentage_to_use = data_percentage_to_use
        self.random_state = random_state
        self.init(vocab_name="ar_w2i")
        # self.max_seq_len += 1  # Add 1 for EOS_TOKEN <-- done in LightningDataModule->set_max_lens()

    def __getitem__(self, idx):
        idx = self.idxs_to_keep[idx]  # to make it work with reduced datasets as well
        x = preprocess_audio(path=self.X[idx])
        y = self.preprocess_transcript(path=self.Y[idx])
        # after adding the collate_fn to the val and test sets, this is no longer needed:
        # if self.partition_type == "train":
        #     return x, self.get_number_of_frames(x), y
        # return x, y
        # always return x, self.get_number_of_frames(x), y
        return x, self.get_number_of_frames(x), y

    def preprocess_transcript(self, path: str):
        y = self.krn_parser.convert(src_file=path)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

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
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i[PADDING_TOKEN] = 0
        i2w[0] = PADDING_TOKEN

        return w2i, i2w

    def get_number_of_frames(self, audio):
        # audio is the output of preprocess_audio
        # audio.shape = [1, freq_bins, time_frames]
        return math.ceil(audio.shape[1] / HEIGHT_REDUCTION) * math.ceil(
            audio.shape[2] / WIDTH_REDUCTION
        )
