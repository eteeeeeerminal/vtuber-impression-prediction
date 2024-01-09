import numpy as np
from torch.utils.data import Dataset, Subset

from .vtuber_imgs_dataset import VTuberImgDataset
from .vtuber_audios_dataset import VTuberAudioDataset
from utils.logging import get_logger

logger = get_logger(__name__)

class MySubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        x, label = self.dataset[self.indices[index]]
        if self.transform:
            x = self.transform(x)

        return x, label

    def __len__(self):
        return len(self.indices)

def split_dataset(
    dataset: VTuberImgDataset | VTuberAudioDataset,
    valid_ratio: float, test_ratio: float,
    train_transform = None, valid_transform = None, test_transform = None,
) -> tuple[MySubset, MySubset, MySubset]:

    data_len = dataset.get_video_ids_len()
    data_indices = np.array(range(data_len))
    np.random.shuffle(data_indices)

    valid_len = int(data_len * valid_ratio)
    test_len = int(data_len * test_ratio)
    train_len = data_len - valid_len - test_len
    logger.info(f"train:valid:test={train_len}:{valid_len}:{test_len}")

    train_indices, valid_indices, test_indices = np.split(data_indices, [train_len, train_len+test_len])

    train_indices = dataset.get_dataset_indices_by_video_indices(train_indices)
    valid_indices = dataset.get_dataset_indices_by_video_indices(valid_indices)
    test_indices = dataset.get_dataset_indices_by_video_indices(test_indices)

    return (
        MySubset(dataset, train_indices, train_transform),
        MySubset(dataset, valid_indices, valid_transform),
        MySubset(dataset, test_indices, test_transform)
    )
