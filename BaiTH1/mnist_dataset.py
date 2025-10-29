import torch
from torch.utils.data import Dataset
import numpy as np
import idx2numpy

def collate_fn(batch):
    # batch là một list các dict: [{"image": ..., "label": ...}, ...]
    images = [np.expand_dims(item["image"], axis=0) for item in batch]
    labels = [item["label"] for item in batch]

    images = np.stack(images, axis=0)  # Ghép thành 1 tensor (B, 1, 28, 28)
    labels = np.array(labels)

    return {
        "image": torch.tensor(images, dtype=torch.float32),
        "label": torch.tensor(labels, dtype=torch.long)
    }

class MNISTDataset(Dataset):
    def __init__(self, image_path: str,label_path:  str):
        images  =  idx2numpy.convert_from_file(image_path)
        labels  =  idx2numpy.convert_from_file(label_path)
        
        self.__data__  = [{
            'image':  np.array(image),
            'label': label
        } for image, label in zip(images.tolist(), labels.tolist())
        ]
        
    def __len__(self):
        return len(self.__data__) 
    
    def __getitem__(self, index: int) -> dict:
        return self.__data__[index]
    