import torch
import os
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

def collate_fn(samples: list[dict]) -> dict:
    images = [s['image'].unsqueeze(0) for s in samples]
    labels = [s['label'] for s in samples]
    images = torch.cat(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"image": images, "label": labels}


class VinafoodDataset(Dataset):
    def __init__(self, image_path: str, image_size=(128, 128)):
        super().__init__()
        self.image_size = image_size
        self.label2idx = {}
        self.idx2label = {}
        self.data = self._load_data(image_path)
        print(f"Số lớp: {len(self.label2idx)}")
        print(f"Các lớp: {self.label2idx}")

    def _load_data(self, path):
        data = []
        label_id = 0

        for folder in sorted(os.listdir(path)):
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                self.idx2label[label_id] = label
                label_id += 1

            folder_path = os.path.join(path, folder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                # Đọc ảnh
                if img_file.endswith('.npy'):
                    img = np.load(img_path)
                else:
                    img = cv.imread(img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img, self.image_size)
                    img = img.astype(np.float32) / 255.0

                img_tensor = torch.tensor(img).permute(2, 0, 1)  # (C, H, W)
                data.append({"image": img_tensor, "label": self.label2idx[label]})

        # Shuffle để tránh bias
        np.random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
