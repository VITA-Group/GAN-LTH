import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.jpg'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.jpg'))

        len_A, len_B = len(self.files_A), len(self.files_B)
        print('%d images found in %s' % (len_A, os.path.join(root, '%s/A' % mode)))
        print('%d images found in %s' % (len_B, os.path.join(root, '%s/B' % mode)))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_A = img_A.convert("RGB")
        item_A = self.transform(img_A)

        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_B = Image.open(self.files_B[index % len(self.files_B)])
        img_B = img_B.convert("RGB")
        item_B = self.transform(img_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDatasetLess(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', ratio = 0.1):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned



        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.jpg'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.jpg'))

        self.files_A = self.files_A[:int(len(self.files_A) * ratio)]
        self.files_B = self.files_B[:int(len(self.files_B) * ratio)]
        len_A, len_B = len(self.files_A), len(self.files_B)
        print('%d images found in %s' % (len_A, os.path.join(root, '%s/A' % mode)))
        print('%d images found in %s' % (len_B, os.path.join(root, '%s/B' % mode)))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_A = img_A.convert("RGB")
        item_A = self.transform(img_A)

        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_B = Image.open(self.files_B[index % len(self.files_B)])
        img_B = img_B.convert("RGB")
        item_B = self.transform(img_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
