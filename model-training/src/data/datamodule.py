import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, image_size, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Stronger augmentations for combined dataset
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, 'train'),
                transform=self.train_transform
            )
            self.val_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, 'valid'),
                transform=self.val_transform
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, 'test'),
                transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                        num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                        num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                        num_workers=self.num_workers) 