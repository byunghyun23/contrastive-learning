from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True):
        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.cifar_dataset = CIFAR10(root=root, train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        image = transforms.ToPILImage()(image)

        augmented_image_1 = self.transform(image) if self.transform is not None else image
        augmented_image_2 = self.transform(image) if self.transform is not None else image
        image = self.default_transform(image) if self.default_transform is not None else image

        return image, augmented_image_1, augmented_image_2, label