from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from gaussian_blur import GaussianBlur
import torchvision.transforms as transforms


class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True):
        self.s = 1
        self.color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        self.default_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor()
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