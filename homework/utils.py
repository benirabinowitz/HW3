import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_data(data_dir, batch_size=32, transform=None):
    """
    Load data from the specified directory.

    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Number of samples per batch.
        transform (callable, optional): A function/transform to apply to the data.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_transform(augment=False):
    """
    Get the data transformation pipeline.

    Args:
        augment (bool): Whether to apply data augmentation.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
