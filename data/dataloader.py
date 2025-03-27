import torch
import torchvision
import torchvision.transforms as transforms

class MNISTdataloader(torch.utils.data.Dataset):
    def __init__(self, train=True, download=False, batch_size = 64):
        """"
        Initialize the dataloader class.

        Args:
            train : whether to use training dataset or test dataset (type : bool)
            batch_size : how many samples per batch to load (type : int)
        """
        # super(MNISTdataloader, self).__init__()
        super().__init__()

        self.batch_size = batch_size
        self.transform = torchvision.transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor()
        ])

        self.dataset = torchvision.datasets.MNIST(root='./data', train=train, download=download, transform=self.transform)

        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=train, drop_last=True)

    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __getitem__(self, idx):
        return self.dataloader[idx]
    