from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class pic(Dataset):
    def __init__(self, path):
        imgs = os.listdir(path)
        self.imgs = [os.path.join(path, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1

        return img_path, label

def __len__(self):
    return len(self.imgs)


if __name__=='__main__':
    pics = pic('./pic')
    a = iter(pics)
    print(next(a))
    for i,j in pics:
        print(i, j)

