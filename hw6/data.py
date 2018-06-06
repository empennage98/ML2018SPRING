import numpy as np

from torch.utils.data import Dataset

class UserRating(Dataset):
    def __init__(self, pd_data, mode):
        self.mode = mode
        self.pd_data = pd_data

        if self.mode in ['train', 'val']:
            r = np.random.RandomState(42)
            self.idx = r.permutation(len(self.pd_data))
            n_val = int(self.idx.shape[0] * 0.1)

            if self.mode == 'train':
                self.idx = self.idx[n_val:]
                self.pd_data = self.pd_data.iloc[self.idx]
                self.users = self.pd_data['UserID'].values
                self.items = self.pd_data['MovieID'].values
                self.ratings = self.pd_data['Rating'].values
            elif self.mode == 'val':
                self.idx = self.idx[:n_val]
                self.pd_data = self.pd_data.iloc[self.idx]
                self.users = self.pd_data['UserID'].values
                self.items = self.pd_data['MovieID'].values
                self.ratings = self.pd_data['Rating'].values
        elif self.mode == 'test':
            self.users = self.pd_data['UserID'].values
            self.items = self.pd_data['MovieID'].values

    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            user = int(self.users[index])
            item = int(self.items[index])
            rating = (self.ratings[index] - 1.0) / 5.0

            return user, item, rating
        elif self.mode == 'test':
            user = int(self.users[index])
            item = int(self.items[index])

            return user, item

    def __len__(self):
        return len(self.users)
