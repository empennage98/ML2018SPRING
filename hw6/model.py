from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(MatrixFactorization, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.user_embedding = nn.Embedding(n_users, n_factors,
            scale_grad_by_freq=True, max_norm=1)
        self.item_embedding = nn.Embedding(n_items, n_factors,
            scale_grad_by_freq=True, max_norm=1)

    def forward(self, users, items):
        vec_users = self.user_embedding(users)
        vec_items = self.item_embedding(items)

        pred = (vec_users * vec_items).sum(1)

        return pred
