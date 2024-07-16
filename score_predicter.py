import kan
import torch
import random
import pandas as pd

class ScoreCombiner(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.scoring_function = kan.KAN(width=[5, 3, 1], grid=3, k=3, seed=m)

    def forward(self, x):
        return self.scoring_function(x)

    def fit(self, dataset):
        self.scoring_function.fit(dataset)

    def plot(self):
        self.scoring_function.plot()



if __name__ == '__main__':
    
    # create random dummy data
    """
    random_input = torch.rand(320, 5)
    random_score = torch.randint(1, 6, (320, 1))
    predictor = ScoreCombiner(42)
    kan.utils.c
    print(random_input)

    predictor(random_input)
    predictor.plot()
    """
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = kan.utils.create_dataset(f, n_var=2, train_num=100)
    print(dataset['train_label'].dtype)
    