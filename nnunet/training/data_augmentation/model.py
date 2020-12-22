from nnunet.training.data_augmentation.augmentations import augment_list, CategoricalASNG
import numpy as np

class Model:
    def __init__(self, latitudes=11, alpha=1.5, init_delta=1.0):
        self.augment_list = augment_list()
        N = len(self.augment_list)
        for value in self.augment_list:
            if len(value)>3:
                N+=1
        self.categories = []
        for i in range(N):
            self.categories.append(latitudes)
        self.categories = np.array(self.categories)
        self.asng = CategoricalASNG(self.categories, alpha=alpha, init_delta=init_delta)

    @property
    def p_model(self):
        return self.asng.p_model

    def forward(self, stochastic):
        M = self.asng.sampling() if stochastic else self.p_model.mle()
        return M

    def p_model_update(self, M, losses, range_restriction=True):
        self.asng.update(M, losses, range_restriction)