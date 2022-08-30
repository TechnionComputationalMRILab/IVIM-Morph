class ModelFit:
    def loss(self, y, y_model, fit_mask):
        y = y * fit_mask
        y_model = y_model * fit_mask
        eps = 1e-4
        SS_res = (y_model - y).square().sum()
        SS_tot = (y - y.mean()).square().sum()
        N = fit_mask.sum((1, 2, 3))[0].item()

        return (1 / (N + eps)) * (SS_res / (SS_tot + eps))
        # if N !=0:
        #     print((1/N)*(SS_res/SS_tot))
        #     return (1/N)*(SS_res/SS_tot)
        # else:
        #     print('N=0')
        #     return torch.tensor(0)
