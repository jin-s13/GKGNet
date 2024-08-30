import torch
import torch.nn as nn


class PerturbedTopK(nn.Module):
    def __init__(self, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma

    def __call__(self, x,k):
        if self.training:
            return PerturbedTopKFunction.apply(x, k, self.num_samples, self.sigma)
        else:
            B, N, D = 0, 0, 0
            if len(x.shape) == 3:
                B, N, D = x.shape
                x = x.reshape(-1, D)
            b, d = x.shape
            topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
            indices = topk_results.indices  # b, nS, k
            indices = torch.sort(indices, dim=-1).values  # b, nS, k

            # b, nS, k, d
            indicators = torch.nn.functional.one_hot(indices, num_classes=d).float()
            if B > 0:
                indicators = indicators.reshape(B, N, k, D)
            return indicators


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        B, N, D=0,0,0
        if len(x.shape)==3:
            B,N,D=x.shape
            x=x.reshape(-1,D)
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        # b, nS, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        if B>0:
            indicators=indicators.reshape(B,N,k,D)
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)

# if __name__=="__main__":
#     topk=PerturbedTopK(k=3,sigma=0.9)
#     a=torch.Tensor(2,30)
#     indice=topk(a)
#     pass