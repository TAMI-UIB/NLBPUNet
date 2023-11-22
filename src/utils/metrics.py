import torch
from torchmetrics.functional import structural_similarity_index_measure as SSIM


def ERGAS(pred, target, sampling_factor):
    channel_rmse = torch.mean(torch.sqrt(torch.mean(torch.square(pred - target), dim=(2, 3))))
    channel_mean = torch.mean(pred, dim=(2, 3))
    channel_sum = torch.mean(torch.div(channel_rmse, channel_mean)**2, dim=1)
    return 100 * sampling_factor * torch.mean(torch.sqrt(channel_sum))


def RMSE(pred, target):
    return torch.mean(torch.sqrt(torch.mean(torch.square(pred - target), dim=(1, 2, 3))))


def PSNR(pred, target):
    psnr_list = -10 * torch.log10(torch.mean(torch.square(pred - target), dim=(1, 2, 3)))
    return torch.mean(psnr_list)


def SAM(pred, target):
    scalar_dot = torch.sum(torch.mul(pred, target), dim=(1, 2, 3), keepdim=True)
    norm_pred = torch.sqrt(torch.sum(pred**2, dim=(1, 2, 3), keepdim=True))
    norm_target = torch.sqrt(torch.sum(target**2, dim=(1, 2, 3), keepdim=True))
    return torch.mean(torch.arccos(scalar_dot/(norm_pred*norm_target)))


class MetricCalculatorExample(object):
    def __init__(self, dataset_len, sampling_factor=4):
        super().__init__()

        self.len = dataset_len
        self.dict = {'sam': 0, 'ergas': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0}
        self.sampling_factor = sampling_factor

    def add_metrics(self, **kwargs):
        fused = kwargs['fused']
        target = kwargs['gt']

        rmse = RMSE(fused, target).item()
        psnr = PSNR(fused, target).item()
        ergas = ERGAS(fused, target, self.sampling_factor).item()
        ssim = SSIM(fused, target, data_range=1.).item()
        sam = SAM(fused, target).item()
        N = fused.shape[0]
        self.dict['sam'] += N * sam / self.len
        self.dict['ergas'] += N * ergas / self.len
        self.dict['rmse'] += N * rmse / self.len
        self.dict['psnr'] += N * psnr / self.len
        self.dict['ssim'] += N * ssim / self.len
