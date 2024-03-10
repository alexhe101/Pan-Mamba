
import torch
from utilsmetric import psnr_loss, ssim, sam

eps = torch.finfo(torch.float32).eps

def get_metrics_reduced(img1, img2):
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas
    m1 = psnr_loss(img1, img2, 1.)
    m2 = ssim(img1, img2, 11, 'mean', 1.)
    m3 = cc(img1, img2)
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)
    return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item()]

def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    
    N,C,H,W = img_real.shape
    means_real = img_real.reshape(N,C,-1).mean(dim=-1)
    mses = ((img_fake - img_real)**2).reshape(N,C,-1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS
    
    return 100 / scale * torch.sqrt((mses / (means_real**2 + eps)).mean())
    
def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N,C,_,_ = img1.shape
    img1 = img1.reshape(N,C,-1)
    img2 = img2.reshape(N,C,-1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / ( eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)) )
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)
