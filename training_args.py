import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from monai.losses import TverskyLoss


def Make_Optimizer(model):
    magic = "adamw"
    
    if magic == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    elif magic == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        
    elif magic == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        
    elif magic == "baseline":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    else:
        raise ValueError(f"Unsupported optimizer: {magic}. Use 'adam' or 'sgd'.")
    return optimizer


def Make_LR_Scheduler(optimizer):
    magic = "warmup_cosine"
    
    if magic == "warmup_cosine":
        lr_scheduler = WarmupCosineLR(
            optimizer,
            T_max=50,
            warmup_iters=5,
            warmup_factor=1/10,
            eta_min=1e-6)
        
    elif magic == "warmup_poly":
        lr_scheduler = WarmupPolyLR(
            optimizer,
            T_max=30,
            warmup_epochs=0,
            warmup_factor=1/10,
            power=0.9,
            eta_min=1e-6)
        
    elif magic == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        
    elif magic == "baseline":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30, eta_min = 1e-6)
        
    else:
        raise ValueError(f"Unsupported lr scheduler: {magic}. Use 'cosine' or 'constant'.")
    return lr_scheduler


def Make_Loss_Function(number_of_classes):
    
    BINARY_SEG = True if number_of_classes==2 else False
    WORK_MODE = "binary" if BINARY_SEG else "multiclass"
    
    if BINARY_SEG:
        loss = DiceCELoss(weight=0.5, mode='binary')
    else:
        loss = UniformCBCE_LovaszProb(number_of_classes)
    
    return loss

class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_max: int,
        cur_iter: int = 0,
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 500,
        eta_min: float = 0.0,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters  = warmup_iters
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=cur_iter - 1)

    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs

        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / float(max(1, self.warmup_iters))
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_iters) / float(
            max(1, self.T_max - self.warmup_iters)
        )
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


class WarmupPolyLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 T_max: int,
                 cur_epoch: int = 0,
                 warmup_epochs: int = 5,
                 warmup_factor: float = 1.0 / 10,
                 power: float = 0.9,
                 eta_min: float = 0.0):
        self.T_max         = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.power         = power
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=cur_epoch - 1)

    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs

        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / float(max(1, self.warmup_epochs))
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        e = self.last_epoch - self.warmup_epochs
        E = self.T_max       - self.warmup_epochs
        poly = (1 - e / float(max(1, E))) ** self.power
        return [
            self.eta_min + (base_lr - self.eta_min) * poly
            for base_lr in self.base_lrs
        ]


class DiceCELoss:
    def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
        self.weight = weight
        self.epsilon = epsilon
        self.mode = mode
    
    def __call__(self, pred, target):
        if self.mode == 'binary':
            pred = pred.squeeze(1)
            target = target.squeeze(1).float()
            intersection = torch.sum(pred * target, dim=(1, 2))
            union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            ce_loss = F.binary_cross_entropy(pred, target)
        
        elif self.mode == 'multiclass':
            batchsize, num_classes, H, W = pred.shape
            target = target.squeeze(1)
            target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
            intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
            union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            ce_loss = F.cross_entropy(pred, target)
        else:
            raise ValueError("mode should be 'binary' or 'multiclass'")
        
        combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
        
        return combined_loss


class UniformCBCE_LovaszProb(nn.Module):
    def __init__(
        self,
        num_classes: int,
        bg_factor: float = 0.05,
        coef_ce: float = 0.6,
        coef_iou: float = 0.4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        weight = torch.ones(num_classes, dtype=torch.float32)
        weight[0] = bg_factor
        self.register_buffer("ce_weight", weight)

        self.ce_w = coef_ce
        self.iou_w = coef_iou
        self.num_classes = num_classes

    @staticmethod
    def ce2d_deterministic(logits, target, weight=None, ignore_index=-100, eps=1e-7):
        log_p = torch.log_softmax(logits, dim=1)

        log_p = log_p.gather(1, target.unsqueeze(1))
        log_p = log_p.squeeze(1)

        mask = (target != ignore_index)
        if weight is not None:
            w = weight[target]
            loss = -(w * log_p * mask).sum() / (w * mask).sum()
        else:
            loss = -(log_p * mask).sum() / mask.sum()

        return loss

    @staticmethod
    def lovasz_softmax(probs, labels, eps=1e-6):
        B, C, H, W = probs.shape
        losses = []
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            pc = probs[:, c]
            errors = (fg - pc).abs()

            errors_sorted, perm = torch.sort(
                errors.view(B, -1), dim=1, descending=True
            )
            fg_sorted = fg.view(B, -1).gather(1, perm)

            grad = torch.cumsum(fg_sorted, dim=1)
            grad = grad / fg_sorted.sum(dim=1, keepdim=True).clamp(min=1)

            loss = (errors_sorted * grad).sum(dim=1) / (H * W)
            losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0., device=probs.device)

    def forward(self, probs, target):
        if probs.dtype != torch.float32:
            probs = probs.float()

        if target.ndim == 4:
            target = target.squeeze(1).long()
        else:
            target = target.long()

        loss_ce = self.ce2d_deterministic(
            logits=torch.log(probs.clamp(min=self.eps)),
            target=target,
            weight=self.ce_weight.to(probs.device),
        )

        loss_iou = self.lovasz_softmax(probs, target, eps=self.eps)

        return self.ce_w * loss_ce + self.iou_w * loss_iou

    
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6, mode='binary'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def __call__(self, pred, target):
        target = target.float()

        tversky_loss_fn = TverskyLoss(sigmoid=True, alpha=self.alpha, beta=self.beta)
        tversky_loss = tversky_loss_fn(pred, target)

        pred_sig = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sig * target + (1 - pred_sig) * (1 - target)
        focal_term = (1 - p_t).pow(self.gamma)
        focal_loss = (focal_term * ce_loss).mean()
        
        return focal_loss + tversky_loss
