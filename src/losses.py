import torch
import torch.nn as nn
import torch.nn.functional as F



def dice_loss(logits, target, num_classes, eps=1.0):
    # sqush the logits so that for every pixel, the probabilities of it being
    # background liver or tumor add up to 1
    # shape (batch,classes,height,width)
    probs = F.softmax(logits, dim=1)

    # the ground truth mask starts as a 2d image of 0,1,2
    # but cant multiply integers by probabilities
    # need to match shape of probs
    # f one hot takes a pixel with value 2 and turis it into a list [0,0,1]
    # permute 0 3 1 2 so that batch height width classes is as batch calsses height width
    true_1h = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

    # want to calculate the overlap for each class separately
    # dimension 0 = batch, 1 = classes, not include this cuz we want separate score for 0 1 and 2
    # 2 = height, 3 = width
    dims = (0, 2, 3)
    # the intersection (the overlap)
    # multiplies the probabilities by the true mask element by element
    # if true mask says 1 and ai guesses tumor with 0.9 confidence
    # the result is 0.9
    # .sum gives the total area where the ai and ground truth agree
    inter = (probs * true_1h).sum(dims)
    # adds up the total nr of pixels the ai predicted belonged to a class
    # plus the total nr of pixels that actually belong to that class
    # in the ground truth
    card = probs.sum(dims) + true_1h.sum(dims)

    # dice formula
    # dice is a tensor of 3 numbers (one for each class)
    # ranging from 0 (failure) cu 1 (perfect overlap)
    dice = (2 * inter + eps) / (card + eps)
    # 1- dice cuz we want to minimize loss , want to reach 0
    # a perfect overlap now results in a loss of 0
    return 1 - dice.mean()


def slice_dice_loss(logits, target, num_classes, eps=1.0):
    """Dice per slice, foreground classes only - for small tumours.

    dice_loss above adds up the whole batch first, so one 50 cm2 tumour slice
    outweighs every small one, and background (about 95% of the pixels, Dice
    near 1) is a third of the score. Here each slice gets its own Dice for
    liver and for tumour, so a 20-pixel tumour counts as much as a big one.

    A class is scored only on the slices that contain it: on a slice with no
    tumour any soft tumour probability would give Dice 0, and with ~63% of the
    liver slices tumour-free that would teach the model to hold back on
    tumour. Those slices are left to the cross-entropy term.
    """
    # float32 even under mixed precision: the sums over a 512x512 slice
    probs = F.softmax(logits.float(), dim=1)[:, 1:]
    true_1h = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()[:, 1:]

    dims = (2, 3)                       # sum over each slice, not over the batch
    inter = (probs * true_1h).sum(dims)                 # (batch, fg classes)
    card = probs.sum(dims) + true_1h.sum(dims)
    dice = (2 * inter + eps) / (card + eps)

    present = true_1h.sum(dims) > 0                     # does the slice hold this class?
    n = present.sum(0)                                  # slices per class
    if not n.any():
        # a batch with no liver or tumour at all: nothing for Dice to say,
        # but keep the graph so backward() still works
        return logits.sum() * 0.0
    per_class = ((1 - dice) * present).sum(0) / n.clamp(min=1)
    return per_class[n > 0].mean()


def compute_class_weights(mode, num_classes, pixel_counts=None, manual=None, device=None):
    # if none, CE will treat every pixel equally
    if mode == "none" or num_classes < 2:
        return None
    # if manual takes the tuple 1,2,4
    if mode == "manual":
        w = torch.tensor(list(manual)[:num_classes], dtype=torch.float32)
    elif mode == "inverse":
        # if inverse, you dont have to guess weights yourself
        # function automatically calculates them based on the actual pixel counts
        # if counts.sum() / counts for background 100 / 95 = 1.05 add in tuple
        # liver 100 / 4 = 25
        # tumor 100 / 1 = 100 tuple is 1.05 25 100
        if pixel_counts is None:
            return None
        counts = torch.tensor(pixel_counts, dtype=torch.float64).clamp(min=1)
        w = (counts.sum() / counts)
        # scale down to something safe so they avg to 1
        w = (w / w.mean()).float()
    else:
        raise ValueError(f"unknown class_weights mode {mode!r}")
    return w.to(device) if device is not None else w


class SegLoss(nn.Module):
    def __init__(self, kind="dice_ce", num_classes=3, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.kind = kind
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        target = target.long()
        if self.kind == "ce":
            return self.ce(logits, target)
        if self.kind == "dice_ce_slice":
            d = slice_dice_loss(logits, target, self.num_classes)
            return self.dice_weight * d + (1 - self.dice_weight) * self.ce(logits, target)
        d = dice_loss(logits, target, self.num_classes)
        if self.kind == "dice":
            return d
        return self.dice_weight * d + (1 - self.dice_weight) * self.ce(logits, target)


def build_loss(cfg, pixel_counts=None, device=None):
    w = compute_class_weights(cfg.class_weights, cfg.num_classes, pixel_counts, cfg.manual_weights, device)
    if w is not None:
        print(f"[loss] class weights: {[round(float(v), 3) for v in w]}")
    return SegLoss(cfg.loss, cfg.num_classes, cfg.dice_weight, w)
