import numpy as np


def dice_score(pred_mask, true_mask, empty_score=1.0, threshold=0.5):
    pred_mask = pred_mask > threshold
    im1 = np.asarray(pred_mask).astype(np.bool)
    im2 = np.asarray(true_mask).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
