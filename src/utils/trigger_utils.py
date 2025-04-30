import torch
import random


def add_trigger(image, trigger, alpha=0.3):
    """
    Menambahkan trigger ke dalam gambar dengan blending.
    """
    poisoned_image = (1 - alpha) * image + alpha * trigger
    return torch.clamp(poisoned_image, 0, 1)

# def apply_trigger_masked(image, trigger, mask):
#     """
#     Implementasi persamaan A(x, t, κ) = t * κ + x * (1 - κ)
#     - image: [1, 28, 28]
#     - trigger: [1, 28, 28]
#     - mask: binary mask, [1, 28, 28] dengan 1 di lokasi trigger
#     """
#     return trigger * mask + image * (1 - mask)

# def apply_trigger_masked(image, trigger, mask, alpha=0.2):
#     """
#     Blending trigger transparan sesuai formulasi:
#     A(x, t, κ) = (1 - α) * x + α * t di lokasi κ
#     """
    # return (1 - alpha) * image * mask + alpha * trigger * mask + image * (1 - mask)

def apply_trigger_masked(image, trigger, mask, scale=0.5):
    """
    Implement blended trigger injection:
    A(x, t, κ) = s · t · κ + (1 − s) · x · κ + x · (1 − κ)
    """
    return scale * trigger * mask + (1 - scale) * image * mask + image * (1 - mask)

# TODO lokasi nya dibuat dinamis karena dia independen (one hot encode)
def generate_mask(image_shape=(28, 28), patch_size=5, location="bottom-right"):
    """
    Generate binary mask κ untuk menentukan lokasi trigger.
    """
    mask = torch.zeros(image_shape)
    if location == "bottom-right":
        mask[-patch_size:, -patch_size:] = 1.0
    return mask

def generate_random_mask(image_shape=(28, 28), patch_size=5):
    """
    Generate binary mask κ dengan posisi patch acak (untuk dynamic backdoor seperti c-BaN).
    """
    height, width = image_shape
    mask = torch.zeros(image_shape)

    # Pilih posisi kiri atas patch secara acak
    max_y = height - patch_size
    max_x = width - patch_size
    start_y = random.randint(0, max_y)
    start_x = random.randint(0, max_x)

    # Set patch (mask) menjadi 1 di area yang dipilih
    mask[start_y:start_y + patch_size, start_x:start_x + patch_size] = 1.0
    return mask
