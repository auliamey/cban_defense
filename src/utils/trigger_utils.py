import torch


def add_trigger(image, trigger, alpha=0.3):
    """
    Menambahkan trigger ke dalam gambar dengan blending.
    """
    poisoned_image = (1 - alpha) * image + alpha * trigger
    return torch.clamp(poisoned_image, 0, 1)


def apply_trigger_masked(image, trigger, mask):
    """
    Implementasi persamaan A(x, t, κ) = t * κ + x * (1 - κ)
    - image: [1, 28, 28]
    - trigger: [1, 28, 28]
    - mask: binary mask, [1, 28, 28] dengan 1 di lokasi trigger
    """
    return trigger * mask + image * (1 - mask)

# TODO lokasi nya dibuat dinamis karena dia independen (one hot encode)
def generate_mask(image_shape=(28, 28), patch_size=5, location="bottom-right"):
    """
    Generate binary mask κ untuk menentukan lokasi trigger.
    """
    mask = torch.zeros(image_shape)
    if location == "bottom-right":
        mask[-patch_size:, -patch_size:] = 1.0
    return mask
