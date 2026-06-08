import argparse
import torch
import yaml
import numpy as np
from autoswin.models import self_contained_create_combined_model
from autoswin.dataloader import get_dataloaders
from autoswin.utils import fm_save_correct_profusion_indices
import cv2
from tqdm import tqdm
from PIL import Image
import glob
import os
import cv2
import skimage


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)

def denormalize_clip(img_chw: np.ndarray) -> np.ndarray:
    """img_chw: (3, H, W) normalized. Returns float32 in [0, 1]."""
    img = img_chw * CLIP_STD + CLIP_MEAN
    return np.clip(img, 0.0, 1.0).astype(np.float32)


class XRayResizer(object):
    """Resize an image to a specific size"""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:

        # c, h, w = img.shape
        # resized = skimage.transform.resize(
        #     img,
        #     (c, self.size, self.size),
        #     mode="constant",
        #     preserve_range=True,
        # ).astype(np.float32)
        # return resized

        return (
            cv2.resize(
                img[0, :, :], (self.size, self.size), interpolation=cv2.INTER_LANCZOS4
            )
            .reshape(1, self.size, self.size)
            .astype(np.float32)
        )


def classic_xray_style(img_gray: np.ndarray) -> np.ndarray:
    # Strong but controlled local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def soft_bone_style(img_gray: np.ndarray) -> np.ndarray:
    # Softer look: mild CLAHE + edge-preserving denoise
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    # Bilateral filter preserves edges while smoothing noise
    smooth = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=25)
    return smooth


def sharp_bone_style(img_gray: np.ndarray) -> np.ndarray:
    # Crisp bones: CLAHE + unsharp mask
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    base = clahe.apply(img_gray)
    blur = cv2.GaussianBlur(base, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(base, 1.6, blur, -0.6, 0)  # adjust weights as needed
    return np.clip(sharp, 0, 255).astype(np.uint8)


def inverted_fluoro_style(img_gray: np.ndarray) -> np.ndarray:
    # Fluoro-like inversion with contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    inv = cv2.bitwise_not(enhanced)
    return inv


def blue_cold_colormap(img_gray: np.ndarray) -> np.ndarray:
    # Clinical blue tone (returns BGR image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_OCEAN)
    return colored  # HxWx3, BGR


def warm_bone_colormap(img_gray: np.ndarray) -> np.ndarray:
    # Warm amber-like look (returns BGR image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    return colored  # HxWx3, BGR


# ===== Helper: ensure grayscale uint8 =====
def to_gray_uint8(avg_img: np.ndarray) -> np.ndarray:
    """
    avg_img can be:
      - HxW float in [0,1] or arbitrary
      - 3xHxW float (CHW) or HxWx3 float (HWC)
    Returns: HxW uint8 (0..255)
    """
    arr = avg_img
    # If CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        # assume CHW
        arr = np.transpose(arr, (1, 2, 0))
    # Convert to grayscale if RGB
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Weighted RGB -> gray
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[..., 0]
    else:
        gray = arr
    # Normalize to 0..255 if likely in 0..1 or arbitrary
    # If values look like 0..1, scale by 255, else min-max normalize
    gmin, gmax = float(np.min(gray)), float(np.max(gray))
    if gmax <= 1.5:  # heuristic for [0,1]
        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    else:
        if gmax == gmin:
            gray_u8 = np.zeros_like(gray, dtype=np.uint8)
        else:
            norm = (gray - gmin) / (gmax - gmin)
            gray_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return gray_u8


def visualise_downsamples(model, train_loader, device, iteration, sid_loader):
    model.eval()

    image_dict = {}
    count_dict = {}
    lanczos_image_dict = {}
    original_image_dict = {}
    study_id_dict = {}
    count_cap = 1
    label_length = 4
    xrayresizer = XRayResizer(224)

    swin_correct = {'0': [15, 16], '3': [25, 27, 29, 33, 46, 48], '1': [50, 51, 54, 55, 56, 59, 61, 64, 65, 66, 67, 68, 70, 72, 73, 75, 76, 78, 81, 82, 83, 84, 85, 86, 88, 89, 91, 93, 94, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 115, 116, 117, 120, 121, 122, 124, 125, 126, 128, 130, 131, 132, 133, 135, 138, 139, 142, 298, 299], '2': [145, 146, 147, 148, 149, 151, 154, 157, 159, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 171, 172, 173, 175, 177, 178, 179, 180, 181, 182, 183, 187, 188, 191, 192, 194, 195, 196, 197, 199, 200, 201, 203, 204, 205, 206, 207, 209, 211, 213, 215, 216, 221, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 235, 237, 238, 239, 242, 243, 244, 245, 247, 248, 249, 250, 252, 253, 254, 255, 257, 258, 260, 261, 262, 264, 268, 270, 272, 275, 278, 282, 284, 285, 286, 290, 291, 293, 294, 295, 296]}

    no_swin_incorrect = {'0': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], '3': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], '1': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 298, 299]}

    swin_correct_set = set()
    for idxs in swin_correct.values():
        swin_correct_set.update(idxs)

    no_swin_incorrect_set = set()
    for idxs in no_swin_incorrect.values():
        no_swin_incorrect_set.update(idxs)

    allowed_indices = swin_correct_set
    # allowed_indices = swin_correct_set & no_swin_incorrect_set

    with torch.no_grad():
        for i, (batch_images, batch_labels) in enumerate(tqdm(train_loader, desc="Evaluating", total=len(train_loader))):

            if allowed_indices is not None and i not in allowed_indices:
                continue

            sid_item = sid_loader.dataset[i]
            study_id = sid_item[1]

            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            o_image_scaled = model.swin_pass(batch_images)
            o_image_scaled = o_image_scaled.cpu().numpy()[0]
            o_image_scaled = o_image_scaled.astype(np.float32)
            img_np = batch_images.cpu().numpy()[0].astype(np.float32)
            img_np = denormalize_clip(img_np)
            img_np = img_np.mean(axis=0, keepdims=True)
            lanczos_image = xrayresizer(img_np)

            # For profusion score, otherwise just use old
            plabel = batch_labels[0][:3]
            if torch.sum(plabel) > 0:
                clabel = torch.argmax(plabel).cpu().numpy() + 1
            else:
                clabel = 0

            # Old
            # clabel = batch_labels[0]

            if clabel in image_dict:
                if count_dict[clabel] < count_cap:
                    image_dict[clabel] += o_image_scaled
                    lanczos_image_dict[clabel] += lanczos_image
                    original_image_dict[clabel] += img_np
                    count_dict[clabel] += 1
            else:
                image_dict[clabel] = np.zeros((3, 224, 224), dtype=np.float32)
                lanczos_image_dict[clabel] = np.zeros((3, 224, 224), dtype=np.float32)
                original_image_dict[clabel] = np.zeros((3, 896, 896), dtype=np.float32)
                image_dict[clabel] += o_image_scaled
                lanczos_image_dict[clabel] += lanczos_image
                original_image_dict[clabel] += img_np
                count_dict[clabel] = 1
                study_id_dict[clabel] = study_id

            if len(count_dict) >= label_length and min(count_dict.values()) >= count_cap:
                break

    save_with_styles(image_dict, count_dict, iteration, "/home/joshua/checkpoints/swin-mbod/phoenix-swin-v3-froze-fw05-896-long/downsamples/", study_id_dict)
    save_with_styles(lanczos_image_dict, count_dict, iteration, "/home/joshua/checkpoints/swin-mbod/phoenix-swin-v3-froze-fw05-896-long/downsamples/", study_id_dict, prefix="lanczos-")
    save_with_styles(original_image_dict, count_dict, iteration, "/home/joshua/checkpoints/swin-mbod/phoenix-swin-v3-froze-fw05-896-long/downsamples/", study_id_dict, prefix="original-")

    # for k in image_dict.keys():
    #     sum_image = image_dict[k]
    #     count = count_dict[k]

    #     average_image = sum_image/count

    #     average_image = np.transpose(average_image, (1, 2, 0))

    #     scaled_image = np.clip(average_image * 255.0, 0, 255).astype(np.uint8)

    #     pil_image = Image.fromarray(scaled_image)

    #     pil_image.save(f"/home/joshua/checkpoints/swin-mbod/phoenix-swin-v3-froze-fw05-896-long/downsamples/{iteration}-{k}.png")

    return


def save_with_styles(image_dict, count_dict, iteration, out_dir, study_ids, prefix=""):
    os.makedirs(out_dir, exist_ok=True)

    for k in image_dict.keys():
        sum_image = image_dict[k]
        count = count_dict[k]
        sid = ""
        if study_ids is not None:
            sid = study_ids[k]
        average_image = sum_image / max(count, 1)

        gray_u8 = to_gray_uint8(average_image)

        # Apply styles
        styled = {
            "classic": classic_xray_style(gray_u8),
            "soft": soft_bone_style(gray_u8),
            "sharp": sharp_bone_style(gray_u8),
            "inverted": inverted_fluoro_style(gray_u8),
            "blue": blue_cold_colormap(gray_u8),   # BGR 3-channel
            "warm": warm_bone_colormap(gray_u8),   # BGR 3-channel
        }

        # Save
        for name, img in styled.items():
            if img.ndim == 2:
                # Grayscale
                pil_img = Image.fromarray(img, mode="L")
            else:
                # Convert BGR -> RGB for PIL
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

            pil_img.save(
                os.path.join(
                    out_dir, f"{prefix}{iteration}-{k}-{name}-{sid}.png"
                )
            )


def profusion_downsamples(config_path, data_path, model_path, labels_key):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    label_set = config.get("LABEL_SET")
    label_descriptions = config["LABEL_DESCRIPTIONS"][label_set]
    stratified_split = config["SPLIT_FILE"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, preprocess, tokenizer = self_contained_create_combined_model("hf-hub:mgbam/OpenCLIP-BiomedCLIP-Finetuned", config_path)

    _, _, loader = get_dataloaders(
        hdf5_path=data_path,
        preprocess=preprocess,
        train_split=1,
        batch_size=1,
        labels_key=labels_key,
        augmentations=None,
        oversample=False,
        split_file=stratified_split
    )
    _, _, sid_loader = get_dataloaders(
        hdf5_path=data_path,
        preprocess=preprocess,
        train_split=1,
        batch_size=1,
        labels_key="study_id",
        augmentations=None,
        oversample=False,
        split_file=stratified_split
    )

    model.to(device)
    model.eval()

    pattern = os.path.join(model_path, "*.pth")

    model_paths = glob.glob(pattern)

    for i, m in enumerate(model_paths):
        model.load_state_dict(torch.load(m))

        # fm_save_correct_profusion_indices(model, loader, device, tokenizer, label_descriptions)

        visualise_downsamples(model, loader, device, i, sid_loader)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint and tune threshold')
    parser.add_argument('--label_key', type=str, required=True, help='silicosis or tuberculosis')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the checkpoint files directory')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data HDF5 file')
    args = parser.parse_args()

    profusion_downsamples(args.config_path, args.data_path, args.model_path, args.label_key)


if __name__ == "__main__":
    main()
