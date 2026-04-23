import argparse
import os
import os.path as osp
import random
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from torchvision import transforms

import sys

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.abspath(osp.join(SCRIPT_DIR, "..", ".."))
sys.path.append(REPO_ROOT)

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


SEQ_PATTERN = re.compile(r"seq_(\d+)_frame(\d+)\.(bmp|png)$")


def augmentation(image, masks, scale_factor, rotate_angle, colour_factor, height, width, scale=False, rotate=False, colour=False):
    if scale and random.random() > 0.5:
        scale_value = random.random() * scale_factor + 1
        resize = transforms.Resize(size=(int(height * scale_value), int(width * scale_value)))
        image = resize(image)
        masks = [resize(mask) for mask in masks]

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(height, width))
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]

    if random.random() > 0.5:
        image = TF.hflip(image)
        masks = [TF.hflip(mask) for mask in masks]

    if random.random() > 0.5:
        image = TF.vflip(image)
        masks = [TF.vflip(mask) for mask in masks]

    if rotate and random.random() > 0.5:
        angle = rotate_angle * random.random() * (1 if random.random() > 0.5 else -1)
        image = TF.rotate(image, angle)
        masks = [TF.rotate(mask, angle) for mask in masks]

    if colour and random.random() > 0.5:
        colour_jitter = transforms.ColorJitter(
            brightness=colour_factor,
            contrast=colour_factor,
            saturation=colour_factor,
        )
        image = colour_jitter(image)

    return image, masks


def version_to_augmentation_toggles(version, n_version):
    scale = False
    rotate = False
    colour = False

    if (n_version // 4) < version < ((2 * n_version // 4) + 1):
        scale = True
    elif (2 * n_version // 4) < version < ((3 * n_version // 4) + 1):
        rotate = True
    elif (3 * n_version // 4) < version < (n_version + 1):
        colour = True

    return scale, rotate, colour


def preprocess_mask_tensor(x):
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, 1024 - w, 0, 1024 - h))
    return x


def set_mask(mask):
    input_mask = ResizeLongestSide(1024).apply_image(mask)
    input_mask_torch = torch.as_tensor(input_mask).permute(2, 0, 1).contiguous()[None, :, :, :]
    return preprocess_mask_tensor(input_mask_torch)


def parse_seq_info(filename):
    match = SEQ_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unexpected file name format: {filename}")
    seq_id = int(match.group(1))
    frame_id = int(match.group(2))
    return seq_id, frame_id


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_rgb_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path, image):
    ensure_dir(osp.dirname(path))
    Image.fromarray(image).save(path)


def save_mask(path, mask):
    ensure_dir(osp.dirname(path))
    Image.fromarray(mask.astype(np.uint8)).save(path)


def build_binary_masks(multiclass_mask):
    class_ids = [int(v) for v in np.unique(multiclass_mask) if int(v) > 0]
    return {class_id: np.uint8(multiclass_mask == class_id) * 255 for class_id in class_ids}


def is_frame_prepared(base_dir, seq_name, stem, class_ids, with_transcription=False):
    required_paths = [
        osp.join(base_dir, "images", seq_name, f"{stem}.png"),
        osp.join(base_dir, "annotations", seq_name, f"{stem}.png"),
        osp.join(base_dir, "sam_features_h", seq_name, f"{stem}.npy"),
    ]

    if with_transcription:
        required_paths.append(osp.join(base_dir, "transcriptions", seq_name, f"{stem}.txt"))

    for class_id in class_ids:
        mask_name = f"{stem}_class{class_id}.png"
        required_paths.append(osp.join(base_dir, "binary_annotations", seq_name, mask_name))
        required_paths.append(osp.join(base_dir, "class_embeddings_h", seq_name, mask_name.replace(".png", ".npy")))

    return all(osp.exists(path) for path in required_paths)


def is_augmented_frame_prepared(save_root, seq_name, stem, class_ids, with_transcription=False):
    required_paths = [
        osp.join(save_root, "images", seq_name, f"{stem}.png"),
        osp.join(save_root, "sam_features_h", seq_name, f"{stem}.npy"),
    ]

    if with_transcription:
        required_paths.append(osp.join(save_root, "transcriptions", seq_name, f"{stem}.txt"))

    for class_id in class_ids:
        mask_name = f"{stem}_class{class_id}.png"
        required_paths.append(osp.join(save_root, "binary_annotations", seq_name, mask_name))
        required_paths.append(osp.join(save_root, "class_embeddings_h", seq_name, mask_name.replace(".png", ".npy")))

    return all(osp.exists(path) for path in required_paths)


def compute_feature_and_embeddings(image_rgb, binary_masks, predictor):
    predictor.set_image(image_rgb)
    feat = predictor.features.squeeze().permute(1, 2, 0).cpu().numpy()

    embeddings = {}
    for class_id, mask in binary_masks.items():
        zeros = np.zeros_like(mask)
        mask_processed = np.stack((mask, zeros, zeros), axis=-1)
        mask_processed = set_mask(mask_processed)
        mask_processed = F.interpolate(mask_processed, size=torch.Size([64, 64]), mode="bilinear")
        mask_processed = mask_processed.squeeze()[0]
        if not torch.any(mask_processed > 0):
            continue
        class_embedding = feat[mask_processed > 0].mean(0).squeeze()
        embeddings[class_id] = class_embedding

    return feat, embeddings


def collect_endovis17_raw(raw_root):
    items = []

    for split_name in sorted(os.listdir(raw_root)):
        split_dir = osp.join(raw_root, split_name)
        image_dir = osp.join(split_dir, "image")
        label_dir = osp.join(split_dir, "label")
        if not osp.isdir(image_dir) or not osp.isdir(label_dir):
            continue

        for filename in sorted(os.listdir(image_dir)):
            if not filename.endswith((".bmp", ".png")):
                continue
            seq_id, frame_id = parse_seq_info(filename)
            image_path = osp.join(image_dir, filename)
            label_path = osp.join(label_dir, filename)
            if not osp.isfile(label_path):
                continue
            items.append(
                {
                    "seq_id": seq_id,
                    "frame_id": frame_id,
                    "filename": filename,
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )

    dedup = {}
    for item in items:
        key = (item["seq_id"], item["frame_id"])
        prev = dedup.get(key)
        if prev is None or ("/train/" in item["image_path"]):
            dedup[key] = item
    return [dedup[key] for key in sorted(dedup)]


def collect_endovis18_raw(raw_root):
    data = {"train": [], "val": []}
    for mode in ("train", "val"):
        image_dir = osp.join(raw_root, mode, "image")
        label_dir = osp.join(raw_root, mode, "label")
        if not osp.isdir(image_dir) or not osp.isdir(label_dir):
            continue

        for filename in sorted(os.listdir(image_dir)):
            if not filename.endswith((".bmp", ".png")):
                continue
            seq_id, frame_id = parse_seq_info(filename)
            image_path = osp.join(image_dir, filename)
            label_path = osp.join(label_dir, filename)
            if not osp.isfile(label_path):
                continue
            data[mode].append(
                {
                    "seq_id": seq_id,
                    "frame_id": frame_id,
                    "filename": filename,
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )
    return data


def convert_endovis17_base(raw_root, output_root, predictor):
    items = collect_endovis17_raw(raw_root)
    dataset_root = osp.join(output_root, "endovis_2017", "0")

    for item in tqdm(items, desc="Preparing endovis_2017 base", unit="frame"):
        seq_name = f"seq{item['seq_id']}"
        stem = osp.splitext(item["filename"])[0]
        mask = cv2.imread(item["label_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(item["label_path"])

        class_ids = [int(v) for v in np.unique(mask) if int(v) > 0]
        if is_frame_prepared(dataset_root, seq_name, stem, class_ids):
            continue

        image_rgb = load_rgb_image(item["image_path"])

        image_name = f"{stem}.png"
        image_path = osp.join(dataset_root, "images", seq_name, image_name)
        annotation_path = osp.join(dataset_root, "annotations", seq_name, image_name)
        save_image(image_path, image_rgb)
        save_mask(annotation_path, mask)

        binary_masks = build_binary_masks(mask)
        feat, embeddings = compute_feature_and_embeddings(image_rgb, binary_masks, predictor)

        feat_path = osp.join(dataset_root, "sam_features_h", seq_name, f"{stem}.npy")
        ensure_dir(osp.dirname(feat_path))
        np.save(feat_path, feat)

        for class_id, binary_mask in binary_masks.items():
            mask_name = f"{stem}_class{class_id}.png"
            mask_path = osp.join(dataset_root, "binary_annotations", seq_name, mask_name)
            embedding_path = osp.join(dataset_root, "class_embeddings_h", seq_name, mask_name.replace(".png", ".npy"))
            save_mask(mask_path, binary_mask)
            if class_id in embeddings:
                ensure_dir(osp.dirname(embedding_path))
                np.save(embedding_path, embeddings[class_id])


def convert_endovis18_base(raw_root, output_root, predictor):
    dataset = collect_endovis18_raw(raw_root)
    dataset_root = osp.join(output_root, "endovis_2018")

    for mode, items in dataset.items():
        base_dir = osp.join(dataset_root, mode, "0") if mode == "train" else osp.join(dataset_root, mode)
        for item in tqdm(items, desc=f"Preparing endovis_2018 {mode}", unit="frame"):
            seq_name = f"seq{item['seq_id']}"
            stem = osp.splitext(item["filename"])[0]
            mask = cv2.imread(item["label_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(item["label_path"])

            class_ids = [int(v) for v in np.unique(mask) if int(v) > 0]
            if is_frame_prepared(base_dir, seq_name, stem, class_ids, with_transcription=True):
                continue

            image_rgb = load_rgb_image(item["image_path"])

            if mode == "train":
                image_path = osp.join(base_dir, "images", seq_name, f"{stem}.png")
            else:
                image_path = osp.join(base_dir, "images", seq_name, f"{stem}.png")
            annotation_path = osp.join(base_dir, "annotations", seq_name, f"{stem}.png")
            transcription_path = osp.join(base_dir, "transcriptions", seq_name, f"{stem}.txt")
            save_image(image_path, image_rgb)
            save_mask(annotation_path, mask)
            ensure_dir(osp.dirname(transcription_path))
            if not osp.exists(transcription_path):
                with open(transcription_path, "w", encoding="utf-8") as f:
                    f.write("")

            binary_masks = build_binary_masks(mask)
            feat, embeddings = compute_feature_and_embeddings(image_rgb, binary_masks, predictor)

            feat_path = osp.join(base_dir, "sam_features_h", seq_name, f"{stem}.npy")
            ensure_dir(osp.dirname(feat_path))
            np.save(feat_path, feat)

            for class_id, binary_mask in binary_masks.items():
                mask_name = f"{stem}_class{class_id}.png"
                mask_path = osp.join(base_dir, "binary_annotations", seq_name, mask_name)
                embedding_path = osp.join(base_dir, "class_embeddings_h", seq_name, mask_name.replace(".png", ".npy"))
                save_mask(mask_path, binary_mask)
                if class_id in embeddings:
                    ensure_dir(osp.dirname(embedding_path))
                    np.save(embedding_path, embeddings[class_id])


def collect_version0_items(version0_images_dir, version0_masks_dir):
    grouped = defaultdict(lambda: {"image": None, "masks": []})

    for subdir, _, files in os.walk(version0_images_dir):
        seq_name = osp.basename(subdir)
        for filename in sorted(files):
            if filename.endswith(".png"):
                stem = osp.splitext(filename)[0]
                grouped[(seq_name, stem)]["image"] = osp.join(subdir, filename)

    for subdir, _, files in os.walk(version0_masks_dir):
        seq_name = osp.basename(subdir)
        for filename in sorted(files):
            if filename.endswith(".png"):
                stem = re.sub(r"_class\d+$", "", osp.splitext(filename)[0])
                grouped[(seq_name, stem)]["masks"].append(osp.join(subdir, filename))

    return grouped


def generate_augmented_versions(dataset_root, train_subdir, n_versions, predictor):
    version0_root = osp.join(dataset_root, train_subdir, "0") if train_subdir else osp.join(dataset_root, "0")
    version0_images_dir = osp.join(version0_root, "images")
    version0_masks_dir = osp.join(version0_root, "binary_annotations")
    items = collect_version0_items(version0_images_dir, version0_masks_dir)

    scale_factor = 0.2
    rotate_angle = 30
    colour_factor = 0.4

    sorted_items = sorted(items.items())

    for (seq_name, stem), item in tqdm(sorted_items, desc=f"Augmenting {osp.basename(dataset_root)} {train_subdir or 'train'}", unit="frame"):
        if item["image"] is None or not item["masks"]:
            continue

        class_ids = []
        for mask_path in sorted(item["masks"]):
            class_match = re.search(r"_class(\d+)\.png$", osp.basename(mask_path))
            if class_match is not None:
                class_ids.append(int(class_match.group(1)))
        class_ids = sorted(set(class_ids))
        if not class_ids:
            continue

        original_frame = Image.fromarray(load_rgb_image(item["image"]))
        original_masks = []
        mask_names = []
        for mask_path in sorted(item["masks"]):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            original_masks.append(Image.fromarray(np.uint8(mask > 0)))
            mask_names.append(osp.basename(mask_path))

        if not original_masks:
            continue

        width, height = original_frame.size

        for version in tqdm(range(1, n_versions + 1), desc=f"{seq_name}/{stem}", unit="version", leave=False):
            save_root = osp.join(dataset_root, train_subdir, str(version)) if train_subdir else osp.join(dataset_root, str(version))
            if is_augmented_frame_prepared(save_root, seq_name, stem, class_ids, with_transcription=bool(train_subdir)):
                continue

            random.seed(version)
            torch.manual_seed(version)
            np.random.seed(version)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(version)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            scale, rotate, colour = version_to_augmentation_toggles(version, n_versions)
            frame, masks = augmentation(
                original_frame,
                original_masks,
                scale_factor,
                rotate_angle,
                colour_factor,
                height,
                width,
                scale=scale,
                rotate=rotate,
                colour=colour,
            )
            frame = np.asarray(frame)
            masks = [np.asarray(mask) * 255 for mask in masks]
            binary_masks = {}
            for mask_name, mask in zip(mask_names, masks):
                class_match = re.search(r"_class(\d+)\.png$", mask_name)
                if class_match is None:
                    continue
                binary_masks[int(class_match.group(1))] = mask.astype(np.uint8)

            feat, embeddings = compute_feature_and_embeddings(frame, binary_masks, predictor)

            image_path = osp.join(save_root, "images", seq_name, f"{stem}.png")
            feat_path = osp.join(save_root, "sam_features_h", seq_name, f"{stem}.npy")
            save_image(image_path, frame)
            ensure_dir(osp.dirname(feat_path))
            np.save(feat_path, feat)

            if train_subdir:
                transcription_path = osp.join(save_root, "transcriptions", seq_name, f"{stem}.txt")
                ensure_dir(osp.dirname(transcription_path))
                if not osp.exists(transcription_path):
                    with open(transcription_path, "w", encoding="utf-8") as f:
                        f.write("")

            for class_id, mask in binary_masks.items():
                mask_name = f"{stem}_class{class_id}.png"
                mask_path = osp.join(save_root, "binary_annotations", seq_name, mask_name)
                embedding_path = osp.join(save_root, "class_embeddings_h", seq_name, mask_name.replace(".png", ".npy"))
                save_mask(mask_path, mask)
                if class_id in embeddings:
                    ensure_dir(osp.dirname(embedding_path))
                    np.save(embedding_path, embeddings[class_id])


def build_predictor(sam_checkpoint, device):
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["endovis_2017", "endovis_2018", "all"], default="all")
    parser.add_argument("--raw-data-root", default="../../data")
    parser.add_argument("--output-root", default="../../data/train_data")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--n-version", type=int, default=40)
    parser.add_argument("--skip-augment", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_predictor(args.sam_checkpoint, device)

    if args.dataset in ("endovis_2017", "all"):
        raw_root = osp.join(args.raw_data_root, "endovis2017")
        convert_endovis17_base(raw_root, args.output_root, predictor)
        if not args.skip_augment:
            generate_augmented_versions(osp.join(args.output_root, "endovis_2017"), "", args.n_version, predictor)

    if args.dataset in ("endovis_2018", "all"):
        raw_root = osp.join(args.raw_data_root, "endovis2018")
        convert_endovis18_base(raw_root, args.output_root, predictor)
        if not args.skip_augment:
            generate_augmented_versions(osp.join(args.output_root, "endovis_2018"), "train", args.n_version, predictor)


if __name__ == "__main__":
    main()
