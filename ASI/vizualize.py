import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageTk

from CLIP import clip
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from model_forward_test import model_forward_function
from segment_anything import sam_model_registry


ASI_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
CKP_ROOT = PROJECT_ROOT / "ckp"
WORK_DIRS_ROOT = ASI_ROOT / "work_dirs"

FOLD_SEQ = {
    0: [1, 3],
    1: [2, 5],
    2: [4, 8],
    3: [6, 7],
}

DATASET_CONFIGS = {
    "endovis_2017": {
        "num_class": 7,
        "num_tokens": 4,
        "class_names": [
            "Bipolar Forceps",
            "Prograsp Forceps",
            "Large Needle Driver",
            "Vessel Sealer",
            "Grasping Retractor",
            "Monopolar Curved Scissors",
            "Other Instruments",
        ],
        "texts": [
            "This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.",
            "This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.",
            "This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.",
            "This is a Vessel Sealer, specifically designed for the permanent closure of blood vessels. It uses advanced energy technology to fuse vessel walls, ensuring a secure seal with minimal thermal spread. Its ergonomic design and precise control make it ideal for a variety of surgical procedures, including laparoscopic operations and complex dissections, where control and reliability are paramount.",
            "This is a Grasping Retractor, engineered for optimal exposure and manipulation of tissues and organs during surgery. Its design allows for a firm grip and controlled retraction, minimizing tissue trauma. The device's versatility and durability make it a staple in surgeries requiring precision and care, such as abdominal and thoracic procedures.",
            "These are Monopolar Curved Scissors, a vital tool for cutting and dissecting tissues with precision. Integrated with monopolar energy, they allow surgeons to cut and coagulate simultaneously, reducing bleeding and operation time. The curved design enhances visibility and access in tight spaces, making them indispensable in minimally invasive surgeries and complex anatomical areas.",
            "This category includes various Other Medical Instruments tailored for specific surgeries, from diagnostic aids to tools like suction devices, clip appliers, and retractors for better exposure. Designed for precision, patient care, and meeting modern surgery's demands, these essential tools support cutting, clamping, retracting, and coagulating, showcasing surgical diversity and technological advancement.",
        ],
    },
    "endovis_2018": {
        "num_class": 7,
        "num_tokens": 2,
        "class_names": [
            "Bipolar Forceps",
            "Prograsp Forceps",
            "Large Needle Driver",
            "Monopolar Curved Scissors",
            "Ultrasound Probe",
            "Suction Instrument",
            "Clip Applier",
        ],
        "texts": [
            "This is a Bipolar Forceps, commonly used for coagulating tissues and vessels with precision. Its insulated shaft and fine tips allow for delicate tissue manipulation and reduced thermal spread during surgeries such as microdissections and neurosurgery.",
            "This is a Prograsp Forceps, designed for a firm grip and precise manipulation of tissues and organs during laparoscopic procedures. Its ergonomic design ensures steady handling and its durable construction provides a reliable performance in complex surgeries.",
            "This is a Large Needle Driver, perfect for suturing with its strong, stable grip and precise control. It's particularly useful in procedures requiring large sutures, providing surgeons with the precision and durability needed for effective tissue approximation.",
            "This is a Monopolar Curved Scissors, an essential tool for precise cutting and dissection, offering monopolar energy for cauterization, reducing bleeding risks. Its curved design allows for fine, detailed movements, ideal for intricate surgical areas.",
            "This is an Ultrasound Probe, a high-resolution imaging tool crucial for diagnostic and intraoperative procedures. It provides real-time images of internal structures, guiding interventions and ensuring precision in procedures like biopsies or fluid aspirations.",
            "This is a Suction Instrument, designed to efficiently remove fluids and debris from the surgical site, maintaining a clear view for the surgeon. Its ergonomic design and reliable performance make it a staple in maintaining operative field clarity.",
            "This is a Clip Applier, a vital tool for quickly and securely ligating vessels during surgery. Its precise mechanism ensures the safe and effective deployment of clips, reducing the risk of complications and promoting efficient hemostasis.",
        ],
    },
}

MASK_COLORS = {
    1: (230, 57, 70),
    2: (29, 78, 216),
    3: (22, 163, 74),
    4: (249, 115, 22),
    5: (168, 85, 247),
    6: (14, 165, 233),
    7: (234, 179, 8),
}

VIEW_MODES = [
    "Prediction Overlay",
    "Ground Truth Overlay",
    "Prediction Only",
    "Ground Truth Only",
    "Input Only",
    "Input | Prediction | Ground Truth",
]


def resolve_default_checkpoint(dataset_name: str, fold: int) -> Path | None:
    candidates = []
    if dataset_name == "endovis_2017":
        candidates.append(WORK_DIRS_ROOT / dataset_name / str(fold) / "model_ckp.pth")
    else:
        candidates.append(WORK_DIRS_ROOT / dataset_name / "model_ckp.pth")

    candidates.extend(sorted(WORK_DIRS_ROOT.glob(f"{dataset_name}*/**/model_ckp.pth")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def seq_sort_key(name: str):
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else name


def blend_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float, class_names: list[str]) -> Image.Image:
    base = image_rgb.astype(np.float32)
    overlay = base.copy()

    for class_id, color in MASK_COLORS.items():
        class_pixels = mask == class_id
        if np.any(class_pixels):
            overlay[class_pixels] = np.array(color, dtype=np.float32)

    mixed = np.where(mask[..., None] > 0, (1.0 - alpha) * base + alpha * overlay, base)
    result = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))

    legend_height = 32 + 24 * len(class_names)
    canvas = Image.new("RGB", (result.width, result.height + legend_height), (245, 245, 245))
    canvas.paste(result, (0, 0))

    draw = ImageDrawSafe(canvas)
    draw.text((12, result.height + 8), "Classes", fill=(30, 30, 30))
    for idx, class_name in enumerate(class_names, start=1):
        y = result.height + 12 + idx * 20
        color = MASK_COLORS[idx]
        draw.rectangle((12, y, 28, y + 12), fill=color)
        draw.text((38, y - 2), f"{idx}: {class_name}", fill=(30, 30, 30))

    return canvas


class ImageDrawSafe:
    def __init__(self, image: Image.Image):
        from PIL import ImageDraw

        self._draw = ImageDraw.Draw(image)

    def text(self, xy, text, fill):
        self._draw.text(xy, text, fill=fill)

    def rectangle(self, box, fill):
        self._draw.rectangle(box, fill=fill)


def mask_to_color(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in MASK_COLORS.items():
        rgb[mask == class_id] = color
    return Image.fromarray(rgb)


def compose_horizontal(images: list[Image.Image], titles: list[str]) -> Image.Image:
    from PIL import ImageDraw

    gap = 12
    title_height = 28
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images) + title_height
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for image, title in zip(images, titles):
        canvas.paste(image, (x, title_height))
        draw.text((x + 8, 6), title, fill=(30, 30, 30))
        x += image.width + gap
    return canvas


class SurgicalSegModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_key = None
        self.clip_model = None
        self.sam_prompt_encoder = None
        self.sam_decoder = None
        self.prototype_prompt_encoder = None
        self.learnable_prototypes = None
        self.prototypes = None

    def ensure_loaded(self, dataset_name: str, fold: int, checkpoint_path: Path):
        key = (dataset_name, fold, str(checkpoint_path.resolve()))
        if self.loaded_key == key:
            return

        config = DATASET_CONFIGS[dataset_name]
        self.clip_model, _ = clip.load("ViT-L/14", device=self.device)
        sam_checkpoint = CKP_ROOT / "sam" / "sam_vit_h_4b8939.pth"
        model_type = "vit_h_no_image_encoder"

        self.sam_prompt_encoder, self.sam_decoder = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
        self.sam_prompt_encoder.to(self.device)
        self.sam_decoder.to(self.device)
        self.sam_prompt_encoder.eval()
        self.sam_decoder.eval()

        self.learnable_prototypes = Learnable_Prototypes(
            num_classes=config["num_class"],
            feat_dim=256,
        ).to(self.device)
        self.prototype_prompt_encoder = Prototype_Prompt_Encoder(
            feat_dim=256,
            hidden_dim_dense=128,
            hidden_dim_sparse=128,
            size=64,
            num_tokens=config["num_tokens"],
            num_class=config["num_class"],
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.prototype_prompt_encoder.load_state_dict(checkpoint["prototype_prompt_encoder_state_dict"])
        self.sam_decoder.load_state_dict(checkpoint["sam_decoder_state_dict"])
        self.learnable_prototypes.load_state_dict(checkpoint["prototypes_state_dict"])

        for module in [
            self.sam_prompt_encoder,
            self.sam_decoder,
            self.prototype_prompt_encoder,
            self.learnable_prototypes,
        ]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        with torch.no_grad():
            text_tokens = clip.tokenize(config["texts"]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens).to(self.device)
            self.prototypes = self.learnable_prototypes(text_features)

        self.loaded_key = key

    def predict_frame(self, sam_feat_path: Path, output_shape: tuple[int, int], num_class: int, threshold: float = 0.0):
        if self.prototypes is None:
            raise RuntimeError("Model is not loaded.")

        sam_feat = np.load(sam_feat_path)
        sam_feat_tensor = torch.from_numpy(sam_feat).float()
        if sam_feat_tensor.ndim != 3:
            raise ValueError(f"Unexpected SAM feature shape: {sam_feat_tensor.shape}")

        sam_batch = sam_feat_tensor.unsqueeze(0).repeat(num_class, 1, 1, 1).to(self.device)
        cls_ids = torch.arange(1, num_class + 1, device=self.device)

        with torch.no_grad():
            preds, pred_quality = model_forward_function(
                self.prototype_prompt_encoder,
                self.sam_prompt_encoder,
                self.sam_decoder,
                sam_batch,
                self.prototypes,
                cls_ids,
            )

        preds = preds.detach().cpu().numpy()
        pred_quality = pred_quality.detach().cpu().numpy()

        combined = np.zeros(preds.shape[-2:], dtype=np.uint8)
        pred_masks = preds > threshold
        for order_index in np.argsort(pred_quality):
            class_id = order_index + 1
            combined[pred_masks[order_index]] = class_id

        if combined.shape != output_shape:
            combined = cv2.resize(
                combined,
                (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return combined, pred_quality


class VisualizationApp:
    def __init__(self, root: tk.Tk, initial_dataset: str, initial_fold: int, initial_checkpoint: str | None):
        self.root = root
        self.root.title("ASI Surgical Segmentation Viewer")
        self.root.geometry("1600x950")

        self.model = SurgicalSegModel()
        self.prediction_cache = {}
        self.tree_nodes = {}
        self.node_to_meta = {}
        self.current_photo = None
        self.current_selection = None

        self.dataset_var = tk.StringVar(value=initial_dataset)
        self.split_var = tk.StringVar(value="val")
        self.fold_var = tk.IntVar(value=initial_fold)
        self.view_var = tk.StringVar(value=VIEW_MODES[0])
        self.alpha_var = tk.DoubleVar(value=0.45)
        default_ckpt = initial_checkpoint or str(resolve_default_checkpoint(initial_dataset, initial_fold) or "")
        self.checkpoint_var = tk.StringVar(value=default_ckpt)

        self._build_ui()
        self.refresh_tree()

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Dataset").grid(row=0, column=0, sticky="w")
        dataset_combo = ttk.Combobox(
            controls,
            textvariable=self.dataset_var,
            values=sorted(DATASET_CONFIGS.keys()),
            state="readonly",
            width=18,
        )
        dataset_combo.grid(row=0, column=1, padx=(6, 12))
        dataset_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_dataset_changed())

        ttk.Label(controls, text="Split").grid(row=0, column=2, sticky="w")
        split_combo = ttk.Combobox(
            controls,
            textvariable=self.split_var,
            values=["train", "val"],
            state="readonly",
            width=10,
        )
        split_combo.grid(row=0, column=3, padx=(6, 12))
        split_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_tree())

        ttk.Label(controls, text="Fold").grid(row=0, column=4, sticky="w")
        fold_combo = ttk.Combobox(
            controls,
            textvariable=self.fold_var,
            values=[0, 1, 2, 3],
            state="readonly",
            width=8,
        )
        fold_combo.grid(row=0, column=5, padx=(6, 12))
        fold_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_tree())

        ttk.Label(controls, text="View").grid(row=0, column=6, sticky="w")
        view_combo = ttk.Combobox(
            controls,
            textvariable=self.view_var,
            values=VIEW_MODES,
            state="readonly",
            width=28,
        )
        view_combo.grid(row=0, column=7, padx=(6, 12))
        view_combo.bind("<<ComboboxSelected>>", lambda _event: self.render_current_selection())

        ttk.Label(controls, text="Overlay").grid(row=0, column=8, sticky="w")
        alpha_scale = ttk.Scale(
            controls,
            from_=0.15,
            to=0.85,
            variable=self.alpha_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=lambda _value: self.render_current_selection(),
        )
        alpha_scale.grid(row=0, column=9, padx=(6, 12))

        ttk.Label(controls, text="Checkpoint").grid(row=1, column=0, sticky="w", pady=(10, 0))
        checkpoint_entry = ttk.Entry(controls, textvariable=self.checkpoint_var, width=100)
        checkpoint_entry.grid(row=1, column=1, columnspan=8, sticky="ew", padx=(6, 12), pady=(10, 0))

        browse_button = ttk.Button(controls, text="Browse", command=self.browse_checkpoint)
        browse_button.grid(row=1, column=9, pady=(10, 0))

        reload_button = ttk.Button(controls, text="Reload Model", command=self.reload_model_and_render)
        reload_button.grid(row=1, column=10, padx=(12, 0), pady=(10, 0))

        refresh_button = ttk.Button(controls, text="Refresh Data", command=self.refresh_tree)
        refresh_button.grid(row=0, column=10, padx=(12, 0))

        controls.columnconfigure(8, weight=1)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=10)
        right = ttk.Frame(body, padding=10)
        body.add(left, weight=1)
        body.add(right, weight=4)

        ttk.Label(left, text="Select Input Frame").pack(anchor="w")
        self.tree = ttk.Treeview(left, show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(right, textvariable=self.status_var).pack(anchor="w")

        self.image_label = ttk.Label(right, anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def on_dataset_changed(self):
        dataset_name = self.dataset_var.get()
        default_ckpt = resolve_default_checkpoint(dataset_name, self.fold_var.get())
        if default_ckpt:
            self.checkpoint_var.set(str(default_ckpt))
        self.refresh_tree()

    def browse_checkpoint(self):
        selected = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
            initialdir=str(WORK_DIRS_ROOT),
        )
        if selected:
            self.checkpoint_var.set(selected)

    def reload_model_and_render(self):
        self.model.loaded_key = None
        self.prediction_cache.clear()
        self.render_current_selection()

    def _get_data_root(self) -> Path:
        return DATA_ROOT / "train_data" / self.dataset_var.get()

    def _get_sequence_filter(self):
        dataset_name = self.dataset_var.get()
        split = self.split_var.get()
        if dataset_name != "endovis_2017":
            return None
        val_seqs = {f"seq{seq}" for seq in FOLD_SEQ[self.fold_var.get()]}
        if split == "val":
            return val_seqs
        return {f"seq{seq}" for seq in range(1, 11)} - val_seqs

    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        self.tree_nodes.clear()
        self.node_to_meta.clear()
        self.current_selection = None

        dataset_name = self.dataset_var.get()
        split = self.split_var.get()
        sequence_filter = self._get_sequence_filter()
        root = self._get_data_root()

        try:
            if dataset_name == "endovis_2017":
                version_dirs = sorted([path for path in root.iterdir() if path.is_dir() and path.name.isdigit()], key=lambda path: int(path.name))
                for version_dir in version_dirs:
                    image_root = version_dir / "images"
                    if not image_root.is_dir():
                        continue
                    version_node = self.tree.insert("", "end", text=f"version {version_dir.name}", open=False)
                    for seq_dir in sorted([path for path in image_root.iterdir() if path.is_dir()], key=lambda path: seq_sort_key(path.name)):
                        if sequence_filter and seq_dir.name not in sequence_filter:
                            continue
                        seq_node = self.tree.insert(version_node, "end", text=seq_dir.name, open=False)
                        for frame_path in sorted(seq_dir.glob("*.png")):
                            frame_node = self.tree.insert(seq_node, "end", text=frame_path.name)
                            self.node_to_meta[frame_node] = {
                                "dataset": dataset_name,
                                "split": split,
                                "version": version_dir.name,
                                "sequence": seq_dir.name,
                                "frame_path": frame_path,
                            }
            else:
                split_root = root / split
                if split == "train":
                    version_dirs = sorted([path for path in split_root.iterdir() if path.is_dir() and path.name.isdigit()], key=lambda path: int(path.name))
                    for version_dir in version_dirs:
                        image_root = version_dir / "images"
                        if not image_root.is_dir():
                            continue
                        version_node = self.tree.insert("", "end", text=f"version {version_dir.name}", open=False)
                        for seq_dir in sorted([path for path in image_root.iterdir() if path.is_dir()], key=lambda path: seq_sort_key(path.name)):
                            seq_node = self.tree.insert(version_node, "end", text=seq_dir.name, open=False)
                            for frame_path in sorted(seq_dir.glob("*.png")):
                                frame_node = self.tree.insert(seq_node, "end", text=frame_path.name)
                                self.node_to_meta[frame_node] = {
                                    "dataset": dataset_name,
                                    "split": split,
                                    "version": version_dir.name,
                                    "sequence": seq_dir.name,
                                    "frame_path": frame_path,
                                }
                else:
                    image_root = split_root / "images"
                    for seq_dir in sorted([path for path in image_root.iterdir() if path.is_dir()], key=lambda path: seq_sort_key(path.name)):
                        seq_node = self.tree.insert("", "end", text=seq_dir.name, open=False)
                        for frame_path in sorted(seq_dir.glob("*.png")):
                            frame_node = self.tree.insert(seq_node, "end", text=frame_path.name)
                            self.node_to_meta[frame_node] = {
                                "dataset": dataset_name,
                                "split": split,
                                "version": None,
                                "sequence": seq_dir.name,
                                "frame_path": frame_path,
                            }
        except FileNotFoundError:
            self.status_var.set(f"Data root not found: {root}")
            return

        top_nodes = self.tree.get_children()
        if top_nodes:
            self.tree.item(top_nodes[0], open=True)
        self.status_var.set("Select a frame from the tree.")

    def on_tree_select(self, _event):
        selection = self.tree.selection()
        if not selection:
            return
        node = selection[0]
        if node not in self.node_to_meta:
            return
        self.current_selection = self.node_to_meta[node]
        self.render_current_selection()

    def _resolve_paths(self, meta: dict):
        dataset_name = meta["dataset"]
        split = meta["split"]
        sequence = meta["sequence"]
        frame_path = meta["frame_path"]
        frame_name = frame_path.name
        frame_stem = frame_path.stem
        version = meta["version"]

        if dataset_name == "endovis_2017":
            version_root = self._get_data_root() / version
            image_path = version_root / "images" / sequence / frame_name
            sam_feat_path = version_root / "sam_features_h" / sequence / f"{frame_stem}.npy"
            gt_path = version_root / "annotations" / sequence / frame_name
        else:
            base_root = self._get_data_root() / split
            if split == "train":
                base_root = base_root / version
            image_path = base_root / "images" / sequence / frame_name
            sam_feat_path = base_root / "sam_features_h" / sequence / f"{frame_stem}.npy"
            gt_path = base_root / "annotations" / sequence / frame_name

        return image_path, sam_feat_path, gt_path

    def _load_gt_mask(self, gt_path: Path, image_shape: tuple[int, int]):
        if gt_path.is_file():
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                if gt_mask.shape != image_shape:
                    gt_mask = cv2.resize(gt_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                return gt_mask.astype(np.uint8)
        return np.zeros(image_shape, dtype=np.uint8)

    def _build_render_image(self, image_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, class_names: list[str]):
        alpha = float(self.alpha_var.get())
        view_mode = self.view_var.get()

        input_image = Image.fromarray(image_rgb)
        pred_overlay = blend_overlay(image_rgb, pred_mask, alpha, class_names)
        gt_overlay = blend_overlay(image_rgb, gt_mask, alpha, class_names)
        pred_only = mask_to_color(pred_mask)
        gt_only = mask_to_color(gt_mask)

        if view_mode == "Prediction Overlay":
            return pred_overlay
        if view_mode == "Ground Truth Overlay":
            return gt_overlay
        if view_mode == "Prediction Only":
            return pred_only
        if view_mode == "Ground Truth Only":
            return gt_only
        if view_mode == "Input Only":
            return input_image
        return compose_horizontal(
            [input_image, pred_overlay, gt_overlay],
            ["Input", "Prediction", "Ground Truth"],
        )

    def render_current_selection(self):
        if not self.current_selection:
            return

        meta = self.current_selection
        dataset_name = meta["dataset"]
        fold = self.fold_var.get()
        config = DATASET_CONFIGS[dataset_name]
        checkpoint_text = self.checkpoint_var.get().strip()
        if not checkpoint_text:
            self.status_var.set("Select a checkpoint first.")
            return

        checkpoint_path = Path(checkpoint_text)
        if not checkpoint_path.is_file():
            self.status_var.set(f"Checkpoint not found: {checkpoint_path}")
            return

        image_path, sam_feat_path, gt_path = self._resolve_paths(meta)
        if not image_path.is_file():
            self.status_var.set(f"Image not found: {image_path}")
            return
        if not sam_feat_path.is_file():
            self.status_var.set(f"SAM feature not found: {sam_feat_path}")
            return

        self.status_var.set(f"Running inference for {meta['sequence']}/{image_path.name} ...")
        self.root.update_idletasks()

        try:
            self.model.ensure_loaded(dataset_name, fold, checkpoint_path)

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_shape = image_rgb.shape[:2]

            cache_key = (dataset_name, fold, str(checkpoint_path.resolve()), str(sam_feat_path))
            if cache_key not in self.prediction_cache:
                self.prediction_cache[cache_key] = self.model.predict_frame(
                    sam_feat_path=sam_feat_path,
                    output_shape=image_shape,
                    num_class=config["num_class"],
                )
            pred_mask, pred_quality = self.prediction_cache[cache_key]
            gt_mask = self._load_gt_mask(gt_path, image_shape)

            rendered = self._build_render_image(
                image_rgb=image_rgb,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                class_names=config["class_names"],
            )
            rendered = ImageOps.contain(rendered, (1100, 820))
            self.current_photo = ImageTk.PhotoImage(rendered)
            self.image_label.configure(image=self.current_photo)

            qualities = ", ".join(f"{idx + 1}:{score:.3f}" for idx, score in enumerate(pred_quality))
            self.status_var.set(
                f"{dataset_name} | {meta['split']} | {meta['sequence']}/{image_path.name} | qualities [{qualities}]"
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Visualization error", str(exc))


def parse_args():
    parser = argparse.ArgumentParser(description="Show ASI segmentation results in a GUI.")
    parser.add_argument("--dataset", default="endovis_2017", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--fold", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--checkpoint", default=None, help="Path to a trained model checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    app = VisualizationApp(
        root=root,
        initial_dataset=args.dataset,
        initial_fold=args.fold,
        initial_checkpoint=args.checkpoint,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
