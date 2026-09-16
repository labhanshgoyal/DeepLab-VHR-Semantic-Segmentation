import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

class CocoSegDataset(Dataset):
    def __init__(self, root_dir, annotation_file, num_classes=10, transform=None):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.getImgIds())

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        w, h = image.size
        mask = np.zeros((h, w, self.num_classes), dtype=np.float32)

        for ann in anns:
            cat_id = ann["category_id"] - 1
            binary_mask = self.coco.annToMask(ann)
            mask[:, :, cat_id] = np.maximum(mask[:, :, cat_id], binary_mask)

        # convert mask to single-channel index image for PIL compatibility
        # background = 0, class 1 = 1, class 2 = 2, etc.
        # where multiple classes overlap, take the last one
        mask_index = np.zeros((h, w), dtype=np.uint8)
        for c in range(self.num_classes):
            mask_index[mask[:, :, c] > 0] = c + 1
        mask_pil = Image.fromarray(mask_index)

        if self.transform:
            image, mask_pil = self.transform(image, mask_pil)

        # convert single-channel index mask back to multi-channel binary [C, H, W]
        if isinstance(mask_pil, Image.Image):
            mask_np = np.array(mask_pil)
        elif isinstance(mask_pil, torch.Tensor):
            mask_np = mask_pil.squeeze(0).numpy() if mask_pil.dim() == 3 else mask_pil.numpy()
        else:
            mask_np = mask_pil
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        multi_mask = np.zeros((self.num_classes, mask_np.shape[0], mask_np.shape[1]), dtype=np.float32)
        for c in range(self.num_classes):
            multi_mask[c] = (mask_np == (c + 1)).astype(np.float32)

        return image, torch.from_numpy(multi_mask)

def get_dataloaders(config):
    data_cfg = config["data"]
    from utils.transforms import get_train_transforms, get_val_transforms
    dataset_dir = data_cfg["dataset_dir"]
    images_dir = data_cfg["images_dir"]
    train_dataset = CocoSegDataset(
        root_dir=images_dir,
        annotation_file=os.path.join(dataset_dir, data_cfg["train_annotations"]),
        num_classes=config["model"]["num_classes"],
        transform=get_train_transforms(data_cfg["image_size"]),
    )
    val_dataset = CocoSegDataset(
        root_dir=images_dir,
        annotation_file=os.path.join(dataset_dir, data_cfg["val_annotations"]),
        num_classes=config["model"]["num_classes"],
        transform=get_val_transforms(data_cfg["image_size"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    return train_loader, val_loader