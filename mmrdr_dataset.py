"""Original PyTorch dataset loader for locked MMRDR manifest CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

LESIONS = ("MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD")
REQUIRED = ("image", "modality", "grade", *LESIONS, "lr", "split")
ROOTS = {
    "CFP": Path(r"C:\Datasets\MMRDR\MMRDR-CFP"),
    "UWF": Path(r"C:\Datasets\MMRDR\MMRDR-UWF"),
}


class PadToSquare:
    """Pad without cropping so peripheral retinal content is retained."""

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        canvas = Image.new("RGB", (side, side), (self.fill,) * 3)
        canvas.paste(image, (left, top))
        return canvas


def build_transform(image_size: int = 512, training: bool = False):
    steps = [PadToSquare()]
    if training:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    steps += [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


class MMRDRDataset(Dataset):
    def __init__(self, manifest, roots=None, transform=None):
        self.manifest = Path(manifest)
        self.frame = pd.read_csv(self.manifest)
        missing = sorted(set(REQUIRED) - set(self.frame.columns))
        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")
        modalities = set(self.frame["modality"].unique())
        if len(modalities) != 1:
            raise ValueError(f"Manifest must contain one modality, found {modalities}")
        self.modality = modalities.pop()
        self.roots = ROOTS if roots is None else roots
        if self.modality not in self.roots:
            raise ValueError(f"No image root configured for {self.modality}")
        self.transform = transform or build_transform(training=False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.roots[self.modality] / Path(row["image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        return {
            "image": self.transform(image),
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "lesions": torch.tensor(row[list(LESIONS)].to_numpy(dtype="float32")),
            "laterality": torch.tensor(int(row["lr"]), dtype=torch.long),
            "path": str(path),
        }


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a locked MMRDR manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    dataset = MMRDRDataset(
        args.manifest,
        transform=build_transform(args.image_size, args.training),
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers)))
    print("manifest:", args.manifest)
    print("samples:", len(dataset), "modality:", dataset.modality)
    print("images:", tuple(batch["image"].shape), batch["image"].dtype)
    print("grades:", tuple(batch["grade"].shape), batch["grade"].tolist())
    print("lesions:", tuple(batch["lesions"].shape))
    print("finite:", bool(torch.isfinite(batch["image"]).all()))
    print("first path:", batch["path"][0])


if __name__ == "__main__":
    main()
