import math
import os
from pathlib import Path

import torch

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as func
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import dataset_utils as util

def check_distribution(distr, out_dir, save_name="distr", title="", ylim=None):
    bins = np.linspace(0, 1, len(distr) + 1)
    centers = [(bins[i] + bins[i+1])/2 for i in range((len(distr)))]
    plt.figure(figsize=(8,8))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots()
    ax.bar(centers, distr, width=0.1, align='center', color= '#BBD5E8', edgecolor='#2E75B6')
    for i in range(len(distr)):
        plt.text(centers[i], distr[i] + 0.1, str(int(distr[i])), ha='center')
    plt.title(title, fontdict={'size':24})
    # plt.xlabel("Difficulty Area", fontdict={'size':20})
    # plt.ylabel("Count", fontdict={'size':20})
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    y_max = math.ceil(max(distr) * 1.1)
    if ylim is None or ylim[1] < y_max:
        plt.ylim((0, y_max))
    else:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"{save_name}.png"))
    plt.close()

def check_metadata_df(df, save_name, out_dir):
    scores = list(df["score"])
    ave_scores = np.average(scores)
    print(f"(check) {len(scores)} images in {save_name} \n"
          f"with average difficulty of {ave_scores:.3f}\n")
    area = [0, 1.1]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # plt.title(f"class: {cls}")
    plt.xlim([-0.2, 1.2])
    plt.xticks(np.arange(0, 1.1, 0.1))

    sns.kdeplot(scores, fill=True)

    plt.subplot(1, 2, 2)
    plt.title(f"ave:{ave_scores:.3f}")
    # bins = list(np.linspace(0, 1, 11))
    bins = list(np.arange(area[0], area[1], 0.1))
    counts, _, patches = plt.hist(scores, bins=bins, color='skyblue', edgecolor='black')
    for i in range(len(patches)):
        plt.text(bins[i] + 0.1 / 2.0, counts[i] + 0.1, str(int(counts[i])), ha='center')
    plt.savefig(os.path.join(out_dir, f"{save_name}.png"))
    plt.close()

# generate the figure of difficulty distribution according to the metadata
def check_metadata(file, area=None, space=0.1):
    df = pd.read_json(file, lines=True)
    scores = list(df["score"])
    ave_scores = np.average(scores)
    cls_name = file.parent.name
    print(f"(check) {len(scores)} images in {cls_name} \n"
          f"with average difficulty of {ave_scores:.3f}\n")

    if area is None:
        area = [0, 1.1]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # plt.title(f"class: {cls}")
    plt.xlim([-0.2, 1.2])
    plt.xticks(np.arange(0, 1.1, 0.1))

    sns.kdeplot(scores, fill=True)

    plt.subplot(1, 2, 2)
    plt.title(f"ave:{ave_scores:.3f}")
    bins = list(np.arange(area[0], area[1], space))
    counts, _, patches = plt.hist(scores, bins=bins, color='blue', edgecolor='black')
    for i in range(len(patches)):
        plt.text(bins[i] + space / 2.0, counts[i] + 0.1, str(int(counts[i])), ha='center')
    plt.savefig(file.parent.parent.joinpath(f"{cls_name}.png"))

def check_metadata_dir(root_path):
    for cls_path in root_path.iterdir():
        if cls_path.is_dir():
            meta_file = cls_path.joinpath("metadata.jsonl")
            check_metadata(meta_file)


def score_img(image, model, real_label=0):
    if  isinstance(real_label, str):
        real_label = int(real_label)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = func.softmax(outputs, dim=1)
        return probs[0][real_label].item()


def generate_metadata(root_path, lists=None, override=False):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    save_name = "metadata.jsonl"
    n = 0
    classes = sorted([f.name for f in root_path.iterdir() if f.is_dir()]) if lists is None else lists
    for i, c in enumerate(classes):
        c_path = root_path.joinpath(c)
        save_path = c_path.joinpath(save_name)
        if Path.exists(save_path):
            if override:
                print(f"metadata existed in {c}")
            else:
                continue
        imagenet_label = util.imagenet_id2label[c]

        images = [f for suff in util.image_suff for f in c_path.glob(f"*.{suff}")]
        assert len(images) > 0
        names, labels, scores = [], [], []
        for image in images:
            prob = score_img(image, model, real_label=imagenet_label)
            names.append(image.name)
            labels.append(i)
            scores.append(1 - prob)
        with open(save_path, "w") as f:
            for num in range(len(images)):
                f.write(f'{{"file_name": "{names[num]}", "label": "{labels[num]}", "score": {scores[num]:.4f}}}\n')
        n += 1

    if n > 0:
        print(f"metadata generated for {n} classes in {root_path.parent}")


def main():
    p = Path()
    p = p.joinpath("test")
    p.mkdir()
    print(p.is_dir())
    print(type(p.stem))
    print(f"{__name__}\n{__file__}\n")

if __name__ == "__main__":
    main()