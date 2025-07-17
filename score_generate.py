import math
import os
import glob

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
import dataset_utils as dicts

def check_distribution(distr, title="", save_name="distr", out_dir="", ylim=None):
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
    plt.savefig(os.path.join(out_dir, f"{save_name}.png"))
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
    cls_name = os.path.basename(os.path.dirname(file))
    print(f"(check) {len(scores)} images in {cls_name} \n"
          f"with average difficulty of {ave_scores:.3f}\n")

    if not area:
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
    bins = list(np.arange(area[0], area[1], space))
    counts, _, patches = plt.hist(scores, bins=bins, color='blue', edgecolor='black')
    for i in range(len(patches)):
        plt.text(bins[i] + space / 2.0, counts[i] + 0.1, str(int(counts[i])), ha='center')
    plt.savefig(os.path.join(os.path.dirname(file), "..", f"{cls_name}.png"))

def check_metadata_dir(root):
    dirs = os.listdir(root)
    for d in dirs:
        d_path = os.path.join(root, d)
        if not os.path.isdir(d_path):
            continue
        data_file = str(os.path.join(d_path, "metadata.jsonl"))
        check_metadata(data_file)

# score an image using $model
# return the true label & corresponding accuracy to obtain the metadata file
def score_img(image, model, real_label=0, target_label=0):
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
        return target_label, probs[0][real_label].item()

# generate metadata for all the directories in the $root path
# set $lists to decide which classes to be generated
def generate_metadata(root, lists=None, suff="png", override=False):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    save_name = "metadata.jsonl"
    if lists is not None:
        classes = lists
    else:
        classes = sorted(os.listdir(root))
    for i, c in enumerate(classes):
        c_path = str(os.path.join(root, c))
        if not os.path.isdir(c_path):
            continue
        save_path = os.path.join(c_path, save_name)
        if os.path.exists(save_path):
            print(f"metadata existed in {c}")
            if not override:
                continue
        imagenet_label = dicts.nette_id2label[c]
        images = sorted(glob.glob(os.path.join(c_path, f"*.{suff}")))
        assert len(images) > 0
        names, labels, scores = [], [], []
        for image in images:
            label, prob = score_img(image, model, real_label=imagenet_label, target_label=i)
            names.append(os.path.basename(image))
            labels.append(label)
            scores.append(1 - prob)
        with open(save_path, "w") as f:
            for num in range(len(images)):
                f.write(f'{{"file_name": "{names[num]}", "label": "{labels[num]}", "score": {scores[num]:.4f}}}\n')

    print(f"metadata generated for {len(classes)} classes in {os.path.dirname(root)}")


def main():
    print("main")
    exit()
    root = "/home/user/Sumomo/Project/exp_ICCVW2025/woof_250_50_3/train"
    generate_metadata(root)
    for cls in os.listdir(root):
        check_metadata(os.path.join(root, cls, "metadata.jsonl"))
    exit()
    # root = "/home/user/Sumomo/Project/Dataset/mini_woof_cur50/train"
    # root = "/home/user/Sumomo/Project/Output/score_nette_s10/sync/test"\

    # temp batch-procs

    for i in [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]:
    # for i in [20, 50]:
        root = f"/home/user/Sumomo/Project/exp_ICCVW2025/woof{i}"
        generate_metadata(root, suff="png", for_check=False, override=False)
    exit()


    # # 给一个文件夹里的指定类创建metadata
    # lists_woof = ["n02086240", "n02088364", "n02093754", "n02099601", "n02111889",
    #               "n02087394", "n02089973", "n02096294", "n02105641", "n02115641"]
    # generate_metadata(root, lists=lists_woof, suff="JPEG", for_check=False, override=True)
    # for cls in sorted(os.listdir(root)):
    #     d_path = os.path.join(root, cls)
    #     if not os.path.isdir(d_path):
    #         continue
    #     data_file = str(os.path.join(d_path, "metadata.jsonl"))
    #     check_metadata(data_file, area=[0, 0.11], space=0.01)

    # 给一个文件夹里的所有类创建metadata
    root = "/home/user/Sumomo/Project/Dataset/imageNette/train"
    generate_metadata(root, suff="JPEG", for_check=False, override=True)
    for cls in sorted(os.listdir(root)):
        d_path = os.path.join(root, cls)
        if not os.path.isdir(d_path):
            continue
        data_file = str(os.path.join(d_path, "metadata.jsonl"))
        check_metadata(data_file, area=[0, 0.11], space=0.01)
        check_metadata(data_file, area=[0, 1.1], space=0.1)

    # 给一个图片评分
    # image = "/home/user/Sumomo/Project/Dataset/test/n03417042/n03417042_11.JPEG"
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # model.eval()
    # print(score_img(image, model, label_type="real"))



if __name__ == "__main__":
    main()