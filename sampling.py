import json
import os

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import entropy


import dataset_utils as myset
import score_generate as sg

def find_ok_path(path):
    if '.' in path:
        parts = path.split(".")
        suf = "." + parts[-1]
        path = ".".join(parts[:-1])
    else:
        suf = ""

    ok_path = path
    num_suf = 1
    while os.path.exists(ok_path + suf):
        ok_path = path + f"_{num_suf}"
        num_suf += 1
    return ok_path + suf

def move_selected(data_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dir_name = os.path.dirname(data_file)
    df = pd.read_json(data_file, lines=True)
    # save_dir = str(os.path.join(dir_name, cls))
    for file in df["file_name"].to_list():
        ex_name = os.path.join(dir_name, file)
        new_name = os.path.join(save_dir, file)
        os.system(f"cp {ex_name} {new_name}")

# scale a distribution to the target IPC
def get_sampling_distribution(data, ipc):
    bins = np.arange(0, 1.1, 0.1)
    arr, _ = np.histogram(data, bins=bins)
    scaled = arr * (ipc / arr.sum())
    target = np.floor(scaled).astype(int)
    n_diff = int(ipc - target.sum())
    # 给小数值较大的补上1
    indices = np.argsort(-1 * (scaled - target))
    target[indices[:n_diff]] += 1
    return target

def log_transform(data, th_b=10, th_t=10):

    # for new arrays, do log_max (value) to get results in (0,1]
    inf = 1e-4
    data = np.array(data)
    data_sorted = sorted(data)
    bottom = data_sorted[th_b]
    # avoid the appearance of log 0
    bottom = inf if bottom == 0 else bottom
    top = data_sorted[-1 * (th_t + 1)]
    data[data < bottom] = bottom
    data[data > top] = top

    # divide min value to make values >=1
    data = data / bottom
    base = top / bottom
    data_trans = np.log(data) / np.log(base)
    # avoid the appearance of 0
    data_trans[data_trans < inf] = inf
    return data_trans

def get_threshold(meta_file, size=50, ip_name=None, cls="", save_file="threshold.jsonl"):
    l_value = 0.5
    values = []
    for th_b in range(size // 2):
    # for th_b in [20]:
        values.append([])
        for th_t in range(size // 2):
        # for th_t in [32, 33, 34, 35, 36, 37]:
            with open(meta_file) as f:
                df = pd.read_json(f, lines=True)
            scores = list(df["score"])
            hist_ori, _ = np.histogram(scores, bins=10, range=(0, 1), density=True)
            scores_log = log_transform(scores, th_b=th_b, th_t=th_t)
            hist_log, _ = np.histogram(scores_log, bins=10, range=(0, 1), density=True)
            data_even = np.ones_like(hist_log) / len(hist_log)

            score_sim = entropy(hist_ori, hist_log, base=2)
            score_even = entropy(data_even, hist_log, base=2)
            value = l_value * score_sim + (1 - l_value) * score_even

            values[th_b].append(value)
    arr = np.array(values)
    th_b, th_t = np.unravel_index(np.argmin(arr), arr.shape)

    # save the threshold values for further use if necessary
    if ip_name is not None:
        with open(save_file, "a", encoding="utf-8") as f:
            item = {"set":ip_name, "cls":cls, "b":int(th_b), "t":int(th_t)}
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
    return th_b, th_t

def sampling(ip_data, samp_distr, save_path="", name=""):
    maxs = np.arange(0.1, 1.1, 0.1)
    mins = maxs - 0.1

    df1 = pd.DataFrame()
    df = ip_data
    n_img = np.sum(samp_distr)
    df1 = df[df["score"] == 0]
    for _0, _1, n in zip(mins, maxs, samp_distr):
        df0 = df[df["score"] >= _0]
        if _1 == 1:
            df0 = df0[df0["score"] <= _1]
        else:
            df0 = df0[df0["score"] < _1]
        try:
            df0 = df0.sample(n=n)
        except ValueError:
            pass
            if not name=="":
                print(name)
                print(f"area {_0:.2f}-{_1:.2f}, {len(df0)} pictures found, < demand of {n}")
        df = df[~df['file_name'].isin(df0['file_name'])]
        df1 = pd.concat([df1, df0])
    if not len(df1) == n_img:
        df_append = df.sample(n=n_img - len(df1))
        df1 = pd.concat([df1, df_append])
    save_path = find_ok_path(save_path)
    df1.to_json(save_path, orient="records", lines=True)
    return save_path

def main(args):
    ip_root = args.image_pool
    lists = os.listdir(ip_root)
    ori_root = args.ori_dataset
    sg.generate_metadata(ori_root, lists, suff=args.image_suff)

    classes = os.listdir(ip_root)

    for i, cls in enumerate(classes):
        ip_meta = os.path.join(ip_root, cls, "metadata.jsonl")
        df = pd.read_json(args.th_file, lines=True)
        res = df.loc[(df['set'] == args.ip_name) & (df['cls'] == cls)]
        if len(res) == 0:
            th_b, th_t = get_threshold(ip_meta, size=args.ipc, ip_name=args.ip_name, cls=cls, save_file=args.th_file)
        else:
            th_b, th_t = res.iloc[0]["b"], res.iloc[0]["t"]
        print(f"use bottom_threshold={th_b}, top_threshold={th_t} for class {cls}")

        df = pd.read_json(os.path.join(ori_root, cls, "metadata.jsonl"), lines=True)

        ori_distr = log_transform(list(df["score"]), th_b=th_b, th_t=th_t)
        samp_distr = get_sampling_distribution(ori_distr, args.ipc)

        ip_data = pd.read_json(ip_meta, lines=True)
        ip_data["score"] = log_transform(list(ip_data["score"]), th_b=th_b, th_t=th_t)

        save_path = os.path.join(ip_root, cls, "selected.jsonl")
        for j in range(args.n_sampling):
            dd_root = find_ok_path(args.save_path)
            os.makedirs(dd_root, exist_ok=True)
            if args.show_lack_info:
                sampling(ip_data, samp_distr, save_path=save_path, name=f"{args.ip_name}_{cls}")
            else:
                sampling(ip_data, samp_distr, save_path=save_path)
            move_selected(save_path, os.path.join(dd_root, "train", cls))

    print("main")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_root", type=str, default="")
    parser.add_argument("--ip_name", type=str, default="ImagePool")
    parser.add_argument("--ori-dataset", type=str, default="ImageNet")
    parser.add_argument("--save-path", type=str, default="output")
    parser.add_argument("--ipc", type=int, default=1)
    samp_distr = list(myset.samp_distr.keys()) +["scale"]
    parser.add_argument("--sample-distribution", type=str, choices=samp_distr, default="scale")
    parser.add_argument("--image-suff", type=str, default="JPEG")
    parser.add_argument("--th-file", type=str, default="threshold.jsonl")
    parser.add_argument("--n-sampling", type=int, default=1)
    parser.add_argument("--show-lack-info", action="store_true")


    args = parser.parse_args()
    main(args)
