import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from scipy.stats import entropy

import dataset_utils as myset
import score_generate as sg


def find_ok_path(path):
    ok_path = path
    num_suf = 0
    while ok_path.exists():
        ok_path = Path(path.stem + f"_{num_suf}")
        num_suf += 1
    return ok_path


def move_selected(data_file, save_path):
    save_path.mkdir(exist_ok=True)
    df = pd.read_json(data_file, lines=True)
    for file in df["file_name"].to_list() + [data_file.name]:
        ex_path = data_file.parent.joinpath(file)
        new_path = save_path.joinpath(file)
        shutil.copy(ex_path, new_path)



# scale a distribution to the target IPC
def get_sampling_distribution(data, ipc):
    bins = np.arange(0, 1.1, 0.1)
    arr, _ = np.histogram(data, bins=bins)
    scaled = arr * (ipc / arr.sum())
    target = np.floor(scaled).astype(int)
    n_diff = int(ipc - target.sum())
    indices = np.argsort(-1 * (scaled - target))
    target[indices[:n_diff]] += 1
    return target

def log_transform(data, th_b=10, th_t=10):
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
        values.append([])
        for th_t in range(size // 2):
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

def sampling(ip_data, samp_distr, save_path, name=""):
    maxs = np.arange(0.1, 1.1, 0.1)
    mins = maxs - 0.1

    df1 = pd.DataFrame()
    df = ip_data
    n_img = np.sum(samp_distr)
    for _0, _1, n in zip(mins, maxs, samp_distr):
        df0 = df[df["score"] >= _0]
        df0 = df0[df0["score"] <= _1] if _1 == 1 else df0[df0["score"] < _1]
        try:
            df0 = df0.sample(n=n)
        except ValueError:
            pass
            if not name=="":
                print(name)
                print(f"area {_0:.2f}-{_1:.2f}, {len(df0)} pictures found, < demand of {n}")
        df = df[~df['file_name'].isin(df0['file_name'])]
        df1 = pd.concat([df1, df0])
    # randomly fill the space
    if not len(df1) == n_img:
        df_append = df.sample(n=n_img - len(df1))
        df1 = pd.concat([df1, df_append])
    df1.to_json(save_path, orient="records", lines=True)
    return save_path

def main(args):
    ip_path = Path(args.ip_root)
    cls_paths = [p for p in ip_path.iterdir() if p.is_dir()]
    ori_path = Path(args.ori_dataset)
    sg.generate_metadata(ip_path)
    sg.generate_metadata(ori_path, [p.name for p in cls_paths])
    for cls_path in cls_paths:
        cls = cls_path.name
        ip_meta_path = cls_path.joinpath("metadata.jsonl")

        # fetch threshold values if stored
        th_path = Path(args.th_file)
        if th_path.is_file() and th_path.stat().st_size > 0:
            df = pd.read_json(args.th_file, lines=True)
            res = df.loc[(df['set'] == ip_path.name) & (df['cls'] == cls)]
            if len(res) > 0:
                th_b, th_t = res.iloc[0]["b"], res.iloc[0]["t"]
            else:
                th_b, th_t = get_threshold(ip_meta_path, size=args.ipc, ip_name=ip_path.name, cls=cls,
                                           save_file=args.th_file)
        else:
            th_b, th_t = get_threshold(ip_meta_path, size=args.ipc, ip_name=ip_path.name, cls=cls, save_file=args.th_file)

        print(f"use bottom_threshold={th_b}, top_threshold={th_t} for class {cls}")

        df = pd.read_json(ori_path.joinpath(cls, "metadata.jsonl"), lines=True)
        ori_distr = log_transform(list(df["score"]), th_b=th_b, th_t=th_t)
        samp_distr = get_sampling_distribution(ori_distr, args.ipc)

        ip_data = pd.read_json(ip_meta_path, lines=True)
        ip_data["score"] = log_transform(list(ip_data["score"]), th_b=th_b, th_t=th_t)

        select_file = ip_path.joinpath(cls, "selected.jsonl")
        for i in range(args.repeat):
            dd_dir_path = Path(f"{args.save_path}_{i}")
            dd_dir_path.mkdir(exist_ok=True)
            if args.show_lack_info:
                sampling(ip_data, samp_distr, save_path=select_file, name=f"{ip_path.name}_{cls}")
            else:
                sampling(ip_data, samp_distr, save_path=select_file)
            move_selected(select_file, dd_dir_path.joinpath(cls))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip-root", type=str, default="")
    parser.add_argument("--ori-dataset", type=str, default="ImageNet")
    parser.add_argument("--save-path", type=str, default="output")
    parser.add_argument("--ipc", type=int, default=1)
    samp_distr = list(myset.samp_distr.keys()) +["scale"]
    parser.add_argument("--sample-distribution", type=str, choices=samp_distr, default="scale")
    parser.add_argument("--th-file", type=str, default="threshold.jsonl")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--show-lack-info", action="store_true")

    args = parser.parse_args()
    main(args)
