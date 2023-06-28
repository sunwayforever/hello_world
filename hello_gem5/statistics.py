#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2023-06-08 11:18
import numpy as np
import pickle
import os


def stat(name, sched_imp, profile_imp, o3_imp):
    print(f"--- {name} ---")
    print(
        f"sched  : min: {np.min(sched_imp):8.3}, max: {np.max(sched_imp):8.3}, "
        f"mean: {np.mean(sched_imp):8.3}, median: {np.median(sched_imp):8.3}"
    )
    print(
        f"profile: min: {np.min(profile_imp):8.3}, max: {np.max(profile_imp):8.3}, "
        f"mean: {np.mean(profile_imp):8.3}, median: {np.median(profile_imp):8.3}"
    )
    print(
        f"o3     : min: {np.min(o3_imp):8.3}, max: {np.max(o3_imp):8.3}, "
        f"mean: {np.mean(o3_imp):8.3}, median: {np.median(o3_imp):8.3}"
    )
    print()


if __name__ == "__main__":
    sched_imp = np.array([])
    profile_imp = np.array([])
    o3_imp = np.array([])
    for root, sub_folders, files in os.walk("./"):
        for f in files:
            if not f.endswith(".pkl"):
                continue
            result = pickle.load(open(f, "rb"))
            default, sched, profile = (
                np.array(result["0"]),
                np.array(result["1"]),
                np.array(result["2"]),
            )
            o3_imp = np.concatenate((o3_imp, default[::2] / default[1::2]))
            o3_imp = np.concatenate((o3_imp, sched[::2] / sched[1::2]))
            o3_imp = np.concatenate((o3_imp, profile[::2] / profile[1::2]))
            sched_imp = np.concatenate((sched_imp, default / sched))
            profile_imp = np.concatenate((profile_imp, sched / profile))

    stat("overall", sched_imp, profile_imp, o3_imp)
    sched_imp = np.reshape(sched_imp, (-1, 6))
    profile_imp = np.reshape(profile_imp, (-1, 6))
    o3_imp = np.reshape(o3_imp, (-1, 6))
    stat("X86", sched_imp[:, 0:2], profile_imp[:, 0:2], o3_imp[:, 0:2])
    stat("RISC-V", sched_imp[:, 2:4], profile_imp[:, 2:4], o3_imp[:, 2:4])
    stat("ARM", sched_imp[:, 4:6], profile_imp[:, 4:6], o3_imp[:, 4:6])
