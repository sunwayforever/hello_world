#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2023-06-06 11:32
import pickle
import sys
import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

variant_mapping = {
    "0": "default",
    "1": "schedule",
    "2": "profile",
}


def benchmark(test_case):
    result = defaultdict(list)
    try:
        result = pickle.load(open(f"{test_case}.pkl", "rb"))
    except (OSError) as e:
        for test_variant in ["0", "1", "2"]:
            for arch in ["X86", "Riscv", "Arm"]:
                elf = f"{test_case}.{test_variant}.{arch}.elf"
                print(elf)
                for cpu in ["MinorCPU", "O3CPU"]:
                    cmd = (
                        f"/home/sunway/source/gem5/build/{arch.upper()}/gem5.opt "
                        f"/home/sunway/source/gem5/configs/example/se.py --cpu-type {arch}{cpu} --caches -c "
                        f"{elf}"
                    )
                    # result[test_variant].append(10)
                    s = os.popen(cmd).read()
                    print(s)
                    for x in s.split("\n"):
                        m = re.match(r"Exiting @ tick ([0-9]+) .*", x)
                        if m:
                            result[test_variant].append(int(m[1]))
        pickle.dump(result, open(f"{test_case}.pkl", "wb"))

    print(result)

    cpus = []
    for arch in ["X86", "Riscv", "Arm"]:
        for cpu in ["Minor", "O3"]:
            cpus.append(f"{arch}{cpu}")

    ind = np.arange(len(cpus))
    width = 0.25

    bars = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    n = 0
    for _, cycles in result.items():
        bar = plt.bar(ind + width * n, cycles, width, color=colors[n])
        bars.append(bar)
        n += 1
    plt.xlabel("cpu")
    plt.ylabel("cycle")
    plt.title(test_case)
    plt.xticks(ind + width / 2, cpus)
    plt.legend(bars, [variant_mapping[i] for i in result.keys()])
    plt.savefig(f"{test_case}.png")


if __name__ == "__main__":
    benchmark(sys.argv[1])
