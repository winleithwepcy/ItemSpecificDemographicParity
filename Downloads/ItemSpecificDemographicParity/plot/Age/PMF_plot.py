import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Updated data
methods = {
    "FOCF(Absolute)": [
        (0.2548, 0.2548, 0.2292, 0.2413, 0.4083),
        (0.2555, 0.2553, 0.2291, 0.2413, 0.4086),
        (0.2556, 0.2555, 0.2292, 0.2412, 0.4089),
        (0.2559, 0.2557, 0.2292, 0.2413, 0.4090),
        (0.2556, 0.2554, 0.2293, 0.2415, 0.4086),
        (0.2556, 0.2554, 0.2296, 0.2417, 0.4086)
    ],
    "FOCF(Value)": [
        (0.2548, 0.2548, 0.2292, 0.2413, 0.4083),
        (0.2555, 0.2553, 0.2289, 0.2413, 0.4086),
        (0.2556, 0.2554, 0.2289, 0.2411, 0.4095),
        (0.2558, 0.2556, 0.2288, 0.2412, 0.4089),
        (0.2555, 0.2553, 0.2288, 0.2414, 0.4095),
        (0.2554, 0.2552, 0.2289, 0.2415, 0.4097)
    ],
    "ISDP(Baseline)": [
        (0.2548, 0.2548, 0.2292, 0.2413, 0.4083),
        (0.3082, 0.3271, 0.2228, 0.2323, 0.3929),
        (0.2982, 0.3248, 0.2064, 0.2172, 0.3817),
        (0.2790, 0.3133, 0.1947, 0.2047, 0.3663),
        (0.2550, 0.2935, 0.1779, 0.1801, 0.3541),
        (0.2333, 0.2704, 0.1538, 0.1752, 0.3413)
    ],
    "ISDP(Calibrated)": [
        (0.2548, 0.2548, 0.2292, 0.2413, 0.4083),
        (0.3920, 0.3080, 0.2137, 0.2628, 0.3698),
        (0.3739, 0.2965, 0.2073, 0.2546, 0.3472),
        (0.3819, 0.3097, 0.1927, 0.2432, 0.3344),
        (0.3836, 0.3197, 0.1837, 0.2450, 0.3155),
        (0.3898, 0.3256, 0.1970, 0.2402, 0.2961)

    ],
    "ISDP(Ours)": [
        (0.5472,	0.6109,	0.3061,	0.3189,	0.4065),
        (0.5439,	0.6177,	0.2773,	0.2904,	0.3945),
        (0.5197,	0.5994,	0.2552,	0.2550,	0.3842),
        (0.4962,	0.5793,	0.2382,	0.2490,	0.3773),
        (0.4584,	0.5503,	0.2062,	0.2219,	0.3613),
        (0.4524,	0.5395,	0.1902,	0.2166,	0.3637)

    ]
}

# Metric indices
ABS_IDX = 1
VAL_IDX = 0
ISDP_R_IDX = 2
ISDP_G_IDX = 3
NDCG_IDX = 4

metrics = {
    "FOCF(Absolute)": ABS_IDX,
    "FOCF(Value)": VAL_IDX,
    "ISDP(Rated)": ISDP_R_IDX,
    "ISDP(General)": ISDP_G_IDX
}

png_files = []

# Generate PNG plots
for metric_name, idx in metrics.items():
    plt.figure()
    for method, rows in methods.items():
        xs = [r[idx] for r in rows]
        ys = [r[NDCG_IDX] for r in rows]
        plt.scatter(xs, ys, label=method)
    plt.xlabel(metric_name)
    plt.ylabel("NDCG@10")
    plt.legend()
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Age/PMF/PMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

