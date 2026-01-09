import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Updated data
methods = {
    "FOCF(Absolute)": [
        (0.2563, 0.2722, 0.2450, 0.2953, 0.4318),
        (0.2560, 0.2720, 0.2443, 0.2953, 0.4313),
        (0.2560, 0.2721, 0.2447, 0.2954, 0.4305),
        (0.2560, 0.2718, 0.2446, 0.2952, 0.4301),
        (0.2558, 0.2716, 0.2443, 0.2951, 0.4305),
        (0.2554, 0.2710, 0.2435, 0.2949, 0.4306),
    ],
    "FOCF(Value)": [
        (0.2563, 0.2722, 0.2450, 0.2953, 0.4318),
        (0.2560, 0.2720, 0.2444, 0.2952, 0.4311),
        (0.2561, 0.2720, 0.2447, 0.2952, 0.4306),
        (0.2560, 0.2716, 0.2447, 0.2948, 0.4311),
        (0.2559, 0.2715, 0.2447, 0.2945, 0.4307),
        (0.2559, 0.2712, 0.2445, 0.2944, 0.4310),
    ],
    "ISDP(Baseline)": [
        (0.2563, 0.2722, 0.2450, 0.2953, 0.4318),
        (0.2547, 0.2740, 0.2436, 0.2930, 0.4148),
        (0.2335, 0.2572, 0.2260, 0.2668, 0.4075),
        (0.2302, 0.2592, 0.2320, 0.2579, 0.4009),
        (0.2098, 0.2396, 0.2133, 0.2309, 0.3990),
        (0.2018, 0.2356, 0.2076, 0.2270, 0.3836),
    ],
    "ISDP(Calibrated)": [
        (0.2563, 0.2722, 0.2450, 0.2953, 0.4318),
        (0.2668, 0.2748, 0.2553, 0.3204, 0.4047),
        (0.2601, 0.2709, 0.2575, 0.3111, 0.3903),
        (0.2522, 0.2667, 0.2375, 0.3191, 0.3811),
        (0.2520, 0.2677, 0.2422, 0.3058, 0.3725),
        (0.2560, 0.2696, 0.2557, 0.3071, 0.3611),
    ],
    "ISDP(Ours)": [
        (0.5472,	0.6109,	0.3061,	0.3189,	0.4065),
        (0.5439,	0.6177,	0.2773,	0.2904,	0.3945),
        (0.5197,	0.5994,	0.2552,	0.2550,	0.3842),
        (0.4962,	0.5793,	0.2382,	0.2490,	0.3773),
        (0.4584,	0.5503,	0.2062,	0.2219,	0.3613),
        (0.4524,	0.5395,	0.1902,	0.2166,	0.3637),
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Gender/PMF/PMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

