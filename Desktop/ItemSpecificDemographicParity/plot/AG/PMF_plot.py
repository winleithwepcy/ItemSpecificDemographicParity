import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Data organized per method and metric/value pairs (Absolute, Value, ISDP-Rated, ISDP-General, NDCG)
methods = {
    "FOCF(Absolute)": [
        (0.2486,	0.2639,	0.2292,	0.2413,	0.4083),
        (0.2477,	0.2629,	0.2282,	0.2413,	0.4105),
        (0.2473,	0.2625,	0.2282,	0.2415,	0.4098),
        (0.2463,	0.2610,	0.2281,	0.2402,	0.4097),
        (0.2463,	0.2607,	0.2292,	0.2407,	0.4090),
        (0.2453,	0.2601,	0.2290,	0.2392,	0.4085)

    ],
    "FOCF(Value)": [
        (0.2486,	0.2639,	0.2292,	0.2413,	0.4083),
        (0.2491,	0.2649,	0.2292,	0.2426,	0.4093),
        (0.2493,	0.2658,	0.2303,	0.2431,	0.4087),
        (0.2493,	0.2659,	0.2314,	0.2432,	0.4091),
        (0.2487,	0.2652,	0.2314,	0.2428,	0.4079),
        (0.2480,	0.2644,	0.2314,	0.2422,	0.4070)
    ],
    "ISDP(Baseline)": [
        (0.2486,	0.2639,	0.2292,	0.2413,	0.4083),
        (0.2440,	0.2690,	0.2228,	0.2323,	0.3929),
        (0.2268,	0.2540,	0.2064,	0.2172,	0.3817),
        (0.2147,	0.2462,	0.1947,	0.2047,	0.3663),
        (0.1937,	0.2268,	0.1779,	0.1801,	0.3541),
        (0.1798,	0.2168,	0.1538,	0.1752,	0.3413)
    ],
    "ISDP(Calibrated)": [
        (0.2486,	0.2639,	0.2292,	0.2413,	0.4083),
        (0.2617,	0.2601,	0.2137,	0.2628,	0.3698),
        (0.2543,	0.2547,	0.2073,	0.2546,	0.3472),
        (0.2422,	0.2449,	0.1927,	0.2432,	0.3344),
        (0.2402,	0.2440,	0.1837,	0.2450,	0.3155),
        (0.2419,	0.2465,	0.1970,	0.2402,	0.2961)
    ],
    "ISDP(Ours)": [
        (0.3266,	0.3667,	0.3061,	0.3189,	0.4065),
        (0.2974,	0.3449,	0.2773,	0.2904,	0.3945),
        (0.2681,	0.3294,	0.2552,	0.2550,	0.3842),
        (0.2548,	0.3172,	0.2382,	0.2490,	0.3773),
        (0.2225,	0.2780,	0.2062,	0.2219,	0.3613),
        (0.2092,	0.2575,	0.1902,	0.2166,	0.3637)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/AG/PMF/PMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

