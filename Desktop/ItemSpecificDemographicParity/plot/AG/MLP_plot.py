import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Data provided
methods = {
    "FOCF(Absolute)": [
        (0.2358,	0.2541,	0.2194,	0.2322,	0.3681),
        (0.2369,	0.2560,	0.2113,	0.2349,	0.3689),
        (0.2300,	0.2463,	0.2120,	0.2283,	0.3641),
        (0.2320,	0.2479,	0.2137,	0.2293,	0.3646),
        (0.2261,	0.2487,	0.2069,	0.2230,	0.3596),
        (0.2359,	0.2566,	0.2118,	0.2386,	0.3633)
    ],
    "FOCF(Value)": [     
        (0.2358,	0.2541,	0.2194,	0.2322,	0.3681),
        (0.2426,	0.2565,	0.2235,	0.2341,	0.3707),
        (0.2411,	0.2572,	0.2180,	0.2427,	0.3657),
        (0.2384,	0.2566,	0.2166,	0.2370,	0.3659),
        (0.2358,	0.2532,	0.2181,	0.2305,	0.3707),
        (0.2342,	0.2516,	0.2106,	0.2309,	0.3679)
    ],
    "ISDP(Baseline)": [
        (0.2358,	0.2541,	0.2194,	0.2322,	0.3681),
        (0.2374,	0.2615,	0.2041,	0.2344,	0.3488),
        (0.2234,	0.2501,	0.1890,	0.2233,	0.3371),
        (0.1961,	0.2336,	0.1673,	0.1931,	0.3205),
        (0.1813,	0.2228,	0.1553,	0.1790,	0.3036),
        (0.1639,	0.2017,	0.1400,	0.1618,	0.2963)
    ],
    "ISDP(Calibrated)": [
        (0.2358,	0.2541,	0.2194,	0.2322,	0.3681),
        (0.2343,	0.2360,	0.1922,	0.2393,	0.3320),
        (0.2295,	0.2404,	0.1897,	0.2332,	0.3222),
        (0.2168,	0.2351,	0.1762,	0.2192,	0.2995),
        (0.2075,	0.2237,	0.1673,	0.2108,	0.2857),
        (0.2030,	0.2209,	0.1664,	0.2068,	0.2785)
    ],
    "ISDP(Ours)": [
        (0.1307,	0.1497,	0.1142,	0.1317,	0.3244),
        (0.1303,	0.1590,	0.1100,	0.1372,	0.3157),
        (0.1150,	0.1484,	0.0898,	0.1249,	0.2747),
        (0.1127,	0.1448,	0.0862,	0.1342,	0.2616),
        (0.0798,	0.1068,	0.0431,	0.1024,	0.2395),
        (0.0238,	0.0306,	0.0278,	0.0171,	0.2007)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/AG/MLP/MLP_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

