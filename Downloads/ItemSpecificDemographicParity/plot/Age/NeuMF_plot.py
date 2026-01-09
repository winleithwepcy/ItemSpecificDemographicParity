import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

methods = {
    "FOCF(Absolute)": [
        (0.2521,	0.2470,	0.2461,	0.2590,	0.4158),
        (0.2487,	0.2445,	0.2481,	0.2572,	0.4158),
        (0.2530,	0.2490,	0.2526,	0.2551,	0.4164),
        (0.2549,	0.2529,	0.2480,	0.2573,	0.4157),
        (0.2466,	0.2474,	0.2488,	0.2595,	0.4141),
        (0.2500,	0.2487,	0.2444,	0.2588,	0.4157)

    ],
    "FOCF(Value)": [
        (0.2521,	0.2470,	0.2461,	0.2590,	0.4158),
        (0.2505,	0.2466,	0.2478,	0.2575,	0.4163),
        (0.2488,	0.2457,	0.2490,	0.2585,	0.4165),
        (0.2536,	0.2509,	0.2514,	0.2593,	0.4163),
        (0.2584,	0.2546,	0.2478,	0.2603,	0.4159),
        (0.2552,	0.2519,	0.2465,	0.2588,	0.4156)
    ],
    "ISDP(Baseline)": [
        (0.2521,	0.2470,	0.2461,	0.2590,	0.4158),
        (0.3254,	0.3408,	0.2366,	0.2568,	0.4017),
        (0.2955,	0.3214,	0.2186,	0.2313,	0.3961),
        (0.2795,	0.3177,	0.2074,	0.2158,	0.3796),
        (0.2595,	0.3053,	0.1901,	0.1870,	0.3714),
        (0.2452,	0.2988,	0.1732,	0.1818,	0.3532)
    ],
    "ISDP(Calibrated)": [
        (0.2521,	0.2470,	0.2461,	0.2590,	0.4158),
        (0.3695,	0.2721,	0.2289,	0.2871,	0.3798),
        (0.3240,	0.2645,	0.1737,	0.2133,	0.3417),
        (0.2750,	0.2618,	0.1338,	0.1733,	0.3087),
        (0.2813,	0.2761,	0.1139,	0.1554,	0.2993),
        (0.2634,	0.2710,	0.1027,	0.1497,	0.2836)
    ],
    "ISDP(Ours)": [
        (0.3704,	0.3544,	0.1935,	0.1858,	0.4163,	0.6950),
        (0.3521,	0.3762,	0.1676,	0.1885,	0.4104,	0.6886),
        (0.3509,	0.3954,	0.1707,	0.1976,	0.4001,	0.6755),
        (0.3214,	0.3736,	0.1640,	0.1979,	0.3825,	0.6525),
        (0.2707,	0.3323,	0.1495,	0.1771,	0.3684,	0.6338),
        (0.2642,	0.3305,	0.1569,	0.1920,	0.3512,	0.6175)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Age/NeuMF/NeuMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

