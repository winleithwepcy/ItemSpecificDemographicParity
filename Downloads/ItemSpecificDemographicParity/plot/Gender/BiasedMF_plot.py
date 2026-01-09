import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Data organized per method and metric/value pairs (Absolute, Value, ISDP-Rated, ISDP-General, NDCG)
methods = {
    "FOCF(Absolute)": [
        (0.2653, 0.2775, 0.2550, 0.2987, 0.4254),
        (0.2661, 0.2786, 0.2562, 0.2990, 0.4260),
        (0.2665, 0.2790, 0.2564, 0.2994, 0.4256),
        (0.2670, 0.2794, 0.2566, 0.3000, 0.4252),
        (0.2675, 0.2795, 0.2577, 0.3000, 0.4249),
        (0.2672, 0.2788, 0.2578, 0.2996, 0.4251)
    ],
    "FOCF(Value)": [
        (0.2653, 0.2775, 0.2550, 0.2987, 0.4254),
        (0.2659, 0.2783, 0.2558, 0.2992, 0.4262),
        (0.2662, 0.2787, 0.2558, 0.2997, 0.4251),
        (0.2665, 0.2788, 0.2561, 0.2997, 0.4253),
        (0.2662, 0.2781, 0.2563, 0.2991, 0.4258),
        (0.2656, 0.2773, 0.2567, 0.2983, 0.4261)
    ],
    "ISDP(Baseline)": [
        (0.2653, 0.2775, 0.2550, 0.2987, 0.4254),
        (0.2504, 0.2713, 0.2494, 0.2800, 0.4142),
        (0.2335, 0.2591, 0.2461, 0.2532, 0.4076),
        (0.2132, 0.2429, 0.2246, 0.2315, 0.4048),
        (0.2062, 0.2445, 0.2144, 0.2223, 0.3887),
        (0.1893, 0.2311, 0.2016, 0.1956, 0.3758)
    ],
    "ISDP(Calibrated)": [
        (0.2653, 0.2775, 0.2550, 0.2987, 0.4254),
        (0.2561, 0.2637, 0.2400, 0.3123, 0.4024),
        (0.2579, 0.2674, 0.2552, 0.3141, 0.3845),
        (0.2565, 0.2709, 0.2439, 0.3137, 0.3700),
        (0.2658, 0.2748, 0.2649, 0.3152, 0.3680),
        (0.2527, 0.2616, 0.2475, 0.3093, 0.3606)
    ],
    "ISDP(Ours)": [
        (0.4340,	0.4492,	0.2267,	0.2194,	0.4362),
        (0.4166,	0.4427,	0.1977,	0.2357,	0.4257),
        (0.3864, 	0.4268,	0.1883,	0.2233,	0.4124),
        (0.3353, 	0.3976,	0.1689,	0.1904,	0.3998),
        (0.3121, 	0.3807,	0.1617,	0.1750,	0.3876),
        (0.2922,	0.3631,	0.1596,	0.1802,	0.3702)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Gender/BiasedMF/BiasedMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

