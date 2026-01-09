import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Data organized per method and metric/value pairs (Absolute, Value, ISDP-Rated, ISDP-General, NDCG)
methods = {
    "FOCF(Absolute)": [
        (0.2667,	0.2645,	0.2339,	0.2468,	0.4037),
        (0.2670,	0.2647,	0.2342,	0.2468,	0.4034),
        (0.2675,	0.2649,	0.2343,	0.2468,	0.4037),
        (0.2674,	0.2644,	0.2347,	0.2464,	0.4035),
        (0.2667,	0.2636,	0.2349,	0.2458,	0.4039),
        (0.2663,	0.2629,	0.2353,	0.2454,	0.4043)

    ],
    "FOCF(Value)": [
        (0.2667,	0.2645,	0.2339,	0.2468,	0.4037),
        (0.2670,	0.2648,	0.2341,	0.2467,	0.4032),
        (0.2675,	0.2651,	0.2342,	0.2466,	0.4036),
        (0.2673,	0.2646,	0.2344,	0.2460,	0.4035),
        (0.2664,	0.2635,	0.2345,	0.2452,	0.4037),
        (0.2658,	0.2626,	0.2347,	0.2444,	0.4041)
    ],
    "ISDP(Baseline)": [
        (0.2667,	0.2645,	0.2339,	0.2468,	0.4037),
        (0.3233,	0.3413,	0.2285,	0.2366,	0.3898),
        (0.3135,	0.3439,	0.2106,	0.2211,	0.3813),
        (0.2982,	0.3435,	0.1918,	0.2071,	0.3711),
        (0.2723,	0.3227,	0.1878,	0.1855,	0.3569),
        (0.2467,	0.3042,	0.1688,	0.1761,	0.3494)
    ],
    "ISDP(Calibrated)": [
        (0.2667,	0.2645,	0.2339,	0.2468,	0.4037),
        (0.3848,	0.3012,	0.2203,	0.2583,	0.3566),
        (0.3671,	0.2864,	0.2044,	0.2354,	0.3329),
        (0.3980,	0.3136,	0.2040,	0.2399,	0.3146),
        (0.3977,	0.3079,	0.1908,	0.2383,	0.2930),
        (0.3946,	0.2997,	0.1878,	0.2398,	0.2758)
    ],
    "ISDP(Ours)": [
        (0.4340,	0.4492,	0.2267,	0.2194,	0.4362),
        (0.4166,	0.4427,	0.1977,	0.2357,	0.4257),
        (0.3864,	0.4268,	0.1883,	0.2233,	0.4124),
        (0.3353,	0.3976,	0.1689,	0.1904,	0.3998),
        (0.3121,	0.3807,	0.1617,	0.1750,	0.3876),
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Age/BiasedMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

