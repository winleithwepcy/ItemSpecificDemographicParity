import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Updated data
methods = {
    "FOCF(Absolute)": [
        (0.2544,	0.266,	0.2339,	0.2468,	0.40370),
        (0.2519,	0.2645,	0.2318,	0.2446,	0.4034),
        (0.2502,	0.2633,	0.2308,	0.2431,	0.4024),
        (0.2479,	0.2611,	0.2293,	0.2410,	0.4040),
        (0.2464,	0.2601,	0.2289,	0.2396,	0.4046),
        (0.2462,	0.2599,	0.2297,	0.2386,	0.4037)
    ],
    "FOCF(Value)": [
        (0.2544,	0.2664,	0.2339,	0.2468,	0.4037),
        (0.2539,	0.2661,	0.2330,	0.2462,	0.4036),
        (0.2527,	0.2651,	0.2322,	0.2448,	0.4034),
        (0.2517,	0.2645,	0.2327,	0.2432,	0.4036),
        (0.2511,	0.2640,	0.2331,	0.2421,	0.4037),
        (0.2497,	0.2632,	0.2327,	0.2405,	0.4029)
    ],
    "ISDP(Baseline)": [
        (0.2544,	0.2664,	0.2339,	0.2468,	0.4037),
        (0.2501,	0.2710,	0.2285,	0.2366,	0.3898),
        (0.2331,	0.2623,	0.2106,	0.2211,	0.3813),
        (0.2159,	0.2524,	0.1918,	0.2071,	0.3711),
        (0.2027,	0.2452,	0.1878,	0.1855,	0.3569),
        (0.1865,	0.2309,	0.1688,	0.1761,	0.3494)
    ],
    "ISDP(Calibrated)": [
        (0.2544,	0.2664,	0.2339,	0.2468,	0.4037),
        (0.2592,	0.2560,	0.2203,	0.2583,	0.3566),
        (0.2396,	0.2349,	0.2044,	0.2354,	0.3329),
        (0.2464,	0.2422,	0.2040,	0.2399,	0.3146),
        (0.2376,	0.2306,	0.1908,	0.2383,	0.2930),
        (0.2396,	0.2294,	0.1878,	0.2398,	0.2758)
    ],
    "ISDP(Ours)": [
        (0.2320,	0.2450,	0.2267,	0.2194,	0.4362),
        (0.2282,	0.2499,	0.1977,	0.2357,	0.4257),
        (0.2122,	0.2380,	0.1883,	0.2233,	0.4124),
        (0.1853,	0.2180,	0.1689,	0.1904,	0.3998),
        (0.1707,	0.2066,	0.1617,	0.1750,	0.3876),
        (0.1720,	0.2126,	0.1596,	0.1802,	0.3702)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/AG/BiasedMF/BiasedMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

