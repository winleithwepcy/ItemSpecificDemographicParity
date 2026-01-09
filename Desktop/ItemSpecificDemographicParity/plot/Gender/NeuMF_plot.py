import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

methods = {
    "FOCF(Absolute)": [
        (0.2537,0.2783,0.2440,0.2946,0.4174),
        (0.2519,0.2678,0.2382,0.2898,0.4192),
        (0.2577,0.2759,0.2440,0.2968,0.4213),
        (0.2612,0.2790,0.2499,0.3001,0.4224),
        (0.2570,0.2775,0.2418,0.2987,0.4190),
        (0.2620,0.2791,0.2437,0.3061,0.4201),
    ],
    "FOCF(Value)": [
        (0.2537,0.2783,0.2440,0.2946,0.4174),
        (0.2596,0.2800,0.2520,0.2930,0.4201),
        (0.2632,0.2858,0.2412,0.3109,0.4186),
        (0.2634,0.2814,0.2545,0.2961,0.4170),
        (0.2559,0.2799,0.2477,0.2919,0.4235),
        (0.2569,0.2768,0.2422,0.3005,0.4192),
    ],
    "ISDP(Baseline)": [
        (0.2537,0.2783,0.2440,0.2946,0.4174),
        (0.2469,0.2697,0.2444,0.2796,0.3995),
        (0.2288,0.2525,0.2268,0.2569,0.4013),
        (0.2266,0.2571,0.2284,0.2511,0.3941),
        (0.2025,0.2365,0.2080,0.2207,0.3907),
        (0.1880,0.2283,0.2014,0.1926,0.3777),
    ],
    "ISDP(Calibrated)": [
        (0.2537,0.2783,0.2440,0.2946,0.4174),
        (0.2551,0.2625,0.2336,0.3214,0.3973),
        (0.2276,0.2392,0.2118,0.2908,0.3933),
        (0.1913,0.2053,0.1774,0.2424,0.3625),
        (0.1782,0.1968,0.1660,0.2180,0.3443),
        (0.1555,0.1823,0.1451,0.1949,0.3345),
    ],
    "ISDP(Ours)": [
        (0.3791,0.3652,0.1954,	0.1984,	0.4128),
        (0.3616,0.3845,0.1747,	0.2018,	0.4117),	
        (0.3523,0.3836,0.1743, 0.2091,	0.3994),	
        (0.3214,0.3705,0.1686,	0.1946,	0.3848),	
        (0.3018,0.3579,0.1671,	0.1981,	0.3682),	
        (0.2931,0.3586,0.1713,	0.1982,	0.3560)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Gender/NeuMF/NeuMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

