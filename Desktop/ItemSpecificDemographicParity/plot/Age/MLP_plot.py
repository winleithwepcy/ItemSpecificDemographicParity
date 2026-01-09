import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Data provided
methods = {
    "FOCF(Absolute)": [
        (0.2344,0.2384,0.2194,0.2322,0.3681),
        (0.2366,0.2387,0.2142,0.2357,0.3636),
        (0.2141,0.2212,0.2150,0.2252,0.3684),
        (0.2205,0.2231,0.2049,0.2265,0.3697),
        (0.2316,0.2362,0.2095,0.2390,0.3683),
        (0.2255,0.2298,0.2101,0.2281,0.3652)
    ],
    "FOCF(Value)": [     
        (0.2344,0.2384,0.2194,0.2322,0.3681),
        (0.2137,0.2134,0.2177,0.2324,0.3685),
        (0.2178,0.2219,0.2180,0.2353,0.3650),
        (0.2310,0.2394,0.2171,0.2330,0.3624),
        (0.2084,0.2127,0.2027,0.2206,0.3673),
        (0.2234,0.2313,0.2118,0.2299,0.3652)
    ],
    "ISDP(Baseline)": [
        (0.2344,0.2384,0.2194,0.2322,0.3681),
        (0.3300,0.3550,0.2041,0.2344,0.3488),
        (0.3196,0.3498,0.1890,0.2233,0.3371),
        (0.2844,0.3231,0.1673,0.1931,0.3205),
        (0.2629,0.3104,0.1553,0.1790,0.3036),
        (0.2510,0.3013,0.1400,0.1618,0.2963)
    ],
    "ISDP(Calibrated)": [
        (0.2344,0.2384,0.2194,0.2322,0.3681),
        (0.3309,0.2578,0.1922,0.2393,0.3320),
        (0.3310,0.2751,0.1897,0.2332,0.3222),
        (0.3457,0.3002,0.1762,0.2192,0.2995),
        (0.3200,0.2766,0.1673,0.2108,0.2857),
        (0.3178,0.2762,0.1664,0.2068,0.2785)
    ],
    "ISDP(Ours)": [

        (0.2400,0.2707,	0.1142, 0.1317,	0.3244),
        (0.2116,0.2521,	0.1100, 0.1372, 0.3157),
        (0.1606,0.2095,	0.0898,	0.1249,	0.2747),
        (0.1527,0.1990,	0.0862,	0.1342,	0.2616),
        (0.0942,0.1297,	0.0431,	0.1024,	0.2395),
        (0.0427,0.0536,	0.0278,	0.0171,	0.2007)

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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Age/MLP/MLP_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

