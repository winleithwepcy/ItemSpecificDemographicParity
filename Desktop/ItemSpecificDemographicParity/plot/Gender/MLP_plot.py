import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

# Data provided
methods = {
    "FOCF(Absolute)": [
        (0.2256,0.2476,0.2296,0.2533,0.3977),
        (0.2454,0.2674,0.2410,0.2835,0.3986),
        (0.2315,0.2515,0.2287,0.2676,0.4005),
        (0.2362,0.2570,0.2401,0.2678,0.3919),
        (0.2458,0.2703,0.2386,0.2837,0.3947),
        (0.2318,0.2574,0.2293,0.2562,0.3944),
    ],
    "FOCF(Value)": [
        (0.2256,0.2476,0.2296,0.2533,0.3977),
        (0.2352,0.2583,0.2381,0.2641,0.3976),
        (0.2301,0.2506,0.2240,0.2671,0.3943),
        (0.2238,0.2438,0.2224,0.2573,0.3907),
        (0.2272,0.2467,0.2305,0.2589,0.3921),
        (0.2409,0.2642,0.2330,0.2769,0.3961),
    ],
    "ISDP(Baseline)": [
        (0.2256,0.2476,0.2296,0.2533,0.3977),
        (0.2250,0.2528,0.2252,0.2613,0.3834),
        (0.2189,0.2457,0.2089,0.2687,0.3666),
        (0.2164,0.2445,0.2117,0.2538,0.3694),
        (0.1920,0.2234,0.1995,0.2249,0.3578),
        (0.1772,0.2118,0.1863,0.2013,0.3401),
    ],
    "ISDP(Calibrated)": [
        (0.2256,0.2476,0.2296,0.2533,0.3977),
        (0.2283,0.2548,0.2204,0.2829,0.3714),
        (0.2122,0.2332,0.2091,0.2690,0.3578),
        (0.2116,0.2385,0.2072,0.2710,0.3401),
        (0.2039,0.2303,0.1955,0.2499,0.3260),
        (0.2098,0.2397,0.2014,0.2622,0.3181),
    ],
    "ISDP(Ours)": [
        (0.2444,	0.2661,	0.1217,	0.1318,	0.3408),
        (0.2836,	0.3299,	0.1466,	0.1831,	0.3203),
        (0.2072,	0.2551,	0.0999,	0.1456, 0.2904),
        (0.1362,	0.1783,	0.0659,	0.1239,	0.2636),
        (0.1809,	0.2265,	0.0867,	0.1432,	0.2748),
        (0.1053,	0.1438,	0.0616,	0.1293,	0.2342)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/Gender/MLP/MLP_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

