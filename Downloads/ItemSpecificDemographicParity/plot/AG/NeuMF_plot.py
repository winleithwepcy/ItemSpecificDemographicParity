import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import letter
import os

methods = {
    "FOCF(Absolute)": [
        (0.2637,	0.2742,	0.2461,	0.2590,	0.4158),
        (0.2626,	0.2746,	0.2484,	0.2569,	0.4164),
        (0.2621,	0.2743,	0.2466,	0.2596,	0.4149),
        (0.2603,	0.2731,	0.2481,	0.2551,	0.4159),
        (0.2594,	0.2736,	0.2485,	0.2536,	0.4162),
        (0.2595,	0.2726,	0.2460,	0.2567,	0.4155) 
    ],
    "FOCF(Value)": [
        (0.2637,	0.2742,	0.2461,	0.2590,	0.4158),
        (0.2675,	0.2801,	0.2517,	0.2614,	0.4151),
        (0.2640,	0.2753,	0.2478,	0.2618,	0.4180),
        (0.2639,	0.2759,	0.2487,	0.2621,	0.4135),
        (0.2682,	0.2774,	0.2511,	0.2653,	0.4169),
        (0.2673,	0.2759,	0.2545,	0.2601,	0.4162)
    ],
    "ISDP(Baseline)": [
        (0.2637,	0.2742,	0.2461,	0.2590,	0.4158),
        (0.2643,	0.2846,	0.2366,	0.2568,	0.4017),
        (0.2402,	0.2688,	0.2186,	0.2313,	0.3961),
        (0.2267,	0.2617,	0.2074,	0.2158,	0.3796),
        (0.2040,	0.2407,	0.1901,	0.1870,	0.3714),
        (0.1914,	0.2326,	0.1732,	0.1818,	0.3532)
    ],
    "ISDP(Calibrated)": [
        (0.2637,	0.2742,	0.2461,	0.2590,	0.4158),
        (0.2781,	0.2640,	0.2289,	0.2871,	0.3798),
        (0.2134,	0.2209,	0.1737,	0.2133,	0.3417),
        (0.1730,	0.1970,	0.1338,	0.1733,	0.3087),
        (0.1533,	0.1787,	0.1139,	0.1554,	0.2993),
        (0.1442,	0.1735,	0.1027,	0.1497,	0.2836)
    ],
    "ISDP(Ours)": [
        (0.1985,	0.1924,	0.1935,	0.1858,	0.4163),
        (0.1869,	0.2053,	0.1676,	0.1885,	0.4104),
        (0.1961,	0.2201,	0.1707,	0.1976,	0.4001),
        (0.1870,	0.2172,	0.1640,	0.1979,	0.3825),
        (0.1648,	0.2032,	0.1495,	0.1771,	0.3684),
        (0.1756,	0.2223,	0.1569,	0.1920,	0.3512)
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
    filename = f"/home/win/ISDP_ML_Ours/result/Modified/AG/NeuMF/NeuMF_{metric_name}.png"
    plt.savefig(filename)
    plt.close()
    png_files.append(filename)

