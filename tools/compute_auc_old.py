import os
import argparse
import numpy as np
from sklearn import metrics


def get_avg_recalls(logpath):
    k_ar = {}
    lines = [line.rstrip('\n') for line in open(logpath)]
    for i in range(len(lines)):
        if "Average Recall" in lines[i] and "maxDets=" in lines[i]:
            print("\n", lines[i])
            ns_line = lines[i].replace(' ', '')
            k = int(ns_line.split('maxDets=')[-1].split(']')[0])
            ar = float(lines[i].split()[-1])
            print(k, ar)
            k_ar[k] = ar
            
    return k_ar

            
            






def main():
    parser = argparse.ArgumentParser(description="Compute AUC")
    parser.add_argument(
        "logpath",
        type=str,
    )
    args = parser.parse_args()

    k_ar = get_avg_recalls(args.logpath)
    k_ar[1] = 0  # Need this to complete curve, set K to 1 because we take log later (log(1)=0)

    ks = [1, 10, 30, 50, 100, 300, 500, 1000]
    ars = [k_ar[k] for k in ks]

    print("\n\n")
    print(ks)
    print(ars)
    x = np.array(ks)
    x_log = np.log(x) / np.log(1000)
    y = np.array(ars)
    auc = metrics.auc(x_log, y)
    print('AUC score:', auc)


    # K (number of shots)
    #x = np.array([1., 10., 30., 50., 100., 300., 500., 1000.])
    #x_log = np.log(x) / np.log(1000)
    # Average Recall scores
    #y = np.array([0.0, 18.0, 26.5, 29.6, 33.4, 39.0, 41.5, 45.0])
    #y *= 0.01
    #auc = metrics.auc(x_log, y)
    #print('AUC score:', auc)


if __name__ == "__main__":
    main()
