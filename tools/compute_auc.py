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

    # Initialize k_ar
    k_ar = {1: 0}
    # Loop through lines of log file
    lines = [line.rstrip('\n') for line in open(args.logpath)]
    for i in range(len(lines)):
        if "Average Recall" in lines[i] and "maxDets=" in lines[i]:
            print("\n", lines[i])
            ns_line = lines[i].replace(' ', '')
            k = int(ns_line.split('maxDets=')[-1].split(']')[0])
            ar = float(lines[i].split()[-1])
            print(k, ar)
            k_ar[k] = ar

            # If this condition hits, we are at the end of the current eval
            if k == 1500:
                # Compute and print AUC
                ks = [1, 10, 30, 50, 100, 300, 500, 1000]
                ars = [k_ar[k] for k in ks]
                #ks_to_print = [10, 30, 100, 300, 1000]
                ks_to_print = [10, 100, 1000]
                ars_to_print = [round(k_ar[k]*100, 1) for k in ks_to_print]
                print("\n\n")
                print(ks)
                print(ars)
                print("\n", ks_to_print)
                print(ars_to_print)
                x = np.array(ks)
                x_log = np.log(x) / np.log(1000)
                y = np.array(ars)
                auc = metrics.auc(x_log, y)
                print('AUC score:', auc)
                print("\n\n\n\n\n")

                # Reset k_ar for potentially the next eval
                k_ar = {1: 0}



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
