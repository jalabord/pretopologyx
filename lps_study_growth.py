import numpy as np
import time
from example_lps_estimation import lps_estimation


def study_growth():
    results = {}
    for i in range(10, 12):
        print("time for size: " + str(i))
        times = list()
        for j in range(10):
            time_temp = lps_estimation(i, [{3}, {5}, {6}], 50, 30)["time"]
            times.append(time_temp)
        results[i] = times.copy()
    print(results)


if __name__ == '__main__':
    study_growth()
