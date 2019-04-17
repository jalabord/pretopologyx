from example_lps_estimation import lps_estimation


def F_measure():
    sizes = [15, 25, 35]
    sizes = [10]
    DNFs = [
        [{3}, {5}, {6}],
        [{3, 5}, {4, 7}, {6}],
        [{2}, {4}, {1, 3}, {3, 6}, {5, 6, 7}]
    ]
    percentages_blocked = [0, 10, 20, 30, 40, 50, 60]
    percentages_blocked = [20]
    results = {}
    for size in sizes:
        results[str(size)] = {}
        for dnf in DNFs:
            results[str(size)][str(dnf)] = {}
            for blocked in percentages_blocked:
                results[str(size)][str(dnf)][str(blocked)] = {"precision": list(), "recall": list()}
                for i in range(10):
                    estimated = lps_estimation(size=size, original_dnf=dnf, percent_blocked=blocked, percent_closures=30)["dnf"]
                    intersection = sum([(s in estimated) for s in dnf])
                    precision = intersection/len(estimated)
                    recall = intersection/len(dnf)
                    results[str(size)][str(dnf)][str(blocked)]["precision"].append(precision)
                    results[str(size)][str(dnf)][str(blocked)]["recall"].append(recall)
    print(results)


if __name__ == '__main__':
    F_measure()
