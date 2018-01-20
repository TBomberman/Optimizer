import numpy as np
# import utils

def get_stats(X):
    # Minimum, maximum, mean, median, and mode of all values of X
    min = np.min(X)
    max = np.max(X)
    mean = np.mean(X)
    med = np.median(X)
    # mode = utils.mode(X)
    print("Minimum: {}\nMaximum: {}\nMean: {}\nMedian {}\nMode: {}\n".format(min, max, mean, med, ""))

    # The 5%, 25%, 50%, 75%, and 95% quantiles of X
    q5 = np.percentile(X, 5)
    q25 = np.percentile(X, 25)
    q50 = np.percentile(X, 50)
    q75 = np.percentile(X, 75)
    q95 = np.percentile(X, 95)
    print("5% quantile: {}\n25% quantile: {}\n50% quantile: {}\n75% quantile: {}\n95% quantile: {}\n".format(q5, q25, q50,
                                                                                                           q75, q95))
