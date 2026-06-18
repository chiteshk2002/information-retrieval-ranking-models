# IFN647 Assignment 2 - paired t-test without external statistics libraries.
# The t-statistic is exact. The p-value uses a normal approximation through erfc;
# The t-statistic is calculated manually from paired per-topic scores.
# The p-value uses a normal approximation because no external statistics package is used.
# If SciPy is permitted, scipy.stats.ttest_rel can be used for an exact paired t-test p-value.
import math


def paired_t_test(model_c_values, baseline_values):
    pairs = [(c, b) for c, b in zip(model_c_values, baseline_values)]
    n = len(pairs)
    if n < 2:
        return 0.0, 1.0
    diffs = [c - b for c, b in pairs]
    mean_diff = sum(diffs) / float(n)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / float(n - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        if mean_diff == 0:
            return 0.0, 1.0
        return float("inf"), 0.0
    t_stat = mean_diff / (sd / math.sqrt(n))
    # Two-tailed normal approximation to p-value.
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return t_stat, p_value


def t_tests_from_rows(rows, metric):
    topic_map = {}
    for row in rows:
        topic_map.setdefault(row["Topic"], {})[row["Model"]] = row[metric]

    model_c = []
    b1 = []
    b2 = []
    for topic in sorted(topic_map.keys()):
        vals = topic_map[topic]
        if "Model_C" in vals and "Baseline1" in vals and "Baseline2" in vals:
            model_c.append(vals["Model_C"])
            b1.append(vals["Baseline1"])
            b2.append(vals["Baseline2"])

    t1, p1 = paired_t_test(model_c, b1)
    t2, p2 = paired_t_test(model_c, b2)
    return [
        {"Metric": metric, "Comparison": "Model_C vs Baseline1", "t_statistic": t1, "p_value": p1},
        {"Metric": metric, "Comparison": "Model_C vs Baseline2", "t_statistic": t2, "p_value": p2},
    ]
