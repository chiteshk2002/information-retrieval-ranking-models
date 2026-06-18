# IFN647 Assignment 2 - evaluation functions
# Implements AP/MAP, Precision@10, DCG@10 using relevance judgement files.

import os
import math
import csv


def read_relevance_file(filepath):
    rel = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                rel[parts[1]] = int(float(parts[2]))
    return rel


def read_ranking_file(filepath):
    ranking = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if first and parts[0].lower().startswith("doc"):
                first = False
                continue
            first = False
            if len(parts) >= 2:
                ranking.append(parts[0])
    return ranking


def average_precision(ranking, relevance):
    total_relevant = len([docid for docid, rel in relevance.items() if rel > 0])
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = 0
    precision_sum = 0.0
    for i, docid in enumerate(ranking, start=1):
        if relevance.get(docid, 0) > 0:
            retrieved_relevant += 1
            precision_sum += retrieved_relevant / float(i)
    return precision_sum / float(total_relevant)


def precision_at_10(ranking, relevance):
    top10 = ranking[:10]
    if len(top10) == 0:
        return 0.0
    relevant_count = sum(1 for docid in top10 if relevance.get(docid, 0) > 0)
    return relevant_count / 10.0


def dcg_at_10(ranking, relevance):
    dcg = 0.0
    for i, docid in enumerate(ranking[:10], start=1):
        rel = 1 if relevance.get(docid, 0) > 0 else 0
        if i == 1:
            dcg += rel
        else:
            dcg += rel / math.log(i, 2)
    return dcg


def evaluate_model_outputs(topics, rel_dir, output_dir):
    models = {
        "Baseline1": "Baseline1_{topic}_Ranking.dat",
        "Baseline2": "Baseline2_{topic}_Ranking.dat",
        "Model_C": "ModelC_{topic}_Ranking.dat",
    }
    rows = []
    for topic_id in sorted(topics.keys()):
        dataset_name = topic_id.replace("R", "Dataset")
        rel_path = os.path.join(rel_dir, dataset_name + ".txt")
        if not os.path.exists(rel_path):
            continue
        relevance = read_relevance_file(rel_path)
        for model_name, pattern in models.items():
            rank_path = os.path.join(output_dir, pattern.format(topic=topic_id))
            if not os.path.exists(rank_path):
                continue
            ranking = read_ranking_file(rank_path)
            rows.append({
                "Topic": topic_id,
                "Model": model_name,
                "AveragePrecision": average_precision(ranking, relevance),
                "Precision@10": precision_at_10(ranking, relevance),
                "DCG@10": dcg_at_10(ranking, relevance),
            })
    return rows


def summarise_results(rows):
    summary = []
    models = sorted(set(row["Model"] for row in rows))
    for model in models:
        model_rows = [row for row in rows if row["Model"] == model]
        if not model_rows:
            continue
        summary.append({
            "Model": model,
            "MAP": sum(r["AveragePrecision"] for r in model_rows) / len(model_rows),
            "Average_P@10": sum(r["Precision@10"] for r in model_rows) / len(model_rows),
            "Average_DCG@10": sum(r["DCG@10"] for r in model_rows) / len(model_rows),
        })
    return summary


def write_csv(filepath, rows, fieldnames):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
