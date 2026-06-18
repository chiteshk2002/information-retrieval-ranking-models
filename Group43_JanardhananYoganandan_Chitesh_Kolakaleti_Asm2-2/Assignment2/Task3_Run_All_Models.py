# IFN647 Assignment 2 - main runner
# Runs Baseline1, Baseline2 and Model_C for all topics/datasets, then evaluates outputs.

import os
from Task3_Data_Loader import (
    extract_data_archives, read_stop_words, parse_topics, parse_topic_records, parse_query,
    parse_rcv_coll, calc_df, calc_collection_tf, dataset_number_from_topic
)
from Task1_Baseline1_BM25 import bm25_scores
from Task1_Baseline2_JM import jm_scores
from Task2_ModelC import model_c_scores
from Task3_Output_Generator import write_ranking, write_expansion_log
from Task4_Evaluation import evaluate_model_outputs, summarise_results, write_csv, read_ranking_file
from Task5_Significance_Test import t_tests_from_rows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "ModelOutputs")
EVAL_DIR = os.path.join(BASE_DIR, "EvaluationOutputs")
STOPWORD_FILE = os.path.join(BASE_DIR, "common-english-words.txt")

# Final selected Model_C parameters
FINAL_ALPHA = 0.05
FINAL_LAMBDA = 0.1
FINAL_TOP_DOCS = 5
FINAL_EXPANSION_TERMS = 1
FINAL_EXPANSION_WEIGHT = 1

# Parameter settings used for systematic validation
VALIDATION_SETTINGS = [
    {"alpha": 0.05, "lambda_value": 0.1, "top_docs": 5, "expansion_terms": 1},
    {"alpha": 0.10, "lambda_value": 0.1, "top_docs": 5, "expansion_terms": 1},
    {"alpha": 0.20, "lambda_value": 0.1, "top_docs": 5, "expansion_terms": 1},
    {"alpha": 0.05, "lambda_value": 0.3, "top_docs": 5, "expansion_terms": 1},
    {"alpha": 0.05, "lambda_value": 0.1, "top_docs": 10, "expansion_terms": 1},
    {"alpha": 0.05, "lambda_value": 0.1, "top_docs": 5, "expansion_terms": 3},
]


def build_top10_rows(topics, output_dir):
    """Create appendix-ready top 10 ranking rows for all topics and models."""
    patterns = {
        "Baseline1": "Baseline1_{topic}_Ranking.dat",
        "Baseline2": "Baseline2_{topic}_Ranking.dat",
        "Model_C": "ModelC_{topic}_Ranking.dat",
    }
    rows = []
    for topic_id in sorted(topics.keys()):
        for model_name, pattern in patterns.items():
            path = os.path.join(output_dir, pattern.format(topic=topic_id))
            if not os.path.exists(path):
                continue
            ranking = read_ranking_file(path)[:10]
            for rank, docid in enumerate(ranking, start=1):
                rows.append({"Topic": topic_id, "Model": model_name, "Rank": rank, "Doc_ID": docid})
    return rows


def run_all_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    doc_dir, rel_dir = extract_data_archives(DATA_DIR)
    stop_words = read_stop_words(STOPWORD_FILE)
    topics = parse_topics(os.path.join(DATA_DIR, "Topics.txt"))
    topic_records = parse_topic_records(os.path.join(DATA_DIR, "Topics.txt"))
    expansion_log = {}

    print("Total topics:", len(topics))

    for topic_id in sorted(topics.keys()):
        title = topics[topic_id]
        dataset_name = dataset_number_from_topic(topic_id)
        dataset_path = os.path.join(doc_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            print("Skipping", topic_id, "because dataset folder is missing:", dataset_path)
            continue

        print("Processing", topic_id, "-", title)
        coll = parse_rcv_coll(dataset_path, stop_words)
        df = calc_df(coll)
        collection_tf = calc_collection_tf(coll)

        # Assignment baselines use topic title as the query.
        title_query = parse_query(title, stop_words)

        # Model_C uses a richer course-style information-need representation:
        # title + description, then validation-selected hybrid parameters.
        model_c_query = title_query

        baseline1 = bm25_scores(coll, title_query, df, k1=1.2, k2=500, b=0.75)
        baseline2 = jm_scores(coll, title_query, collection_tf, lambda_value=0.3)

        # Model_C applies pseudo relevance feedback: it uses the top 5 BM25-ranked
        # documents to add 1 expansion term, then re-ranks using a hybrid score.
        # These parameters must be justified in the validation section of the report.
        model_c, new_terms = model_c_scores(
            coll, model_c_query, df, collection_tf,
            alpha=FINAL_ALPHA, lambda_value=FINAL_LAMBDA,
            top_docs=FINAL_TOP_DOCS, expansion_terms=FINAL_EXPANSION_TERMS,
            expansion_weight=FINAL_EXPANSION_WEIGHT,
            k1=1.2, k2=500, b=0.75
        )
        expansion_log[topic_id] = new_terms

        write_ranking(os.path.join(OUTPUT_DIR, f"Baseline1_{topic_id}_Ranking.dat"), baseline1, "BM25_Score")
        write_ranking(os.path.join(OUTPUT_DIR, f"Baseline2_{topic_id}_Ranking.dat"), baseline2, "JM_Score")
        write_ranking(os.path.join(OUTPUT_DIR, f"ModelC_{topic_id}_Ranking.dat"), model_c, "ModelC_Score")

    write_expansion_log(os.path.join(EVAL_DIR, "model_c_expansion_terms.txt"), expansion_log)

    rows = evaluate_model_outputs(topics, rel_dir, OUTPUT_DIR)
    summary = summarise_results(rows)

    write_csv(os.path.join(EVAL_DIR, "per_topic_evaluation.csv"), rows,
              ["Topic", "Model", "AveragePrecision", "Precision@10", "DCG@10"])
    write_csv(os.path.join(EVAL_DIR, "evaluation_summary.csv"), summary,
              ["Model", "MAP", "Average_P@10", "Average_DCG@10"])
    print("\nTask 4: Evaluation Summary")
    print("Model MAP Average_P@10 Average_DCG@10")
    for row in summary:
        print(row["Model"], row["MAP"], row["Average_P@10"], row["Average_DCG@10"])

    top10_rows = build_top10_rows(topics, OUTPUT_DIR)
    write_csv(os.path.join(EVAL_DIR, "top10_rankings.csv"), top10_rows,
              ["Topic", "Model", "Rank", "Doc_ID"])

    tests = []
    for metric in ["AveragePrecision", "Precision@10", "DCG@10"]:
        tests.extend(t_tests_from_rows(rows, metric))
    write_csv(os.path.join(EVAL_DIR, "significance_tests.csv"), tests,
              ["Metric", "Comparison", "t_statistic", "p_value"])
    print("\nTask 5: Significance Test")
    print("Metric Comparison t_statistic p_value")
    for row in tests:
        print(row["Metric"], row["Comparison"], row["t_statistic"], row["p_value"])

    print(
        "\nTask 6: Application scenario and future work are written in the final report. No Python output is required for Task 6.")

    print("Done. Ranking files saved in ModelOutputs and evaluation files saved in EvaluationOutputs.")


if __name__ == "__main__":
    run_all_models()
