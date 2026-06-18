# IFN647 Assignment 2 - output helper functions

import os


def write_ranking(filepath, scores, score_name="Score"):
    """Write ranking files without a header line.

    The assignment examples use plain ranking lines, so each line contains:
        Doc_ID Score
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Doc_ID " + score_name + "\n")
        for docid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            f.write(str(docid) + " " + str(score) + "\n")


def write_expansion_log(filepath, expansion_log):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Topic ExpansionTerms\n")
        for topic_id in sorted(expansion_log.keys()):
            f.write(topic_id + " " + ",".join(expansion_log[topic_id]) + "\n")
