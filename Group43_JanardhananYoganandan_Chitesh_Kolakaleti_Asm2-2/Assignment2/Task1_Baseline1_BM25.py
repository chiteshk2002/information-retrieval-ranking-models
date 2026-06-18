# IFN647 Assignment 2 - Baseline1 BM25 model
# Implements the assignment BM25 equation using k1=1.2, k2=500, b=0.75.

import math


def avg_doc_len(coll):
    if coll.get_num_docs() == 0:
        return 0.0
    return sum(doc.get_doc_len() for doc in coll.get_docs().values()) / float(coll.get_num_docs())


def bm25_scores(coll, query_terms, df, k1=1.2, k2=500, b=0.75):
    scores = {}
    n_docs = coll.get_num_docs()
    avg_dl = avg_doc_len(coll)

    if n_docs == 0 or avg_dl == 0:
        return scores

    for docid, doc in coll.get_docs().items():
        dl = doc.get_doc_len()
        k_value = k1 * ((1 - b) + b * (dl / float(avg_dl)))
        score = 0.0

        for term, qf in query_terms.items():
            if term not in df:
                continue

            n = df[term]
            f = doc.get_term_count(term)

            if f == 0:
                continue

            # Assignment Equation 1:
            # log10(1 + ((N - n + 0.5) / (n + 0.5))) times TF and QF weights.
            idf = math.log10(1 + ((n_docs - n + 0.5) / (n + 0.5)))
            tf_weight = ((k1 + 1) * f) / (k_value + f)
            qf_weight = ((k2 + 1) * qf) / float(k2 + qf)

            score += idf * tf_weight * qf_weight

        scores[docid] = score

    return scores
