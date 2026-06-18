# IFN647 Assignment 2 - Baseline2 Jelinek-Mercer language model
# Implements the assignment JM equation using lambda=0.3.

import math


def jm_scores(coll, query_terms, collection_tf, lambda_value=0.3):
    scores = {}
    total_collection_terms = float(sum(collection_tf.values()))
    if total_collection_terms == 0:
        return scores

    for docid, doc in coll.get_docs().items():
        doc_len = float(doc.get_doc_len())
        score = 0.0
        for term, qf in query_terms.items():
            fq_d = doc.get_term_count(term)
            cq = collection_tf.get(term, 0)
            doc_prob = (fq_d / doc_len) if doc_len > 0 else 0.0
            coll_prob = cq / total_collection_terms
            prob = (1 - lambda_value) * doc_prob + lambda_value * coll_prob
            if prob > 0:
                score += qf * math.log10(prob)
            else:
                # Very small fallback to avoid log(0) while retaining low relevance.
                score += qf * math.log10(1e-12)
        scores[docid] = score
    return scores
