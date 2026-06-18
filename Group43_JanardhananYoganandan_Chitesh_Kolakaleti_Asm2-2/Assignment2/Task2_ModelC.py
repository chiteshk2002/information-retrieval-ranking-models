# IFN647 Assignment 2 - Model_C
# Custom model using only course-style methods:
# 1. Initial BM25 ranking.
# 2. Pseudo relevance feedback: assume top-ranked documents are relevant.
# 3. Expand the title query using frequent/high-value terms from the pseudo-relevant documents.
# 4. Re-rank documents using a weighted hybrid of normalised BM25 and normalised JM scores.

from collections import Counter
from Task1_Baseline1_BM25 import bm25_scores
from Task1_Baseline2_JM import jm_scores


def min_max_normalise(scores):
    if not scores:
        return {}
    values = list(scores.values())
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {docid: 0.0 for docid in scores}
    return {docid: (score - min_v) / float(max_v - min_v) for docid, score in scores.items()}


def select_expansion_terms(coll, initial_scores, original_query_terms, top_docs=5, expansion_terms=10):
    # Common Reuters/XML artefacts are removed so the expansion log remains interpretable.
    blocked_terms = set(['quot', 'amp', 'lt', 'gt', 'pct', 'mln', 'dlrs', 'reuter', 'said'])
    ranked_docs = [docid for docid, score in sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)]
    pseudo_docs = ranked_docs[:top_docs]
    expansion_counter = Counter()

    for docid in pseudo_docs:
        doc = coll.get_doc(docid)
        if doc is None:
            continue
        expansion_counter.update(doc.get_term_freq_dict())

    for term in list(original_query_terms.keys()):
        if term in expansion_counter:
            del expansion_counter[term]

    selected_terms = []
    for term, freq in expansion_counter.most_common():
        if term in blocked_terms:
            continue
        if len(term) <= 3 or not term.isalpha():
            continue
        selected_terms.append(term)
        if len(selected_terms) == expansion_terms:
            break
    return selected_terms


def expanded_query_terms(original_query_terms, expansion_terms, expansion_weight=1):
    q = Counter(original_query_terms)
    for term in expansion_terms:
        q[term] += expansion_weight
    return q

def model_c_scores(coll, query_terms, df, collection_tf,
                       alpha=0.05, lambda_value=0.1,
                       top_docs=5, expansion_terms=1, expansion_weight=1,
                       k1=1.2, k2=500, b=0.75):
    # If expansion_terms is 0, validation has selected no pseudo-relevance expansion.
    # Otherwise, use initial BM25 ranking to select expansion terms from pseudo-relevant documents.
    if expansion_terms > 0 and top_docs > 0:
        initial_bm25 = bm25_scores(coll, query_terms, df, k1=k1, k2=k2, b=b)
        new_terms = select_expansion_terms(coll, initial_bm25, query_terms, top_docs, expansion_terms)
        q_expanded = expanded_query_terms(query_terms, new_terms, expansion_weight)
    else:
        new_terms = []
        q_expanded = query_terms

    bm25_expanded = bm25_scores(coll, q_expanded, df, k1=k1, k2=k2, b=b)
    jm_expanded = jm_scores(coll, q_expanded, collection_tf, lambda_value=lambda_value)

    bm25_norm = min_max_normalise(bm25_expanded)
    jm_norm = min_max_normalise(jm_expanded)

    scores = {}
    for docid in coll.get_docs().keys():
        scores[docid] = alpha * bm25_norm.get(docid, 0.0) + (1 - alpha) * jm_norm.get(docid, 0.0)
    return scores, new_terms
