# IFN647 Assignment 2 - data loading and preprocessing
# This file follows the parsing/pre-processing style used in the IFN647 workshop code:
# read RCV XML files, use <newsitem itemid>, process text inside <text>, remove punctuation/digits,
# lower-case, stem terms, remove common-Englishwords, and represent documents as bag-of-words.

import os
import re
import glob
import zipfile
import string
from collections import Counter
from stemming.porter2 import stem


class BowDoc:
    def __init__(self, docid):
        self.docid = str(docid)
        self.terms = Counter()
        self.doc_len = 0

    def add_term(self, term):
        self.terms[term] += 1

    def set_doc_len(self, doc_len):
        self.doc_len = doc_len

    def get_docid(self):
        return self.docid

    def get_term_count(self, term):
        return self.terms.get(term, 0)

    def get_term_list(self):
        return list(self.terms.keys())

    def get_term_freq_dict(self):
        return dict(self.terms)

    def get_doc_len(self):
        return self.doc_len


class BowColl:
    def __init__(self):
        self.docs = {}

    def add_doc(self, doc):
        self.docs[doc.get_docid()] = doc

    def get_docs(self):
        return self.docs

    def get_doc(self, docid):
        return self.docs.get(str(docid))

    def get_num_docs(self):
        return len(self.docs)

    def get_total_terms(self):
        return sum(doc.get_doc_len() for doc in self.docs.values())


def extract_data_archives(data_dir):
    """Extract Doc_Collection.zip and Relevant_Judgements.zip only if needed."""
    doc_dir = os.path.join(data_dir, "Doc_Collection")
    rel_dir = os.path.join(data_dir, "Relevant_Judgements")

    if not os.path.isdir(doc_dir):
        with zipfile.ZipFile(os.path.join(data_dir, "Doc_Collection.zip"), "r") as zip_ref:
            zip_ref.extractall(data_dir)

    if not os.path.isdir(rel_dir):
        with zipfile.ZipFile(os.path.join(data_dir, "Relevant_Judgements.zip"), "r") as zip_ref:
            zip_ref.extractall(data_dir)

    return doc_dir, rel_dir


def read_stop_words(stopword_file):
    with open(stopword_file, "r", encoding="utf-8", errors="ignore") as f:
        return set([w.strip() for w in f.read().split(',') if w.strip()])


def preprocess_text(text, stop_words):
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    terms = []
    for raw in text.split():
        term = stem(raw.lower())
        if len(term) > 2 and term not in stop_words:
            terms.append(term)
    return terms


def parse_query(query_text, stop_words):
    return Counter(preprocess_text(query_text, stop_words))


def parse_topics(topic_file):
    """Return {topic_id: title}. Baseline queries must use the title."""
    with open(topic_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    topics = {}
    blocks = re.findall(r"<Topic>(.*?)</Topic>", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        num_match = re.search(r"<num>\s*Number:\s*(R\d+)", block, flags=re.IGNORECASE)
        title_match = re.search(r"<title>\s*(.*?)(?:\n\s*<desc>|\n\s*<narr>|$)", block, flags=re.DOTALL | re.IGNORECASE)
        if num_match and title_match:
            topic_id = num_match.group(1).strip()
            title = " ".join(title_match.group(1).strip().split())
            topics[topic_id] = title
    return topics


def parse_rcv_coll(inputpath, stop_words):
    """Parse RCV XML files in one Dataset folder into a BowColl."""
    coll = BowColl()
    xml_files = glob.glob(os.path.join(inputpath, "*.xml"))

    for file_path in xml_files:
        curr_doc = None
        inside_text = False
        word_count = 0
        with open(file_path, "r", encoding="iso-8859-1", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if inside_text is False:
                    if line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                docid = part.split("=")[1].split('"')[1]
                                curr_doc = BowDoc(docid)
                                break
                    if line.startswith("<text>"):
                        inside_text = True
                elif line.startswith("</text>"):
                    break
                elif curr_doc is not None:
                    line = line.replace("<p>", "").replace("</p>", "")
                    raw_no_tags = re.sub(r"<[^>]+>", " ", line)
                    raw_terms = raw_no_tags.translate(str.maketrans('', '', string.digits))
                    raw_terms = raw_terms.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).split()
                    word_count += len(raw_terms)
                    for term in preprocess_text(raw_no_tags, stop_words):
                        curr_doc.add_term(term)
        if curr_doc is not None:
            curr_doc.set_doc_len(word_count)
            coll.add_doc(curr_doc)
    return coll


def calc_df(coll):
    df = {}
    for docid, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            df[term] = df.get(term, 0) + 1
    return df


def calc_collection_tf(coll):
    ctf = Counter()
    for docid, doc in coll.get_docs().items():
        ctf.update(doc.get_term_freq_dict())
    return ctf


def dataset_number_from_topic(topic_id):
    return topic_id.replace("R", "Dataset")


def parse_topic_records(topic_file):
    """Return {topic_id: {'title':..., 'desc':..., 'narr':...}} for Model_C experiments."""
    with open(topic_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    records = {}
    blocks = re.findall(r"<Topic>(.*?)</Topic>", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        num_match = re.search(r"<num>\s*Number:\s*(R\d+)", block, flags=re.IGNORECASE)
        title_match = re.search(r"<title>\s*(.*?)(?:\n\s*<desc>|\n\s*<narr>|$)", block, flags=re.DOTALL | re.IGNORECASE)
        desc_match = re.search(r"<desc>\s*Description:\s*(.*?)(?:\n\s*<narr>|$)", block, flags=re.DOTALL | re.IGNORECASE)
        narr_match = re.search(r"<narr>\s*Narrative:\s*(.*)$", block, flags=re.DOTALL | re.IGNORECASE)
        if num_match and title_match:
            topic_id = num_match.group(1).strip()
            records[topic_id] = {
                "title": " ".join(title_match.group(1).strip().split()),
                "desc": " ".join(desc_match.group(1).strip().split()) if desc_match else "",
                "narr": " ".join(narr_match.group(1).strip().split()) if narr_match else "",
            }
    return records
