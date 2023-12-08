import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
import tempfile
import os
import urllib.request
import csv
from itertools import islice

def _filter_docs(docs, code, lang):
    if docs in code:
        if lang == 'python':
            start_doc = code.find('"""')
            end_doc = code.find('"""', start_doc + 1)
            code = code[:start_doc].rstrip(' ') + code[end_doc + 3:].lstrip('\n')
        else:
            return None
    return code

class CodeSearchNetQueryRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = 'python'

    @property
    def description(self):
        return {
            'name': 'CodeSearchNetQueryRetrieval',
            'hf_hub_name': 'jinaai/code_search_net_dedupe',
            'reference': 'https://github.com/github/CodeSearchNet',
            "description": (
                "CodeSearchNet is a collection of datasets and benchmarks that explore the problem of code retrieval using natural language."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["python", "java", "javascript", "go", "php", "ruby"],
            "eval_langs": ["en"],
            "main_score": "mrr_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(self.description['hf_hub_name'], split=self._EVAL_SPLIT)
        self.queries = {self._EVAL_SPLIT: {}}
        self.corpus = {self._EVAL_SPLIT: {}}
        self.relevant_docs = {self._EVAL_SPLIT: {}}

        url_to_id = {}
        for idx, row in enumerate(data):
            code = _filter_docs(row['docstring'], row['function'], self._EVAL_SPLIT)
            self.corpus[self._EVAL_SPLIT][f'd{idx}'] = {'text': code}
            url_to_id[row['url']] = f'd{idx}'

        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_url = 'https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/annotationStore.csv'
            data_path = os.path.join(tmpdir, 'annotationStore.csv')
            urllib.request.urlretrieve(annotation_url, data_path)
            filtered_ground_truth = []
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                for row in islice(reader, 1, None):
                    lang, query, url, relevance, _ = row
                    relevance = int(relevance)
                    if lang.lower() == self._EVAL_SPLIT and relevance > 0:
                        filtered_ground_truth.append((query, url, relevance))

        distinct_queries = set(row[0] for row in filtered_ground_truth)
        for idx, query in enumerate(distinct_queries):
            self.queries[self._EVAL_SPLIT][f'q{idx}'] = query
            self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'] = {}
            relevant_docs = filter(lambda row: row[0] == query, filtered_ground_truth)
            for _, url, relevance in relevant_docs:
                self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'][url_to_id[url]] = relevance

        self.data_loaded = True
