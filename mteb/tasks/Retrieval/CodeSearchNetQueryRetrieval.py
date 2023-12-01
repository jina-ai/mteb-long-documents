import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from .CodeSearchNetAdvRetrieval import remove_comments_and_docstrings
import tempfile
import os
import urllib.request
import csv
from itertools import islice

class CodeSearchNetRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = 'python'

    @property
    def description(self):
        return {
            'name': 'CodeSearchNetRetrieval',
            'hf_hub_name': 'jinaai/code_search_net_dedupe',
            'reference': 'https://github.com/github/CodeSearchNet',
            "description": (
                "CodeSearchNet is a collection of datasets and benchmarks that explore the problem of code retrieval using natural language."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["python", "java", "javascript", "go", "php", "ruby"],
            "eval_langs": ["en"],
            "main_score": "mrr",
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
            code = remove_comments_and_docstrings(row['function'], remove_comments=False)
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
                    if lang == self._EVAL_SPLIT and relevance > 0:
                        filtered_ground_truth.append((query, url, relevance))

        distinct_queries = set(row[0] for row in filtered_ground_truth)
        for idx, query in enumerate(distinct_queries):
            self._queries[self._EVAL_SPLIT][f'q{idx}'] = query
            self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'] = {}
            relevant_docs = filter(lambda row: row[0] == query, filtered_ground_truth)
            for _, url, relevance in relevant_docs:
                self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'][url_to_id[url]] = relevance

        self.data_loaded = True
