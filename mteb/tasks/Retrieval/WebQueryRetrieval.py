import tempfile
import json
import os
import urllib.request
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WebQueryRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = 'test'
    
    @property
    def description(self):
        return {
            'name': 'WebQueryRetrieval',
            'reference': 'https://github.com/microsoft/CodeXGLUE',
            "description": (
                "Researchers from Microsoft Research Asia, Developer Division, and Bing introduce CodeXGLUE, "
                "a benchmark dataset and open challenge for code intelligence. "
                "It includes a collection of code intelligence tasks and a platform for model evaluation and comparison."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "mrr",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        with tempfile.TemporaryDirectory() as tmp:
            url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/NL-code-search-WebQuery/data/test_webquery.json"
            test_file = os.path.join(tmp, "test_webquery.json")
            urllib.request.urlretrieve(url, test_file)
            with open(test_file, "r") as f:
                data = json.load(f)
            os.remove(test_file)

        self.queries = {self._EVAL_SPLIT: {}}
        self.corpus = {self._EVAL_SPLIT: {}}
        self.relevant_docs = {self._EVAL_SPLIT: {}}
        for idx, row in enumerate(data):
            code = row['code']
            query = row['doc']
            self.queries[self._EVAL_SPLIT][f'q{idx}'] = query
            self.corpus[self._EVAL_SPLIT][f'd{idx}'] = {'text': code}
            self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'] = {f'd{idx}': 1}

        self.data_loaded = True
