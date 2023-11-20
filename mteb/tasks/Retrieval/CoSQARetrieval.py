import tempfile
import json
import os
import urllib.request
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CoSQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = 'dev'

    @property
    def description(self):
        return {
            'name': 'CoSQARetrieval',
            'reference': 'https://github.com/microsoft/CodeXGLUE',
            "description": (
                "Researchers from Microsoft Research Asia, Developer Division, and Bing introduce CodeXGLUE, "
                "a benchmark dataset and open challenge for code intelligence. "
                "It includes a collection of code intelligence tasks and a platform for model evaluation and comparison."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en"],
            "main_score": "mrr",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        with tempfile.TemporaryDirectory() as tmp:
            url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-dev.json"
            test_file = os.path.join(tmp, "test_cosqa.json")
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
            label = row['label']
            self.corpus[self._EVAL_SPLIT][f'd{idx}'] = {'text': code}
            if label == 1:
                self.queries[self._EVAL_SPLIT][f'q{idx}'] = query
                self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'] = {f'd{idx}': 1}

        self.data_loaded = True
