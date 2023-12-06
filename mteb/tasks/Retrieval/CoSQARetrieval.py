import tempfile
import json
import os
import urllib.request
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CoSQARetrieval(AbsTaskRetrieval):
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
            "eval_splits": ["dev", "train"],
            "eval_langs": ["en"],
            "main_score": "mrr",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        eval_splits = kwargs.get("eval_splits")
        eval_splits = eval_splits if eval_splits is not None else self.description["eval_splits"]

        self.queries = {split: {} for split in eval_splits}
        self.corpus = {split: {} for split in eval_splits}
        self.relevant_docs = {split: {} for split in eval_splits}

        for split in eval_splits:
            with tempfile.TemporaryDirectory() as tmp:
                url = f"https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-{split}.json"
                test_file = os.path.join(tmp, "cosqa.json")
                urllib.request.urlretrieve(url, test_file)
                with open(test_file, "r") as f:
                    data = json.load(f)
                os.remove(test_file)

            for idx, row in enumerate(data):
                code = row['code']
                query = row['doc']
                label = row['label']
                self.corpus[split][f'd{idx}'] = {'text': code}
                if label == 1:
                    self.queries[split][f'q{idx}'] = query
                    self.relevant_docs[split][f'q{idx}'] = {f'd{idx}': 1}

        self.data_loaded = True
