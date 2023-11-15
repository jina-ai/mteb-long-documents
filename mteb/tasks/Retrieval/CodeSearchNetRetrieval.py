import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CodeSearchNetRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'CodeSearchNetRetrieval',
            'hf_hub_name': 'jinaai/code_search_net_clean',
            'reference': 'https://github.com/github/CodeSearchNet',
            "description": (
                "CodeSearchNet is a collection of datasets and benchmarks that explore the problem of code retrieval using natural language."
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

        data = datasets.load_dataset(self.description['hf_hub_name'], split=self._EVAL_SPLIT)
        self.queries = {self._EVAL_SPLIT: {}}
        self.corpus = {self._EVAL_SPLIT: {}}
        self.relevant_docs = {self._EVAL_SPLIT: {}}
        q = set()
        d = set()
        for idx, row in enumerate(data):
            code = row['code']
            docs = row['docs']
            if docs in q:
                continue
            if code in d:
                continue
            q.add(docs)
            d.add(code)
            self.queries[self._EVAL_SPLIT][f'q{idx}'] = docs
            self.corpus[self._EVAL_SPLIT][f'd{idx}'] = {'text': code}
            self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'] = {f'd{idx}': 1}

        self.data_loaded = True
