import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CodeSearchNetRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'CodeSearchNetRetrieval',
            'hf_hub_name': 'code_search_net',
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
            func_doc_tokens = ' '.join(row['func_documentation_tokens'])
            if func_doc_tokens == '' or len(row['func_documentation_tokens']) <= 3:
                continue
            if func_doc_tokens in q:
                continue
            if row['func_code_string'] in d:
                continue
            q.add(func_doc_tokens)
            d.add(row['func_code_string'])
            self.queries[self._EVAL_SPLIT][f'q{idx}'] = func_doc_tokens
            self.corpus[self._EVAL_SPLIT][f'd{idx}'] = {'text': row['func_code_string']}
            self.relevant_docs[self._EVAL_SPLIT][f'q{idx}'] = {f'd{idx}': 1}

        self.data_loaded = True
