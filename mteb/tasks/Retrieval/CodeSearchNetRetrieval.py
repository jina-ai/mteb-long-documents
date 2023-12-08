import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.MultilingualTask import MultilingualTask


class CodeSearchNetRetrieval(AbsTaskRetrieval, MultilingualTask):
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
            "eval_splits": ["validation", "test"],
            "eval_langs": ["go", "java", "javascript", "php", "python", "ruby"],
            "main_score": "mrr",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        eval_splits = kwargs.get("eval_splits")
        eval_splits = eval_splits if eval_splits is not None else self.description["eval_splits"]

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        for lang in self.langs:
            if lang not in self.description['eval_langs']:
                continue
            self.queries[lang] = {}
            self.corpus[lang] = {}
            self.relevant_docs[lang] = {}
            for split in eval_splits:
                self.queries[lang][split] = {}
                self.corpus[lang][split] = {}
                self.relevant_docs[lang][split] = {}
                data = datasets.load_dataset(self.description['hf_hub_name'], split=f'{split}.{lang}')
                for idx, row in enumerate(data):
                    code = row['code']
                    query = row['doc']
                    self.queries[lang][split][f'q{idx}'] = query
                    self.corpus[lang][split][f'd{idx}'] = {'text': code}
                    self.relevant_docs[lang][split][f'q{idx}'] = {f'd{idx}': 1}

        self.data_loaded = True
