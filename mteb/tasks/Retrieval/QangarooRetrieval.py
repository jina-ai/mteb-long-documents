import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QangarooRetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'test'

    @property
    def description(self):
        return {
            'name': 'QangarooRetrieval',
            'hf_hub_name': 'qangaroo',
            'reference': 'https://qangaroo.cs.ucl.ac.uk/',
            "description": (
                "QAngaroo is a multi hop reading comprehension dataset. The questions can only be answered by taking "
                "multiple sources into account. The version of the dataset serves the purpose of a retrieval dataset. "
                "Thereby, relevant sources for answering a question are concatenated together. In this way it contains "
                "long text snippets where it is not possible to determine relevance only based on a single sentence."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(self.description['hf_hub_name'], 'wikihop', split='validation')
        self.queries = {self._EVAL_SPLIT: {str(i): row['query'] for i, row in enumerate(data)}}
        self.corpus = {self._EVAL_SPLIT: {str(i): {'text': ' '.join(row['supports'])} for i, row in enumerate(data)}}
        self.relevant_docs = {self._EVAL_SPLIT: {str(i): {str(i): 1} for i, row in enumerate(data)}}

        self.data_loaded = True
