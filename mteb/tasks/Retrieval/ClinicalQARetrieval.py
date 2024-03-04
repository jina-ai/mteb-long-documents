import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ClinicalQARetrieval(AbsTaskRetrieval):

    _EVAL_SPLIT = 'train'

    @property
    def description(self):
        return {
            'name': 'ClinicalQARetrieval',
            'hf_hub_name': 'starmpcc/Asclepius-Synthetic-Clinical-Notes',
            'reference': 'starmpcc/Asclepius-Synthetic-Clinical-Notes',
            "description": (
                "NarrativeQA is a dataset for the task of question answering on long narratives. It consists of "
                "realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. "
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["train"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(self.description['hf_hub_name'], split=f'{self._EVAL_SPLIT}')
        self.queries = {self._EVAL_SPLIT: {str(i): row['question'] for i, row in enumerate(data)}}
        self.corpus = {self._EVAL_SPLIT: {str(row['patient_id']): {'text': row['answer']} for row in data}}
        self.relevant_docs = {self._EVAL_SPLIT: {str(i): {row['patient_id']: 1} for i, row in enumerate(data)}}

        self.data_loaded = True
