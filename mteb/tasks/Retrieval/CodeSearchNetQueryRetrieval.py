import datasets
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.MultilingualTask import MultilingualTask
import tempfile
import os
import urllib.request
import csv
from itertools import islice
from tqdm import tqdm

def filter_docs(docs, code, lang):
    if docs in code and lang == 'python':
        start_doc = code.find('"""')
        end_doc = code.find('"""', start_doc + 1)
        if start_doc < 0 or end_doc < 0:
            start_doc = code.find("'''")
            end_doc = code.find("'''", start_doc + 1)
            if start_doc < 0 or end_doc < 0:
                return docs
        code = code[:start_doc].rstrip(' ') + code[end_doc + 3:].lstrip('\n')
    return code


class CodeSearchNetQueryRetrieval(AbsTaskRetrieval, MultilingualTask):
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
            "eval_splits": ["test"],
            "eval_langs": ["python", "java", "javascript", "go", "php", "ruby"],
            "main_score": "mrr_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        eval_splits = kwargs.get("eval_splits")
        eval_splits = eval_splits if eval_splits is not None else self.description["eval_splits"]
        assert eval_splits == [] or eval_splits == ["test"]

        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_url = 'https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/annotationStore.csv'
            data_path = os.path.join(tmpdir, 'annotationStore.csv')
            urllib.request.urlretrieve(annotation_url, data_path)
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                annotation_store = list(reader)

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        for lang in self.langs:
            self.queries[lang] = {}
            self.corpus[lang] = {}
            self.relevant_docs[lang] = {}
            print(f'Loading data for {lang}')
            for split in eval_splits:
                data = datasets.load_dataset(self.description['hf_hub_name'], split=lang)
                self.queries[lang][split] = {}
                self.corpus[lang][split] = {}
                self.relevant_docs[lang][split] = {}

                url_to_id = {}
                for idx, row in tqdm(enumerate(data), total=len(data)):
                    code = filter_docs(row['docstring'], row['function'], lang)
                    self.corpus[lang][split][f'd{idx}'] = {'text': code}
                    url_to_id[row['url']] = f'd{idx}'

                filtered_ground_truth = []
                for row in islice(annotation_store, 1, None):
                    row_lang, query, url, relevance, _ = row
                    relevance = int(relevance)
                    if row_lang.lower() == lang and relevance > 0:
                        filtered_ground_truth.append((query, url, relevance))

                distinct_queries = set(row[0] for row in filtered_ground_truth)
                for idx, query in enumerate(distinct_queries):
                    self.queries[lang][split][f'q{idx}'] = query
                    self.relevant_docs[lang][split][f'q{idx}'] = {}
                    relevant_docs = filter(lambda row: row[0] == query, filtered_ground_truth)
                    for _, url, relevance in relevant_docs:
                        self.relevant_docs[lang][split][f'q{idx}'][url_to_id[url]] = relevance

        self.data_loaded = True
