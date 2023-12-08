import os
import json
import urllib.request
import tempfile
from ...abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class CoSQAPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "CoSQAPC",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["dev", "train"],
            "eval_langs": ["en"],
            "main_score": "ap",
            "revision": "70970daeab8776df92f5ea462b6173c0b46fd2d1",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        eval_splits = kwargs.get("eval_splits")
        eval_splits = eval_splits if eval_splits is not None else self.description["eval_splits"]

        self.dataset = {}
        for split in eval_splits:
            with tempfile.TemporaryDirectory() as tmp:
                url = f"https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-{split}.json"
                print(url)
                test_file = os.path.join(tmp, "test_cosqa.json")
                urllib.request.urlretrieve(url, test_file)
                with open(test_file, "r") as f:
                    data = json.load(f)
                os.remove(test_file)

            self.dataset[split] = [{'sent1': [], 'sent2': [], 'labels': []},]

            for row in data:
                self.dataset[split][0]['sent1'].append(row['code'])
                self.dataset[split][0]['sent2'].append(row['doc'])
                self.dataset[split][0]['labels'].append(row['label'])

        self.data_loaded = True