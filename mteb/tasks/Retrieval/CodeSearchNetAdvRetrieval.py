import tempfile
import tokenize
import io
import os
import subprocess
import pandas as pd
import zipfile
import urllib.request
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval

def remove_comments_and_docstrings(source, remove_comments=True):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT and remove_comments:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT and prev_toktype != tokenize.NEWLINE and start_col > 0:
                out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


class CodeSearchNetAdvRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CodeSearchNetAdvRetrieval',
            'reference': 'https://github.com/github/CodeSearchNet',
            "description": (
                "CodeSearchNet is a collection of datasets and benchmarks that explore the problem of code retrieval using natural language."
            ),
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["valid", "test"],
            "eval_langs": ["en"],
            "main_score": "mrr_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        eval_splits = kwargs.get("eval_splits")
        eval_splits = eval_splits if eval_splits is not None else self.description["eval_splits"]

        with tempfile.TemporaryDirectory() as tmp:
            dset_url = "https://github.com/microsoft/CodeXGLUE/blob/e252e54a74dd55b1294e2379b213b1541dfefaf5/Text-Code/NL-code-search-Adv/dataset.zip"
            zenodo_url = "https://zenodo.org/record/7857872/files/python.zip"
            dset_zip_pth = os.path.join(tmp, 'dataset.zip')
            dset_dir = os.path.join(tmp, 'dataset')
            urllib.request.urlretrieve(dset_url, dset_zip_pth)
            with zipfile.ZipFile(dset_zip_pth, 'r') as zip_ref:
                zip_ref.extractall(tmp)

            python_zip_pth = os.path.join(dset_dir, "python.zip")
            urllib.request.urlretrieve(zenodo_url, python_zip_pth)
            with zipfile.ZipFile(python_zip_pth, 'r') as zip_ref:
                zip_ref.extractall(dset_dir)
            status = subprocess.run(['python', 'preprocess.py'], cwd=dset_dir, capture_output=True)
            assert status.returncode == 0, (status.stdout, status.stderr)

            self.queries = {split: {} for split in eval_splits}
            self.corpus = {split: {} for split in eval_splits}
            self.relevant_docs = {split: {} for split in eval_splits}
            for split in eval_splits:
                data_path = os.path.join(dset_dir, f'{split}.jsonl')
                assert os.path.exists(data_path)
                jsonObj = pd.read_json(path_or_buf=data_path, lines=True)

                for code, doc, url, idx in zip(jsonObj['function'], jsonObj['docstring'], jsonObj['url'], jsonObj['idx']):
                    self.queries[split][url] = doc
                    self.corpus[split][str(idx)] = {'text': remove_comments_and_docstrings(code)}
                    self.relevant_docs[split][url] = {str(idx): 1}

        self.data_loaded = True
