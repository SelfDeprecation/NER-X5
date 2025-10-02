import ast
import pandas as pd


def read_train_csv(path: str):
    df = pd.read_csv(path, sep=';')
    if 'sample' not in df.columns or 'annotation' not in df.columns:
        raise ValueError('train csv must contain columns: sample, annotation')
    def parse_ann(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except Exception:
            return [t.strip() for t in str(x).strip("[]").split(',') if t.strip()]
    df['annotation_parsed'] = df['annotation'].apply(parse_ann)
    df['sample'] = df['sample'].astype(str).str.strip()
    return df


def read_submission_csv(path: str):
    df = pd.read_csv(path, sep=';')
    if 'sample' not in df.columns:
        raise ValueError('submission csv must contain column: sample')
    df['sample'] = df['sample'].astype(str).str.strip()
    return df
