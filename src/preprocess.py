import pandas as pd
import ast

def extract_entities_with_o(annotation_str):
    try:
        annotation_list = ast.literal_eval(annotation_str)
        entities = []
        for item in annotation_list:
            if len(item) >= 3 and isinstance(item[2], str):
                if item[2] == 'O':
                    entities.append('O')
                else:
                    parts = item[2].split('-', 1)
                    if len(parts) > 1:
                        entities.append(parts[1])
        return entities
    except (ValueError, SyntaxError, TypeError):
        return []

df_train = pd.read_csv('data/train.csv', sep=';')
df_sub = pd.read_csv('data/submission.csv', sep=';')

df_train['annotation'] = df_train['annotation'].apply(extract_entities_with_o)
df_test = df_sub['sample']

df_train.to_csv('data/train_cleaned.csv', sep=';', index=False)
df_test.to_csv('data/submission_cleaned.csv', sep=';', index=False)
