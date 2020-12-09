from tqdm import tqdm

tqdm.pandas()

def apply_on_df_columns(data, columns, func, verbose=1):
    for col in columns:
        if verbose:
            tqdm.pandas(desc="Processing " + col + " with " + func.__name__)
            data[col] = data[col].progress_apply(func)
        else:
            data[col] = data[col].apply(func)
    return data

from src.preprocessors.naive_preprocessor import NaivePreprocessor
from src.preprocessors.english_preprocessor import ENPreprocessor
from src.preprocessors.chinese_preprocessor import CNPreprocessor