import pandas as pd

def get_table_from_records(records: pd.DataFrame, index: str, column: str, value: str):
    return records.pivot(index, column, value)