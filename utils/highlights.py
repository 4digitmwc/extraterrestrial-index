import pandas as pd
import numpy as np

def standardize_scores(score_table: pd.DataFrame):
    columns = score_table.columns
    for column in columns:
        score_table[column] = (score_table[column].values - np.nanmean(score_table[column].values)) / (np.nanstd(score_table[columns].values, ddof=1))
    return score_table

def get_highlighted_scores(player: str, score_records: pd.DataFrame, n_scores: int):
    score_records['beatmap'] = score_records['round'] + "_" + score_records['beatmap_type'] + "_" + score_records['beatmap_tag'].astype(str)
    score_table = score_records.pivot('player_name', 'beatmap', 'score_logit')
    # standardize the scores
    standardized_score_table = standardize_scores(score_table)
    
    player_standardized_score = standardized_score_table.loc[player].T
    
    return player_standardized_score.sort_values(ascending=False).head(n_scores).index
