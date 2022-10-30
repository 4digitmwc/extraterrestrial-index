import pandas as pd
import numpy as np

def get_highlighted_scores(player_list: str, records: pd.DataFrame, n_scores: int = 5):
    score_records = records.copy(True)
    score_records['beatmap'] = score_records['round'] + "_" + score_records['beatmap_type'] + "_" + score_records['beatmap_tag'].astype(str)
    beatmap_mean = pd.DataFrame(score_records.groupby('beatmap')['score_logit'].mean())
    beatmap_std = pd.DataFrame(score_records.groupby('beatmap')['score_logit'].std(ddof=1))
    score_records['mean'] = score_records['beatmap'].apply(lambda x: beatmap_mean.loc[x])
    score_records['std'] = score_records['beatmap'].apply(lambda x: beatmap_std.loc[x])
    score_records['standardized'] = (score_records['score_logit'] - score_records['mean']) / score_records['std']
    sort_records = score_records.sort_values(['player_name', 'standardized'], ascending=[True, False])
    top_players = sort_records.groupby('player_name').apply(lambda x : x.sort_values(by = 'standardized', ascending = False).head(n_scores).reset_index(drop = True))
    highlight = top_players[top_players['player_name'].isin(player_list)]

    return highlight[['player_id', 'beatmap', 'score', 'standardized']].reset_index().sort_values('player_name')