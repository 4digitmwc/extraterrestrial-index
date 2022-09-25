import pandas as pd
import numpy as np

def leoflt_performance(records: pd.DataFrame, round: str):
    round_records = records[records['round'] == round]
    median_score = round_records.groupby('beatmap_type_tag').median()[['score']]
    maps_played_by_player = round_records.groupby('player_name').count()[['beatmap_type_tag']]
    median_maps_played = maps_played_by_player.median().loc['beatmap_type_tag']
    
    sum_compared_to_median = pd.DataFrame(np.zeros(len(maps_played_by_player)), index=maps_played_by_player.index, columns=['sum_compared_to_median'])
    
    for record in round_records.values:
        [player_name, round_, type_tag, score] = list(record)
        med = median_score.loc[type_tag, 'score']
        sum_compared_to_median.at[player_name, 'sum_compared_to_median'] += score / med

    # applying formula time

    performance_points = pd.DataFrame(np.zeros(len(maps_played_by_player)), index=maps_played_by_player.index, columns=['performance'])

    for player in performance_points.index:
        ni_mi = sum_compared_to_median.loc[player, 'sum_compared_to_median']
        n = maps_played_by_player.loc[player, 'beatmap_type_tag']
        performance_points.at[player, 'performance'] = ni_mi / n * np.cbrt(n / median_maps_played)
    
    return performance_points