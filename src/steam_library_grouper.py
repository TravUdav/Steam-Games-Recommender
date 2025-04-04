import random
import pandas as pd
import time

def group_user_games(owned_games):
    """Группирует игры пользователя на основе времени игры.

    Функция анализирует список игр пользователя и выделяет наиболее часто и недавно сыгранные игры.

    Аргументы:
        owned_games (list): Список словарей, представляющих игры пользователя.
            Каждый словарь должен содержать ключи 'playtime_2weeks' (время игры за последние 2 недели)
            и 'playtime_forever' (общее время игры).

    Возвращает:
        dict: Словарь, содержащий два списка игр:
            - 'recent_games' (list): Список из 5 недавно сыгранных игр, отсортированных по убыванию 'playtime_2weeks'.
            - 'most_played_games' (list): Список из 5 самых играемых игр, отсортированных по убыванию 'playtime_forever'.
            Если owned_games пуст, возвращает пустые списки для обоих ключей.
    """
    if not owned_games:
        return {
            "recent_games": [],
            "most_played_games": [],
        }

    df_owned_games = pd.DataFrame(owned_games)

    recent_games_base = []
    if not df_owned_games.empty:
        df_recent = df_owned_games.sort_values(by="playtime_2weeks", ascending=False, na_position='last').head(5)
        recent_games_base = df_recent.to_dict(orient="records")

    most_played_games_base = df_owned_games.sort_values(by="playtime_forever", ascending=False).head(5).to_dict(orient="records")

    return {
        "recent_games": recent_games_base,
        "most_played_games": most_played_games_base,
    }