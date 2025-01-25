all_api_requests = {
    "steam_web_api": {
        "get_app_list": {
            "url": "http://api.steampowered.com/ISteamApps/GetAppList/v2/",
            "method": "GET",
            "parameters": {},
            "description": "Получение списка всех приложений Steam."
        },
        "resolve_vanity_url": {
            "url": "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/",
            "method": "GET",
            "parameters": {
                "vanityurl": "Пользовательский vanity URL Steam (обязательный)"
            },
            "description": "Преобразует vanity URL в Steam ID."
        },
         "get_owned_games": {
            "url": "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/",
             "method": "GET",
            "parameters": {
                "steamid": "SteamID64 пользователя (обязательный)",
                "format": "json",
                "include_appinfo": "Включать информацию о приложении (необязательный, по умолчанию 0)",
                 "include_played_free_games": "Включать бесплатные игры (необязательный, по умолчанию 0)"
            },
            "description": "Получает список игр, принадлежащих пользователю Steam."
        },
        "get_app_details": {
            "url": "http://store.steampowered.com/api/appdetails/",
            "method": "GET",
            "parameters": {
                "appids": "ID приложения (обязательный)",
                "cc": "Код валюты (необязательный, по умолчанию 'us')",
                "l": "Код языка (необязательный, по умолчанию 'en')"
            },
             "description": "Получение подробной информации об одном приложении (игре)."
        }
    },
    "steamspy_api": {
        "get_app_details": {
            "url": "https://steamspy.com/api.php",
            "method": "GET",
            "parameters": {
               "request": "Всегда 'appdetails' (обязательный)",
               "appid": "ID приложения (обязательный)"
            },
            "description": "Получение дополнительной информации об игре."
        }
    }
}
"""
Словарь `all_api_requests` содержит определения для всех API запросов, используемых в проекте,
разделенные по категориям: `steam_web_api` для официального Steam Web API и `steamspy_api` для SteamSpy API.

Каждый запрос API определен словарем, включающим следующие ключи:
    - `url`: URL-адрес API endpoint.
    - `method`: HTTP метод запроса (GET, POST и т.д.).
    - `parameters`: Словарь параметров запроса. Ключи словаря - имена параметров, значения - описания параметров и указание на обязательность.
    - `description`: Описание назначения и функциональности API запроса.

Этот словарь служит централизованным хранилищем конфигураций для взаимодействия с различными API,
облегчая поддержку и модификацию API запросов в рамках проекта.
"""