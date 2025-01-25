import os
from dotenv import load_dotenv
from steam_api_client import ApiClient

load_dotenv()
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

class ApiParser:
    """
    Класс для парсинга данных из Steam API и SteamSpy API.

    Инициализирует клиентов API для Steam Web API и SteamSpy API,
    предоставляя методы для выполнения запросов к различным эндпоинтам API.
    """
    def __init__(self):
        """
        Инициализирует ApiParser с клиентами для Steam Web API и SteamSpy API.

        Создает экземпляры ApiClient для 'steam_web_api' и 'steamspy_api',
        используя конфигурации API, определенные в steam_constants.py.
        """
        self.steam_web_api_client = ApiClient("steam_web_api")
        self.steamspy_api_client = ApiClient("steamspy_api")

    def resolve_vanity_url(self, vanity_url):
        """Преобразует vanity URL пользователя Steam в Steam ID.

        Использует Steam Web API для преобразования пользовательского vanity URL
        в 64-битный Steam ID.

        Аргументы:
            vanity_url (str): Пользовательский vanity URL Steam, например, 'gabelogannewell'.

        Возвращает:
            str или None: Steam ID пользователя в виде строки, если преобразование успешно,
                         иначе None, если vanity URL не найден или произошла ошибка.

        Выводит в консоль информационные сообщения о начале запроса и полученном результате.
        """
        print(f"⚙️ Запрос на преобразование vanity URL: {vanity_url}...")
        response = self.steam_web_api_client.make_request(
            request_name="resolve_vanity_url",
            request_params={"vanityurl": vanity_url}
        )
        print(f"✅ Результат преобразования vanity URL: {response}")
        if response["error"] is None and response["response_json"] and response["response_json"]["response"]["success"] == 1:
            return response["response_json"]["response"]["steamid"]
        else:
            return None

    def get_owned_games(self, steam_id):
        """Получает список игр, принадлежащих пользователю Steam, по его Steam ID.

        Использует Steam Web API для запроса списка игр, которыми владеет пользователь,
        включая информацию об играх и время, проведенное в них.

        Аргументы:
            steam_id (str): 64-битный Steam ID пользователя.

        Возвращает:
            list: Список словарей, где каждый словарь представляет игру, принадлежащую пользователю.
                  Возвращает пустой список, если Steam ID не найден или произошла ошибка.
                  Каждый словарь может содержать ключи, такие как 'appid', 'name', 'playtime_forever' и др.

        Выводит в консоль информационные сообщения о начале запроса и полученном результате.
        """
        print(f"⚙️ Запрос списка игр пользователя Steam ID: {steam_id}...")
        response = self.steam_web_api_client.make_request(
            request_name="get_owned_games",
             request_params={
            "steamid": steam_id,
            "format": "json",
            "include_appinfo": 1,
            "include_played_free_games": 1,
            }
        )
        if response["error"] is None and response["response_json"]:
            return response["response_json"].get("response", {}).get("games", [])
        else:
            return []