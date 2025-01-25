import os
import requests
from dotenv import load_dotenv
from steam_constants import all_api_requests

load_dotenv()

STEAM_API_KEY = os.getenv("STEAM_API_KEY")

class ApiClient:
    """
    Класс для выполнения запросов к API Steam и SteamSpy.

    Инициализирует клиент API, позволяющий отправлять запросы к различным эндпоинтам
    Steam Web API и SteamSpy API, используя конфигурации, определенные в steam_constants.py.
    Обеспечивает обработку ответов API и ошибок.
    """
    def __init__(self, api_name):
        """
        Инициализирует ApiClient для указанного имени API.

        Аргументы:
            api_name (str): Имя API, для которого создается клиент.
                            Должно соответствовать ключу верхнего уровня в словаре `all_api_requests`
                            в файле `steam_constants.py` (например, 'steam_web_api' или 'steamspy_api').
        """
        self.api_name = api_name
        self.api_requests = all_api_requests.get(api_name, {})

    def _check_api_request(self, url, params=None, method='GET'):
        """Выполняет HTTP-запрос к API и обрабатывает ответ.

        Отправляет HTTP-запрос (GET или POST) по указанному URL с заданными параметрами.
        Обрабатывает ответ, проверяет статус код и пытается распарсить JSON-ответ, если Content-Type указывает на JSON.
        В случае ошибок HTTP или проблем с парсингом JSON, возвращает структурированный словарь с информацией об ошибке.

        Аргументы:
            url (str): URL-адрес API endpoint.
            params (dict, optional): Словарь параметров запроса. По умолчанию None.
            method (str, optional): HTTP метод запроса ('GET' или 'POST'). По умолчанию 'GET'.

        Возвращает:
            dict: Словарь, содержащий результаты запроса. Структура словаря:
                - 'status_code' (int или None): HTTP статус код ответа, None в случае ошибки запроса.
                - 'response_json' (dict или None): Распарсенный JSON-ответ, если Content-Type ответа 'application/json', иначе None.
                - 'error' (str или None): Описание ошибки, если произошла ошибка при выполнении запроса или парсинге ответа, иначе None.
        """
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, params=params)
            else:
                return {
                    'status_code': None,
                    'response_json': None,
                    'error': 'Недопустимый HTTP метод'
                }
            response.raise_for_status()

            response_content_type = response.headers.get('Content-Type', '')

            if 'application/json' in response_content_type:
                try:
                    response_json = response.json()
                except ValueError:
                    return {
                        'status_code': response.status_code,
                        'response_json': None,
                        'error': 'Ошибка парсинга JSON ответа'
                    }
            else:
                response_json = None

            return {
                'status_code': response.status_code,
                'response_json': response_json,
                'error': None
            }

        except requests.exceptions.RequestException as e:
            return {
                'status_code': None,
                'response_json': None,
                'error': str(e)
            }

    def make_request(self, request_name, request_params=None):
      """Выполняет API-запрос по имени запроса, используя конфигурацию из steam_constants.py.

      Находит информацию о запросе в `self.api_requests` по имени `request_name`.
      Извлекает URL и HTTP метод из конфигурации запроса.
      Добавляет API ключ Steam Web API к параметрам запроса, если это необходимо (для 'steam_web_api').
      Вызывает метод `_check_api_request` для выполнения HTTP-запроса и обработки ответа.

      Аргументы:
          request_name (str): Название запроса API, определенное в `steam_constants.py`.
          request_params (dict, optional): Словарь параметров запроса. Эти параметры будут объединены
                                           с параметрами, специфичными для API (например, API ключ).
                                           По умолчанию None.

      Возвращает:
          dict: Словарь, содержащий результат API-запроса, в том же формате, что и возвращает `_check_api_request`.
                Включает 'status_code', 'response_json' и 'error'.
                Возвращает словарь с ошибкой, если `request_name` не найден в конфигурации API.
      """
      request_info = self.api_requests.get(request_name)
      if not request_info:
        return {'status_code': None, 'response_json': None, 'error': f'Запрос "{request_name}" не найден в API "{self.api_name}"'}

      url = request_info['url']
      method = request_info['method']

      if request_params:
        params = request_params
      else:
        params = {}

      if self.api_name == 'steam_web_api' and 'key' not in params:
        params['key'] = STEAM_API_KEY

      return self._check_api_request(url, params=params, method=method)