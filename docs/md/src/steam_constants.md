# Module `src.steam_constants`

## Global variables

`var all_api_requests`

    

Словарь `all_api_requests` содержит определения для всех API запросов,
используемых в проекте, разделенные по категориям: `steam_web_api` для
официального Steam Web API и `steamspy_api` для SteamSpy API.

Каждый запрос API определен словарем, включающим следующие ключи: \- `url`:
URL-адрес API endpoint. \- `method`: HTTP метод запроса (GET, POST и т.д.). \-
`parameters`: Словарь параметров запроса. Ключи словаря - имена параметров,
значения - описания параметров и указание на обязательность. \- `description`:
Описание назначения и функциональности API запроса.

Этот словарь служит централизованным хранилищем конфигураций для
взаимодействия с различными API, облегчая поддержку и модификацию API запросов
в рамках проекта.

  * ### Super-module

    * `[src](index.html "src")`
  * ### Global variables

    * `all_api_requests`

Generated by [pdoc 0.11.5](https://pdoc3.github.io/pdoc "pdoc: Python API
documentation generator").

