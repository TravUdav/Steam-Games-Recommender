# Module `src.kaggle_downloader`

## Classes

` class SteamGameDataDownloader (dataset_name='fronkongames/steam-games-
dataset',  
output_filename='games.json',  
force_download=False) `

     Expand source code
    
    
    class SteamGameDataDownloader:
        """
        Класс для загрузки набора данных Steam Games с Kaggle.
    
        Обеспечивает загрузку набора данных 'Steam Games Dataset' с платформы Kaggle Hub
        в локальную директорию. Поддерживает принудительную перезагрузку набора данных.
        """
    
        def __init__(self, dataset_name="fronkongames/steam-games-dataset", output_filename="games.json", force_download=False):
            """
            Инициализирует загрузчик набора данных Steam Games.
    
            Аргументы:
                dataset_name (str, optional): Имя набора данных на Kaggle Hub.
                                               По умолчанию "fronkongames/steam-games-dataset".
                output_filename (str, optional): Имя файла для сохранения загруженного набора данных.
                                                 Путь к файлу будет относительно текущей рабочей директории.
                                                 По умолчанию "games.json".
                force_download (bool, optional): Флаг, указывающий, следует ли принудительно загружать набор данных,
                                                 даже если он уже существует локально.
                                                 По умолчанию False.
            """
            self.dataset_name = dataset_name
            self.output_filename = output_filename
            self.force_download = force_download
    
        def download(self):
           """Загружает набор данных Steam Games с Kaggle Hub и возвращает путь к загруженному файлу.
    
           Функция использует библиотеку `kagglehub` для загрузки набора данных,
           определенного в `self.dataset_name`, и сохраняет его под именем `self.output_filename`
           в текущей директории.
    
           Возвращает:
               str: Абсолютный путь к загруженному файлу набора данных.
    
           Выводит в консоль сообщения о начале и завершении загрузки, а также информацию
           о пути к загруженному файлу. В случае возникновения ошибок при загрузке,
           выведет сообщение об ошибке.
    
           Пример вывода в консоль:
               "🔄 Начало загрузки набора данных 'fronkongames/steam-games-dataset' с Kaggle Hub..."
               "💾 Набор данных успешно загружен и сохранен в: /path/to/games.json"
           """
           print(f"🔄 Начало загрузки набора данных '{self.dataset_name}' с Kaggle Hub...")
           try:
               path = kagglehub.dataset_download(self.dataset_name, path=self.output_filename, force_download=self.force_download)
               print(f"💾 Набор данных успешно загружен и сохранен в: {os.path.abspath(path)}")
               return path
           except Exception as e:
               print(f"❌ Ошибка при загрузке набора данных: {e}")
               return None

Класс для загрузки набора данных Steam Games с Kaggle.

Обеспечивает загрузку набора данных 'Steam Games Dataset' с платформы Kaggle
Hub в локальную директорию. Поддерживает принудительную перезагрузку набора
данных.

Инициализирует загрузчик набора данных Steam Games.

Аргументы: dataset_name (str, optional): Имя набора данных на Kaggle Hub. По
умолчанию "fronkongames/steam-games-dataset". output_filename (str, optional):
Имя файла для сохранения загруженного набора данных. Путь к файлу будет
относительно текущей рабочей директории. По умолчанию "games.json".
force_download (bool, optional): Флаг, указывающий, следует ли принудительно
загружать набор данных, даже если он уже существует локально. По умолчанию
False.

### Methods

` def download(self) `

     Expand source code
    
    
    def download(self):
       """Загружает набор данных Steam Games с Kaggle Hub и возвращает путь к загруженному файлу.
    
       Функция использует библиотеку `kagglehub` для загрузки набора данных,
       определенного в `self.dataset_name`, и сохраняет его под именем `self.output_filename`
       в текущей директории.
    
       Возвращает:
           str: Абсолютный путь к загруженному файлу набора данных.
    
       Выводит в консоль сообщения о начале и завершении загрузки, а также информацию
       о пути к загруженному файлу. В случае возникновения ошибок при загрузке,
       выведет сообщение об ошибке.
    
       Пример вывода в консоль:
           "🔄 Начало загрузки набора данных 'fronkongames/steam-games-dataset' с Kaggle Hub..."
           "💾 Набор данных успешно загружен и сохранен в: /path/to/games.json"
       """
       print(f"🔄 Начало загрузки набора данных '{self.dataset_name}' с Kaggle Hub...")
       try:
           path = kagglehub.dataset_download(self.dataset_name, path=self.output_filename, force_download=self.force_download)
           print(f"💾 Набор данных успешно загружен и сохранен в: {os.path.abspath(path)}")
           return path
       except Exception as e:
           print(f"❌ Ошибка при загрузке набора данных: {e}")
           return None

Загружает набор данных Steam Games с Kaggle Hub и возвращает путь к
загруженному файлу.

Функция использует библиотеку `kagglehub` для загрузки набора данных,
определенного в `self.dataset_name`, и сохраняет его под именем
`self.output_filename` в текущей директории.

Возвращает: str: Абсолютный путь к загруженному файлу набора данных.

Выводит в консоль сообщения о начале и завершении загрузки, а также информацию
о пути к загруженному файлу. В случае возникновения ошибок при загрузке,
выведет сообщение об ошибке.

Пример вывода в консоль: "🔄 Начало загрузки набора данных 'fronkongames/steam-
games-dataset' с Kaggle Hub…" "💾 Набор данных успешно загружен и сохранен в:
/path/to/games.json"

  * ### Super-module

    * `[src](index.html "src")`
  * ### Classes

    * #### `SteamGameDataDownloader`

      * `download`

Generated by [pdoc 0.11.5](https://pdoc3.github.io/pdoc "pdoc: Python API
documentation generator").

