import os
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pymorphy2
from langdetect import detect, LangDetectException
from cachetools import cached, LRUCache

_data_cleaner_cache = LRUCache(maxsize=1)

class FileHandler:
    """
    Класс для обработки операций загрузки и сохранения данных из файлов JSON и CSV.

    Предоставляет методы для чтения данных из файлов различных форматов, таких как JSON и CSV,
    в pandas DataFrame, а также для сохранения DataFrame обратно в файлы JSON или CSV.
    """
    def __init__(self):
        """
        Инициализация FileHandler.

        Выводит сообщение в консоль об успешной инициализации FileHandler.
        """
        print("✅ FileHandler инициализирован.")

    def load_data(self, path):
        """
        Загружает данные из файла JSON или CSV в pandas DataFrame.

        Определяет тип файла по расширению и использует соответствующий метод pandas для загрузки данных.
        Поддерживает файлы с расширениями .json, .csv и .txt.

        Аргументы:
            path (str): Путь к файлу, из которого необходимо загрузить данные.

        Возвращает:
            pandas.DataFrame: DataFrame, содержащий загруженные данные.

        Вызывает:
            FileNotFoundError: Если файл по указанному пути не существует.
            ValueError: Если расширение файла не поддерживается.
            Exception: В случае любых других ошибок при чтении файла.

        Выводит в консоль сообщения о начале загрузки, успехе и ошибках.
        """
        print(f"🔄 Загрузка данных из файла: {path}")
        if not os.path.exists(path):
            print(f"❌ Файл не найден: {path}")
            raise FileNotFoundError(f"Файл не найден: {path}")
        try:
            if path.lower().endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
                df = pd.DataFrame(data).T
                print(f"✅ Данные успешно загружены из JSON, форма: {df.shape}")
                return df
            elif path.lower().endswith((".csv", ".txt")):
                df = pd.read_csv(path)
                print(f"✅ Данные успешно загружены из CSV, форма: {df.shape}")
                return df
            else:
                print(f"❌ Неподдерживаемый формат файла: {path}")
                raise ValueError(f"Неподдерживаемый формат файла: {path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки данных из {path}: {e}")
            raise

    def save_data(self, df, path):
        """
        Сохраняет pandas DataFrame в файл JSON или CSV.

        Определяет формат файла по расширению и использует соответствующий метод pandas для сохранения DataFrame.
        Поддерживает файлы с расширениями .json, .csv и .txt.

        Аргументы:
            df (pandas.DataFrame): DataFrame, который необходимо сохранить.
            path (str): Путь к файлу, в который необходимо сохранить данные.

         Вызывает:
            ValueError: Если расширение файла не поддерживается.
            Exception: В случае любых других ошибок при записи файла.

        Выводит в консоль сообщения о начале сохранения, успехе и ошибках.
        """
        print(f"💾 Сохранение данных в файл: {path}")
        try:
            if path.lower().endswith(".json"):
                df.to_json(path)
                print(f"✅ Данные успешно сохранены в JSON, форма: {df.shape}")
            elif path.lower().endswith((".csv", ".txt")):
                df.to_csv(path, index=False)
                print(f"✅ Данные успешно сохранены в CSV, форма: {df.shape}")
            else:
                print(f"❌ Неподдерживаемый формат файла: {path}")
                raise ValueError(f"Неподдерживаемый формат файла: {path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения данных в {path}: {e}")
            raise

class DataCleaner:
    """
    Класс для очистки и предобработки набора данных об играх.

    Предоставляет набор методов для обработки DataFrame, включая удаление столбцов,
    фильтрацию строк по различным критериям, преобразование типов данных,
    объединение и очистку текстовых данных, таких как описания и теги.
    """
    def __init__(self, columns_to_drop=None, min_description_length=30, max_description_length=240, min_tags=3, words_to_remove = ['game', 'world']):
        """
         Инициализация DataCleaner.

         Конфигурирует параметры очистки данных, такие как список удаляемых столбцов,
         минимальная и максимальная длина описаний, минимальное количество тегов и список слов для удаления.
         Также инициализирует лемматизаторы и стоп-слова для английского и русского языков.

        Аргументы:
            columns_to_drop (list, optional): Список имен столбцов, которые будут удалены из DataFrame.
                                             По умолчанию None, используется список столбцов по умолчанию.
            min_description_length (int, optional): Минимальная длина короткого описания игры после очистки.
                                                    Описания короче этого значения будут отфильтрованы. По умолчанию 30.
            max_description_length (int, optional): Максимальная длина короткого описания игры после очистки.
                                                    Описания длиннее этого значения будут отфильтрованы. По умолчанию 240.
            min_tags (int, optional): Минимальное количество тегов, которое должна содержать игра, чтобы остаться в наборе данных.
                                      Игры с меньшим количеством тегов будут отфильтрованы. По умолчанию 3.
            words_to_remove (list, optional): Список слов, которые будут удалены из описаний игр.
                                             Используется для удаления общих и неинформативных слов. По умолчанию ['game', 'world'].

        Выводит сообщение в консоль об успешной инициализации DataCleaner.
        Загружает необходимые ресурсы NLTK (stopwords, wordnet) при первом запуске.
        """
        self.columns_to_drop = columns_to_drop if columns_to_drop else [
            'price', 'dlc_count', 'about_the_game',
            'reviews', 'website', 'support_url',
            'support_email', 'metacritic_score',
            'metacritic_url', 'achievements', 'recommendations',
            'notes', 'full_audio_languages', 'packages',
            'user_score', 'score_rank',
            'screenshots', 'movies',
            'average_playtime_forever', 'average_playtime_2weeks',
            'median_playtime_forever', 'median_playtime_2weeks',
            'peak_ccu'
        ]
        self.min_description_length = min_description_length
        self.max_description_length = max_description_length
        self.min_tags = min_tags
        self.words_to_remove = words_to_remove
        print("✅ DataCleaner инициализирован.")

        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            wordnet_lemmatizer = WordNetLemmatizer()
            wordnet_lemmatizer.lemmatize('cats')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        self.lemmatizer_en = WordNetLemmatizer()
        self.stop_words_en = set(stopwords.words('english'))

        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words_ru = set(stopwords.words('russian'))


    def _drop_unnecessary_columns(self, df):
        """Удаляет заданные столбцы из DataFrame.

        Исключает из DataFrame столбцы, перечисленные в `self.columns_to_drop`.
        Удаление производится на месте (inplace).

        Аргументы:
            df (pandas.DataFrame): DataFrame, из которого нужно удалить столбцы.

        Возвращает:
            pandas.DataFrame: DataFrame с удаленными столбцами.
        """
        columns_to_drop_exist = [col for col in self.columns_to_drop if col in df.columns]
        df.drop(columns=columns_to_drop_exist, inplace=True, errors='ignore')
        return df

    def _filter_rows(self, df):
        """
        Фильтрует строки DataFrame по набору условий.

        Удаляет строки, которые не соответствуют критериям минимального качества данных,
        таким как наличие короткого и подробного описания, имени, изображения, поддерживаемых языков и категорий.

        Аргументы:
            df (pandas.DataFrame): DataFrame для фильтрации.

        Возвращает:
            pandas.DataFrame: Отфильтрованный DataFrame.
        """

        mask_to_remove = (
            ((df['short_description'].isna()) | (df['short_description'] == '')) |
            ((df['detailed_description'].isna()) | (df['detailed_description'] == '')) |
            (df['name'].str.contains('playtest', case=False, na=False)) |
            ((df['header_image'].isna()) | (df['header_image'] == '')) |
            (df['supported_languages'].astype(str) == '[]') |
            (df['categories'].astype(str) == '[]')
        )
        df_filtered = df[~mask_to_remove].copy()
        return df_filtered

    def _filter_name_chars(self, df):
        """
        Фильтрует DataFrame, оставляя только игры с именами, содержащими латинские буквы, цифры и пробелы.

        Удаляет игры, имена которых содержат символы, отличные от латинских букв, цифр и пробелов,
        для обеспечения совместимости и упрощения обработки имен.

        Аргументы:
            df (pandas.DataFrame): DataFrame для фильтрации.

        Возвращает:
            pandas.DataFrame: Отфильтрованный DataFrame.
        """


        def clean_name(text):
             if isinstance(text, str):
                 return re.sub(r'[^a-zA-Z0-9\s]', '', text)
             return text

        df['name'] = df['name'].apply(clean_name)

        def is_valid_name(text):
            if isinstance(text, str):
                return bool(re.fullmatch(r'[a-zA-Z0-9\s]+', text))
            return False

        mask_to_remove = ~df['name'].apply(is_valid_name)
        df_filtered = df[~mask_to_remove].copy()
        return df_filtered


    def _convert_release_date(self, df):
        """
        Преобразует столбец 'release_date' в формат datetime.

        Использует pandas `to_datetime` для преобразования дат выпуска игр в стандартный формат datetime,
        обрабатывая ошибки преобразования и устанавливая некорректные даты в NaT (Not a Time).

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбец 'release_date'.

        Возвращает:
            pandas.DataFrame: DataFrame с преобразованным столбцом 'release_date'.
        """
        def _convert_release_date(self, df):
            if 'release_date' in df.columns:
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            else:
                print("⚠️ Столбец 'release_date' отсутствует в данных. Создан пустой столбец.")
                df['release_date'] = pd.NaT
            return df

    def _convert_bool_columns(self, df):
         """
        Преобразует столбцы 'windows', 'mac', 'linux' в булевый тип данных.

        Приводит значения в столбцах, обозначающих поддержку платформ, к булевому типу,
        чтобы упростить логические операции и уменьшить объем памяти, используемый для хранения этих данных.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбцы для преобразования ('windows', 'mac', 'linux').

        Returns:
            pandas.DataFrame: DataFrame с преобразованными булевыми столбцами.
        """
         if df is None or df.empty:  # Проверка на пустой DF
             print("⚠️ DataFrame пуст. Пропуск преобразования булевых столбцов.")
             return df
        
         initial_shape = df.shape
         bool_mapping = {'true': True, 'false': False}
        
         try:
             for col in ['windows', 'mac', 'linux']:
                 if col in df.columns:
                     df[col] = df[col].astype(str).str.lower().replace({'nan': None})
                     df[col] = df[col].map(bool_mapping).fillna(False).astype(bool)
             return df
         except Exception as e:
             print(f"⚠️ Ошибка в преобразовании столбца {col}: {e}")
             return df  # Возвращаем исходный DF вместо прерывания

    def _extract_owners(self, df):
         """
        Извлекает численное значение из столбца 'estimated_owners'.

        Парсит столбец 'estimated_owners', который содержит диапазоны оценочного количества владельцев,
        и извлекает нижнюю границу диапазона как численное значение, приводя его к целому типу.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбец 'estimated_owners'.

        Возвращает:
            pandas.DataFrame: DataFrame с преобразованным столбцом 'estimated_owners', содержащим целочисленные значения.
        """
         def extract_first_number(owner_range):
            if isinstance(owner_range, str):
                parts = owner_range.split(' ', 1)
                first_part = parts[0].replace(',', '')
                try:
                   return int(first_part)
                except ValueError:
                    return None
            return None

         df['estimated_owners'] = df['estimated_owners'].apply(extract_first_number)
         return df

    def _combine_tags(self, df):
         """
        Объединяет теги из столбцов 'categories', 'genres' и 'tags' в один столбец 'all_tags'.

        Собирает все теги, относящиеся к игре, из разных столбцов и создает единый список тегов для каждой игры,
        чтобы упростить дальнейший анализ и векторизацию тегов.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбцы 'categories', 'genres' и 'tags'.

        Возвращает:
            pandas.DataFrame: DataFrame с добавленным столбцом 'all_tags', содержащим списки объединенных тегов.
        """
         def combine_tags(row):
            all_tags_list = []

            if isinstance(row['categories'], list):
                all_tags_list.extend(row['categories'])

            if isinstance(row['genres'], list):
                all_tags_list.extend(row['genres'])

            if 'tags' in row and isinstance(row['tags'], dict):
                all_tags_list.extend(row['tags'].keys())

            return list(set(all_tags_list))

         df['all_tags'] = df.apply(combine_tags, axis=1)
         return df

    def _replace_empty_values(self, df):
        """
        Заменяет пустые списки и строки в столбцах 'developers' и 'publishers' на None.

        Нормализует значения в столбцах 'developers' и 'publishers', заменяя пустые значения на None,
        чтобы обеспечить единообразие и упростить обработку отсутствующих данных.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбцы 'developers' и 'publishers'.

        Возвращает:
            pandas.DataFrame: DataFrame с замененными пустыми значениями в столбцах 'developers' и 'publishers'.
        """
        def replace_empty_with_none(series):
            def replace_item(item):
                if item == [] or item == [''] or item == [""] or item == "":
                    return None
                return item

            return series.apply(replace_item)

        if 'developers' in df.columns:
            df['developers'] = replace_empty_with_none(df['developers'])
        if 'publishers' in df.columns:
            df['publishers'] = replace_empty_with_none(df['publishers'])
        return df

    def _filter_by_language(self, df):
        """
        Фильтрует DataFrame, оставляя только описания на английском или русском языках.

        Использует `langdetect` для определения языка описаний игр и оставляет только те строки,
        где и короткое, и подробное описания определены как английские ('en') или русские ('ru').

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбцы 'detailed_description' и 'short_description'.

        Возвращает:
            pandas.DataFrame: Отфильтрованный DataFrame, содержащий только описания на английском или русском языках.
        """

        def is_english_or_russian(text):
           if not isinstance(text, str):
                return False
           try:
                lang = detect(text)
                return lang == 'en' or lang == 'ru'
           except LangDetectException:
               return False

        df['detailed_is_en_ru'] = df['detailed_description'].apply(is_english_or_russian)
        df['short_is_en_ru'] = df['short_description'].apply(is_english_or_russian)

        df_filtered = df[df['detailed_is_en_ru'] & df['short_is_en_ru']].copy()
        df_filtered = df_filtered.drop(columns=['detailed_is_en_ru', 'short_is_en_ru'], errors='ignore')
        return df_filtered

    def _clean_and_lemmatize_descriptions(self, df):
        """
        Очищает и лемматизирует текстовые описания игр.

        Применяет очистку текста, включая приведение к нижнему регистру, удаление знаков пунктуации и цифр,
        а также лемматизацию слов для английского и русского языков с использованием NLTK и pymorphy2.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбцы 'detailed_description' и 'short_description'.

        Возвращает:
            pandas.DataFrame: DataFrame с добавленными столбцами '_clean' для очищенных описаний ('detailed_description_clean', 'short_description_clean').
        """

        def clean_and_lemmatize(text, lang='en'):
            if not isinstance(text, str):
                return ""

            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)

            words = text.split()

            if lang == 'ru':
                lemmatized_words = [self.morph.parse(word)[0].normal_form for word in words if word not in self.stop_words_ru]
            else:
                lemmatized_words = [self.lemmatizer_en.lemmatize(word) for word in words if word not in self.stop_words_en]

            return " ".join(lemmatized_words)

        df['detailed_description_clean'] = df['detailed_description'].apply(lambda x: clean_and_lemmatize(x))
        df['short_description_clean'] = df['short_description'].apply(lambda x: clean_and_lemmatize(x))
        return df

    def _remove_specific_words_from_descriptions(self, df):
        """
        Удаляет заданные слова из очищенных описаний игр.

        Исключает слова, перечисленные в `self.words_to_remove`, из очищенных коротких и подробных описаний игр,
        чтобы убрать общие и нерелевантные слова, специфичные для контекста игр.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбцы 'short_description_clean' и 'detailed_description_clean'.

        Возвращает:
            pandas.DataFrame: DataFrame с удаленными специфическими словами из очищенных описаний.
        """

        def remove_words(text):
            if isinstance(text, str):
                words = text.split()
                filtered_words = [word for word in words if word not in self.words_to_remove]
                return ' '.join(filtered_words)
            return text

        df['short_description_clean'] = df['short_description_clean'].apply(remove_words)
        df['detailed_description_clean'] = df['detailed_description_clean'].apply(remove_words)
        return df

    def _filter_description_length(self, df):
        """
        Фильтрует DataFrame по длине очищенных коротких описаний.

        Удаляет строки, где длина очищенного короткого описания не попадает в заданный диапазон
        (`self.min_description_length` и `self.max_description_length`), чтобы обеспечить минимальную и максимальную информативность описаний.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбец 'short_description_clean'.

        Возвращает:
            pandas.DataFrame: Отфильтрованный DataFrame, соответствующий заданным ограничениям по длине описаний.
        """

        if isinstance(data, str):
            df_filtered = df[(df['short_description_clean'].str.len() >= self.min_description_length) & (df['short_description_clean'].str.len() <= self.max_description_length)].copy()
        else:
            df_filtered = df.copy()
        return df_filtered

    def _clean_and_lowercase_tags(self, df):
        """
        Очищает и приводит к нижнему регистру теги в столбце 'all_tags'.

        Удаляет из тегов символы, не являющиеся буквами или цифрами, приводит все теги к нижнему регистру
        и удаляет лишние пробелы, обеспечивая стандартный и очищенный вид тегов.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбец 'all_tags'.

        Возвращает:
            pandas.DataFrame: DataFrame с очищенными и приведенными к нижнему регистру тегами.
        """

        def clean_tags(tags):
            if isinstance(tags, list):
                cleaned_tags = [re.sub(r'[^a-zA-Z0-9\s]', '', tag).lower().strip() for tag in tags]
                return cleaned_tags
            return tags

        df['all_tags'] = df['all_tags'].apply(clean_tags)
        return df

    def _filter_tags_count(self, df):
        """
        Фильтрует DataFrame по минимальному количеству тегов в столбце 'all_tags'.

        Удаляет строки, где количество тегов в списке 'all_tags' меньше, чем заданное значение `self.min_tags`,
        чтобы отфильтровать игры с недостаточным количеством тегов для анализа.

        Аргументы:
            df (pandas.DataFrame): DataFrame, содержащий столбец 'all_tags'.

        Возвращает:
            pandas.DataFrame: Отфильтрованный DataFrame, содержащий только игры с достаточным количеством тегов.
        """

        df_filtered = df[df['all_tags'].apply(lambda x: isinstance(x, list) and len(x) >= self.min_tags)].copy()
        return df_filtered

    def clean_data(self, data):
         """
        Координирует процесс очистки данных DataFrame.

        Вызывает последовательно все методы очистки данных, определенные в классе,
        для обработки DataFrame и подготовки данных к дальнейшему анализу или моделированию.
        Метод может принимать как DataFrame, так и путь к файлу с данными.

        Аргументы:
            data (pandas.DataFrame или str): DataFrame для очистки или путь к файлу (JSON, CSV), который нужно загрузить и очистить.

        Возвращает:
            pandas.DataFrame: Очищенный DataFrame.

        Вызывает:
            ValueError: Если входные данные не являются DataFrame и не строкой (путем к файлу).

        Выводит в консоль сообщения о начале и завершении процесса очистки, а также начальную и конечную форму DataFrame.
        """
         print("🧹 Начинается процесс очистки данных...")
        
         # Загрузка данных
         if isinstance(data, str):
             file_handler = FileHandler()
             df = file_handler.load_data(data)
             apply_description_length_filter = True
         elif isinstance(data, pd.DataFrame):
             df = data.copy()
             apply_description_length_filter = False
         else:
             raise ValueError("❌ Входные данные должны быть pandas DataFrame или путем к файлу.")

         # Отладочная информация
         print(f"📊 Исходная форма: {df.shape}")
         print("Столбцы перед обработкой:", df.columns.tolist())

         # Цепочка обработки с проверками
         processing_steps = [
             self._filter_rows,
             self._combine_tags,
             self._drop_unnecessary_columns,
             self._filter_name_chars,
             self._convert_release_date,
             self._convert_bool_columns,
             self._extract_owners,
             self._replace_empty_values,
             self._filter_by_language,
             self._clean_and_lemmatize_descriptions,
             self._remove_specific_words_from_descriptions,
             lambda x: self._filter_description_length(x) if apply_description_length_filter else x,
             self._clean_and_lowercase_tags,
             self._filter_tags_count
         ]

         for step in processing_steps:
             try:
                 if df is None or df.empty:  # Прекращаем обработку если DF стал пустым
                     print("⚠️ DataFrame пуст. Прекращение обработки.")
                     break
                 df = step(df)
             except Exception as e:
                 print(f"⚠️ Ошибка на шаге {step.__name__}: {e}")
                 continue

         print(f"✅ Процесс очистки завершен. Итоговая форма: {df.shape if df is not None else 0}")
         return df if df is not None else pd.DataFrame()


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'steam_games.json')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'steam_games_data_cleaned_test.json')

    data = {
        'name': [
            'Game 1', 'Game 2', 'Game 3', 'Game 4', 'Game 5', 'Game 6', 'Game 7', 'Game 8', 'Game 9', 'Game 10',
            'Game 11', 'Game 12', 'Game 13', 'Game 14', 'Game 15', 'Game 16', 'Game 17', 'Game 18', 'Game 19', 'Game 20',
            'Тестовая игра на русском', '12345', 'Test Playtest Game', 'Test game with numbers 123', 'Another Playtest Game'
        ],
        'short_description': [
            'This is a short description for game one, it is longer than before, testing here',
            'Another short description, this one is also quite long for game two and more test',
            'A short but meaningful description for game three, just a bit long with a bit extra',
            'Short description number four, is also around 30 characters in length, testing here',
            'Description for game five, this is supposed to be a bit longer than usual for this test',
            'Game six has a description that is long enough for our needs, test here, some more words',
            'Another very short description for game seven, should be long enough here, more here',
            'Description for game eight, lets make this at least thirty char long now for the test',
            'Game nine has a description, should be at least thirty characters long test for fun',
            'A short description for game ten, test is still around thirty in length, more here',
            'Game eleven with description, this one should be around thirty char long, test here',
            'Description for game twelve, making this more than thirty char in length, more text',
            'A short one for game thirteen, trying to make this more than 30 char long, more test',
            'Game fourteen short, lets make sure that this one is more than 30 char long with test',
            'Description for game fifteen, testing with more than 30 characters here now for tests',
            'A short description for game sixteen, this is more than 30 characters long now tests',
            'Game seventeen short description, let us make sure is more than 30 char long more tests',
            'Description for game eighteen, making this more than 30 char in length here and test',
            'Another short description nineteen, testing for more than 30 chars now for extra text',
            'A short description for game twenty, this is a long 30 character test for fun here',
            'Короткое описание тестовой игры', 'short 21', 'Short description of test playtest', 'short description 22 for test', 'Another short description for game'
        ],
        'detailed_description': [
            'This is a detailed description for game one, it is longer than before and has more info here',
            'Another detailed description for game two, this one is also quite long with more content now',
            'A detailed but meaningful description for game three, just a bit long and has details here',
            'Detailed description number four, is also around 30 characters in length and a bit more info',
            'Description for game five, this is supposed to be a bit longer than usual, testing now details',
            'Game six has a description that is long enough for our needs, test here, should be more info',
            'Another very short description for game seven, should be long enough here, with extra details',
            'Description for game eight, lets make this at least thirty char long now and more content test',
            'Game nine has a description, should be at least thirty characters long test now here and more',
            'A short description for game ten, test is still around thirty in length and some extras info here',
            'Game eleven with description, this one should be around thirty char long, test extras here more',
            'Description for game twelve, making this more than thirty char in length, test details now here',
            'A short one for game thirteen, trying to make this more than 30 char long and more here and info',
            'Game fourteen short, lets make sure that this one is more than 30 char long and details for game',
            'Description for game fifteen, testing with more than 30 characters here now with detail info here',
            'A short description for game sixteen, this is more than 30 characters long now details for game',
            'Game seventeen short description, let us make sure is more than 30 char long more now with info',
            'Description for game eighteen, making this more than 30 char in length here and extras here for game',
            'Another short description nineteen, testing for more than 30 chars now with more info here for game',
            'A short description for game twenty, this is a long 30 character test with more details for game',
            'Детальное описание тестовой игры', 'detailed 21', 'Detailed description of test playtest game', 'detailed description with number 123 game test', 'Another detailed description of test game'
        ],
        'header_image': [
            'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10',
            'image11', 'image12', 'image13', 'image14', 'image15', 'image16', 'image17', 'image18', 'image19', 'image20',
            'image21', 'image22','image23', 'image24', 'image25'
        ],
        'supported_languages': [
            ['en'], ['en', 'ru'], ['ru'], ['en', 'fr'], ['de'], ['en', 'es'], ['pt'], ['it'], ['ja'], ['ko'],
            ['zh'], ['en', 'de'], ['fr', 'es'], ['pt', 'it'], ['ja', 'ko'], ['en'], ['ru'], ['en', 'fr'], ['de'], ['en', 'es'],
            ['zh'],['en'], ['en', 'ru'],['ru'],['en']
        ],
        'categories': [
            ['Action', 'Adventure', 'Indie'], ['Adventure', 'Indie', 'RPG'], ['Strategy', 'Simulation', 'Sports'], ['RPG', 'Racing', 'Puzzle'], ['Simulation', 'Casual', 'MMO'],
            ['Action', 'Indie', 'Puzzle'], ['Adventure', 'Strategy', 'Racing'], ['Strategy', 'RPG', 'Casual'], ['Simulation', 'Sports', 'MMO'], ['Racing', 'Puzzle', 'Action'],
            ['Action', 'Indie', 'Adventure'], ['Adventure', 'Strategy', 'RPG'], ['Strategy', 'Simulation', 'Sports'], ['Simulation', 'Sports', 'MMO'], ['Racing', 'Puzzle', 'Casual'],
            ['MMO', 'Puzzle', 'Action'], ['Action', 'RPG', 'Simulation'], ['Adventure', 'Sports', 'Racing'], ['Strategy', 'Puzzle', 'Casual'], ['RPG', 'MMO', 'Action'],
            ['Action', 'Adventure'], ['Strategy', 'RPG'],['Action', 'Indie'],['Strategy','Sports'],['Action','Simulation']
        ],
        'tags': [
            {'Action': '1', 'Adventure': '2', 'Indie': '3'}, {'Indie': '2', 'Adventure': '3', 'RPG': '4'}, {'Strategy': '4', 'Simulation': '5', 'Sports': '6'},
            {'RPG': '5', 'Racing': '6', 'Puzzle': '7'}, {'Simulation': '6', 'Casual': '7', 'MMO': '8'}, {'Action': '7', 'Indie': '8', 'Puzzle': '9'},
            {'Adventure': '8', 'Strategy': '9', 'Racing': '10'}, {'Strategy': '9', 'RPG': '10', 'Casual': '11'}, {'Simulation': '10', 'Sports': '11', 'MMO': '12'},
            {'Racing': '11', 'Puzzle': '12', 'Action': '13'}, {'Action': '12', 'Indie': '13', 'Adventure': '14'}, {'Adventure': '13', 'Strategy': '14', 'RPG': '15'},
            {'Strategy': '14', 'Simulation': '15', 'Sports': '16'}, {'Simulation': '15', 'Sports': '16', 'MMO': '17'}, {'Racing': '16', 'Puzzle': '17', 'Casual': '18'},
            {'MMO': '17', 'Puzzle': '18', 'Action': '19'}, {'Action': '18', 'RPG': '19', 'Simulation': '20'}, {'Adventure': '19', 'Sports': '20', 'Racing': '21'},
            {'Strategy': '20', 'Puzzle': '21', 'Casual': '22'}, {'RPG': '21', 'MMO': '22', 'Action': '23'},
            {'Action': '1', 'Adventure': '2'},{'Strategy': '4', 'RPG': '5'},{'Action': '1', 'Indie': '2'},{'Strategy': '4', 'Sports': '5'},{'Action': '1', 'Simulation': '2'}
        ],
        'genres': [
            ['Action', 'Adventure', 'Indie'], ['Adventure', 'Indie', 'RPG'], ['Strategy', 'Simulation', 'Sports'], ['RPG', 'Racing', 'Puzzle'], ['Simulation', 'Casual', 'MMO'],
            ['Action', 'Indie', 'Puzzle'], ['Adventure', 'Strategy', 'Racing'], ['Strategy', 'RPG', 'Casual'], ['Simulation', 'Sports', 'MMO'], ['Racing', 'Puzzle', 'Action'],
            ['Action', 'Indie', 'Adventure'], ['Adventure', 'Strategy', 'RPG'], ['Strategy', 'Simulation', 'Sports'], ['Simulation', 'Sports', 'MMO'], ['Racing', 'Puzzle', 'Casual'],
            ['MMO', 'Puzzle', 'Action'], ['Action', 'RPG', 'Simulation'], ['Adventure', 'Sports', 'Racing'], ['Strategy', 'Puzzle', 'Casual'], ['RPG', 'MMO', 'Action'],
            ['Action', 'Adventure'], ['Strategy', 'RPG'],['Action', 'Indie'],['Strategy','Sports'],['Action','Simulation']
        ],
        'release_date': [
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01',
            '2023-11-01', '2023-12-01', '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01', '2024-07-01', '2024-08-01',
            '2024-01-01', '2024-01-01','2024-01-01', '2024-01-01', '2024-01-01'
        ],
        'windows': [
            'true', 'true', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
            'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
            'true', 'false', 'true', 'true', 'false'
        ],
        'mac': [
            'true', 'false', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
            'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
            'true', 'false', 'false', 'false', 'true'
        ],
        'linux': [
            'false', 'false', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
            'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
            'false', 'false', 'false', 'false', 'false'
        ],
        'estimated_owners': [
            '0-20000', '20000-50000', '50000-100000', '100000-200000', '200000-500000',
            '500000-1000000', '1000000-2000000', '2000000-5000000', '5000000-10000000', '10000000-20000000',
            '0-20000', '20000-50000', '50000-100000', '100000-200000', '200000-500000',
            '500000-1000000', '1000000-2000000', '2000000-5000000', '5000000-10000000', '10000000-20000000',
            '0-20000', '50000-100000','10000-20000', '200000-500000', '10000000-20000000'
        ],
        'developers': [
            ['Dev 1'], ['Dev 2'], [], ['Dev 4'], ['Dev 5'], ['Dev 6'], ['Dev 7'], ['Dev 8'], ['Dev 9'], ['Dev 10'],
            ['Dev 11'], ['Dev 12'], [], ['Dev 14'], ['Dev 15'], ['Dev 16'], ['Dev 17'], ['Dev 18'], ['Dev 19'], ['Dev 20'],
             ['Dev 21'], ['Dev 22'], ['Dev 23'],['Dev 24'],['Dev 25']
        ],
        'publishers': [
            ['Pub 1'], ['Pub 2'], [], ['Pub 4'], ['Pub 5'], ['Pub 6'], ['Pub 7'], ['Pub 8'], ['Pub 9'], ['Pub 10'],
            ['Pub 11'], ['Pub 12'], [], ['Pub 14'], ['Pub 15'], ['Pub 16'], ['Pub 17'], ['Pub 18'], ['Pub 19'], ['Pub 20'],
            ['Pub 21'], ['Pub 22'], ['Pub 23'], ['Pub 24'], ['Pub 25']
        ]
    }
    test_df = pd.DataFrame(data)

    data_cleaner = DataCleaner()

    print("---------------------")
    print("🧪 Тестирование с DataFrame:")
    try:
        cleaned_df = data_cleaner.clean_data(test_df)
        print("✅ Очищенный DataFrame:")
        print(cleaned_df.head())
    except Exception as e:
          print(f"❌ Ошибка при обработке DataFrame: {e}")

    print("---------------------")

    print("📄 Тестирование с JSON файлом:")
    try:
        cleaned_df_from_file = data_cleaner.clean_data(raw_data_path)
        print("✅ Очищенный DataFrame из JSON файла:")
        print(cleaned_df_from_file.head())

        file_handler = FileHandler()
        file_handler.save_data(cleaned_df_from_file, processed_data_path)
        print(f"💾 Очищенные данные сохранены в: {processed_data_path}")

    except FileNotFoundError as e:
         print(f"❌ Ошибка: Файл не найден: {e}")
    except Exception as e:
         print(f"❌ Ошибка при обработке JSON файла: {e}")

    print("---------------------")
    print("✅ Тесты завершены.")
