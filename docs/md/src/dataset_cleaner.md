# Module `src.dataset_cleaner`

## Classes

` class DataCleaner (columns_to_drop=None,  
min_description_length=30,  
max_description_length=240,  
min_tags=3,  
words_to_remove=['game', 'world']) `

     Expand source code
    
    
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
            initial_shape = df.shape
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
            initial_shape = df.shape
    
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
            initial_shape = df.shape
    
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
            initial_shape = df.shape
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
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
             initial_shape = df.shape
             bool_mapping = {'true': True, 'false': False}
             for col in ['windows', 'mac', 'linux']:
                  if col in df.columns:
                    df[col] = df[col].astype(str).str.lower().replace({'nan': None})
                    df[col] = df[col].map(bool_mapping).fillna(False).astype(bool)
             return df
    
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
             initial_shape = df.shape
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
             initial_shape = df.shape
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
            initial_shape = df.shape
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
            initial_shape = df.shape
    
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
            initial_shape = df.shape
    
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
            initial_shape = df.shape
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
            initial_shape = df.shape
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
            initial_shape = df.shape
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
            initial_shape = df.shape
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
             if isinstance(data, str):
                  file_handler = FileHandler()
                  df = file_handler.load_data(data)
                  apply_description_length_filter = True
             elif isinstance(data, pd.DataFrame):
                 df = data.copy()
                 apply_description_length_filter = False
             else:
                 raise ValueError("❌ Входные данные должны быть pandas DataFrame или путем к файлу.")
    
             initial_shape = df.shape
             print(f"📊 Исходная форма DataFrame перед очисткой: {initial_shape}")
    
             df = self._filter_rows(df)
             df = self._combine_tags(df)
             df = self._drop_unnecessary_columns(df)
             df = self._filter_name_chars(df)
             df = self._convert_release_date(df)
             df = self._convert_bool_columns(df)
             df = self._extract_owners(df)
             df = self._replace_empty_values(df)
             df = self._filter_by_language(df)
             df = self._clean_and_lemmatize_descriptions(df)
             df = self._remove_specific_words_from_descriptions(df)
             if apply_description_length_filter:
                df = self._filter_description_length(df)
             df = self._clean_and_lowercase_tags(df)
             df = self._filter_tags_count(df)
    
             print(f"✅ Процесс очистки данных завершен. Итоговая форма: {df.shape}")
             return df

Класс для очистки и предобработки набора данных об играх.

Предоставляет набор методов для обработки DataFrame, включая удаление
столбцов, фильтрацию строк по различным критериям, преобразование типов
данных, объединение и очистку текстовых данных, таких как описания и теги.

Инициализация DataCleaner.

Конфигурирует параметры очистки данных, такие как список удаляемых столбцов,
минимальная и максимальная длина описаний, минимальное количество тегов и
список слов для удаления. Также инициализирует лемматизаторы и стоп-слова для
английского и русского языков.

Аргументы: columns_to_drop (list, optional): Список имен столбцов, которые
будут удалены из DataFrame. По умолчанию None, используется список столбцов по
умолчанию. min_description_length (int, optional): Минимальная длина короткого
описания игры после очистки. Описания короче этого значения будут
отфильтрованы. По умолчанию 30. max_description_length (int, optional):
Максимальная длина короткого описания игры после очистки. Описания длиннее
этого значения будут отфильтрованы. По умолчанию 240. min_tags (int,
optional): Минимальное количество тегов, которое должна содержать игра, чтобы
остаться в наборе данных. Игры с меньшим количеством тегов будут
отфильтрованы. По умолчанию 3. words_to_remove (list, optional): Список слов,
которые будут удалены из описаний игр. Используется для удаления общих и
неинформативных слов. По умолчанию ['game', 'world'].

Выводит сообщение в консоль об успешной инициализации DataCleaner. Загружает
необходимые ресурсы NLTK (stopwords, wordnet) при первом запуске.

### Methods

` def clean_data(self, data) `

     Expand source code
    
    
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
         if isinstance(data, str):
              file_handler = FileHandler()
              df = file_handler.load_data(data)
              apply_description_length_filter = True
         elif isinstance(data, pd.DataFrame):
             df = data.copy()
             apply_description_length_filter = False
         else:
             raise ValueError("❌ Входные данные должны быть pandas DataFrame или путем к файлу.")
    
         initial_shape = df.shape
         print(f"📊 Исходная форма DataFrame перед очисткой: {initial_shape}")
    
         df = self._filter_rows(df)
         df = self._combine_tags(df)
         df = self._drop_unnecessary_columns(df)
         df = self._filter_name_chars(df)
         df = self._convert_release_date(df)
         df = self._convert_bool_columns(df)
         df = self._extract_owners(df)
         df = self._replace_empty_values(df)
         df = self._filter_by_language(df)
         df = self._clean_and_lemmatize_descriptions(df)
         df = self._remove_specific_words_from_descriptions(df)
         if apply_description_length_filter:
            df = self._filter_description_length(df)
         df = self._clean_and_lowercase_tags(df)
         df = self._filter_tags_count(df)
    
         print(f"✅ Процесс очистки данных завершен. Итоговая форма: {df.shape}")
         return df

Координирует процесс очистки данных DataFrame.

Вызывает последовательно все методы очистки данных, определенные в классе, для
обработки DataFrame и подготовки данных к дальнейшему анализу или
моделированию. Метод может принимать как DataFrame, так и путь к файлу с
данными.

Аргументы: data (pandas.DataFrame или str): DataFrame для очистки или путь к
файлу (JSON, CSV), который нужно загрузить и очистить.

Возвращает: pandas.DataFrame: Очищенный DataFrame.

Вызывает: ValueError: Если входные данные не являются DataFrame и не строкой
(путем к файлу).

Выводит в консоль сообщения о начале и завершении процесса очистки, а также
начальную и конечную форму DataFrame.

` class FileHandler `

     Expand source code
    
    
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

Класс для обработки операций загрузки и сохранения данных из файлов JSON и
CSV.

Предоставляет методы для чтения данных из файлов различных форматов, таких как
JSON и CSV, в pandas DataFrame, а также для сохранения DataFrame обратно в
файлы JSON или CSV.

Инициализация FileHandler.

Выводит сообщение в консоль об успешной инициализации FileHandler.

### Methods

` def load_data(self, path) `

     Expand source code
    
    
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

Загружает данные из файла JSON или CSV в pandas DataFrame.

Определяет тип файла по расширению и использует соответствующий метод pandas
для загрузки данных. Поддерживает файлы с расширениями .json, .csv и .txt.

Аргументы: path (str): Путь к файлу, из которого необходимо загрузить данные.

Возвращает: pandas.DataFrame: DataFrame, содержащий загруженные данные.

Вызывает: FileNotFoundError: Если файл по указанному пути не существует.
ValueError: Если расширение файла не поддерживается. Exception: В случае любых
других ошибок при чтении файла.

Выводит в консоль сообщения о начале загрузки, успехе и ошибках.

` def save_data(self, df, path) `

     Expand source code
    
    
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

Сохраняет pandas DataFrame в файл JSON или CSV.

Определяет формат файла по расширению и использует соответствующий метод
pandas для сохранения DataFrame. Поддерживает файлы с расширениями .json, .csv
и .txt.

Аргументы: df (pandas.DataFrame): DataFrame, который необходимо сохранить.
path (str): Путь к файлу, в который необходимо сохранить данные.

Вызывает: ValueError: Если расширение файла не поддерживается. Exception: В
случае любых других ошибок при записи файла.

Выводит в консоль сообщения о начале сохранения, успехе и ошибках.

  * ### Super-module

    * `[src](index.html "src")`
  * ### Classes

    * #### `DataCleaner`

      * `clean_data`
    * #### `FileHandler`

      * `load_data`
      * `save_data`

Generated by [pdoc 0.11.5](https://pdoc3.github.io/pdoc "pdoc: Python API
documentation generator").

