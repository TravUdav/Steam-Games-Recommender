# Module `src.vectorizer`

## Functions

` def calculate_intra_topic_diversity(model, feature_names, num_top_words=10)
`

     Expand source code
    
    
    def calculate_intra_topic_diversity(model, feature_names, num_top_words=10):
        """Вычисляет разнообразие слов внутри каждой темы, используя энтропию распределения слов.
    
        Аргументы:
            model: Обученная тематическая модель (NMF или LDA).
            feature_names (list): Список названий признаков (слов) из TF-IDF векторизатора.
            num_top_words (int, optional): Количество топ-слов, учитываемых для расчета энтропии. По умолчанию 10.
    
        Возвращает:
            float: Средняя энтропия по всем темам, представляющая внутритопиковое разнообразие. Возвращает -1 в случае ошибки.
        """
        if not hasattr(model, 'components_'):
            print("⚠️ Модель не имеет атрибута components_.")
            return -1
    
        topic_vectors = model.components_
        if isinstance(topic_vectors, cp.ndarray):
            topic_vectors_np = topic_vectors.get()
        else:
            topic_vectors_np = topic_vectors
        num_topics = topic_vectors_np.shape[0]
    
        if num_topics == 0:
            print("⚠️ Нет тем для расчета разнообразия.")
            return -1
    
        topic_entropies = []
        for topic in topic_vectors_np:
            top_word_indices = np.argsort(topic)[::-1][:num_top_words]
            top_word_probabilities = topic[top_word_indices]
            normalized_probabilities = top_word_probabilities / np.sum(top_word_probabilities)
            topic_entropy = entropy(normalized_probabilities, base=2)
            topic_entropies.append(topic_entropy)
    
        return np.mean(topic_entropies) if topic_entropies else -1

Вычисляет разнообразие слов внутри каждой темы, используя энтропию
распределения слов.

Аргументы: model: Обученная тематическая модель (NMF или LDA). feature_names
(list): Список названий признаков (слов) из TF-IDF векторизатора.
num_top_words (int, optional): Количество топ-слов, учитываемых для расчета
энтропии. По умолчанию 10.

Возвращает: float: Средняя энтропия по всем темам, представляющая
внутритопиковое разнообразие. Возвращает -1 в случае ошибки.

` def calculate_topic_coherence(model, vectorizer, texts) `

     Expand source code
    
    
    def calculate_topic_coherence(model, vectorizer, texts):
        """Вычисляет когерентность темы модели.
    
        Аргументы:
            model: Обученная тематическая модель (NMF или LDA).
            vectorizer: Обученный TF-IDF векторизатор.
            texts (list): Список текстов, использованных для обучения модели.
    
        Возвращает:
            float: Значение когерентности темы. Возвращает -999 в случае ошибки.
        """
        try:
            feature_names_cuml = vectorizer.get_feature_names()
    
            if isinstance(feature_names_cuml, cudf.core.series.Series):
                feature_names = feature_names_cuml.to_pandas().tolist()
            elif not isinstance(feature_names_cuml, list):
                return -999
    
            if hasattr(model, 'components_') and feature_names is not None:
                topic_vectors = model.components_
                if isinstance(topic_vectors, cp.ndarray):
                    topic_vectors_np = topic_vectors.get()
                else:
                    topic_vectors_np = topic_vectors
    
                top_words_idx = topic_vectors_np.argsort()[:, ::-1]
                top_words = [[feature_names[i] for i in topic_word_idx[:10]] for topic_word_idx in top_words_idx]
    
                dictionary = Dictionary([text.split() for text in texts])
                tokenized_texts = [text.split() for text in texts]
    
                cm = CoherenceModel(topics=top_words, texts=tokenized_texts, dictionary=dictionary, coherence='u_mass')
    
                coherence_score = cm.get_coherence()
                return coherence_score
            else:
                return -999
        except Exception:
            return -999

Вычисляет когерентность темы модели.

Аргументы: model: Обученная тематическая модель (NMF или LDA). vectorizer:
Обученный TF-IDF векторизатор. texts (list): Список текстов, использованных
для обучения модели.

Возвращает: float: Значение когерентности темы. Возвращает -999 в случае
ошибки.

` def calculate_topic_diversity(model) `

     Expand source code
    
    
    def calculate_topic_diversity(model):
        """Вычисляет разнообразие тем на основе косинусного расстояния между векторами тем.
    
        Аргументы:
            model: Обученная тематическая модель (NMF или LDA).
    
        Возвращает:
            float: Значение разнообразия тем. Возвращает -1 в случае ошибки или если количество тем меньше 2.
        """
        if not hasattr(model, 'components_'):
            print("⚠️ Модель не имеет атрибута components_.")
            return -1
    
        topic_vectors = model.components_
        if isinstance(topic_vectors, cp.ndarray):
            topic_vectors_np = topic_vectors.get()
        else:
            topic_vectors_np = topic_vectors
    
        if topic_vectors_np.shape[0] < 2:
            print("⚠️ Менее двух тем. Невозможно вычислить разнообразие.")
            return -1
    
        num_topics = topic_vectors_np.shape[0]
        total_similarity = 0
        num_pairs = 0
    
        for i in range(num_topics):
          for j in range(i+1, num_topics):
             similarity = cosine_similarity(topic_vectors_np[i].reshape(1, -1), topic_vectors_np[j].reshape(1, -1))[0][0]
             total_similarity += similarity
             num_pairs += 1
    
        if num_pairs == 0:
          return -1
    
        average_similarity = total_similarity / num_pairs
        average_distance = 1 - average_similarity
        return average_distance

Вычисляет разнообразие тем на основе косинусного расстояния между векторами
тем.

Аргументы: model: Обученная тематическая модель (NMF или LDA).

Возвращает: float: Значение разнообразия тем. Возвращает -1 в случае ошибки
или если количество тем меньше 2.

` def clean_text(text) `

     Expand source code
    
    
    def clean_text(text):
        """Выполняет предобработку текста: удаление знаков пунктуации, приведение к нижнему регистру, лемматизация и удаление стоп-слов.
    
        Аргументы:
            text (str): Исходный текст для обработки.
    
        Возвращает:
            str: Очищенный и обработанный текст в виде строки.
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

Выполняет предобработку текста: удаление знаков пунктуации, приведение к
нижнему регистру, лемматизация и удаление стоп-слов.

Аргументы: text (str): Исходный текст для обработки.

Возвращает: str: Очищенный и обработанный текст в виде строки.

` def display_topics(model, feature_names, num_top_words=10) `

     Expand source code
    
    
    def display_topics(model, feature_names, num_top_words=10):
        """Выводит наиболее значимые слова для каждой темы.
    
        Аргументы:
            model: Обученная тематическая модель (NMF или LDA).
            feature_names (list): Список названий признаков (слов) из TF-IDF векторизатора.
            num_top_words (int, optional): Количество топ-слов для отображения для каждой темы. По умолчанию 10.
        """
        for topic_idx, topic in enumerate(model.components_):
            print(f"   Тема #{topic_idx}:", end=' ')
            top_word_indices = topic.argsort()[::-1][:num_top_words]
            top_words = [feature_names[i] for i in top_word_indices]
            print(" ".join(top_words))
        print()

Выводит наиболее значимые слова для каждой темы.

Аргументы: model: Обученная тематическая модель (NMF или LDA). feature_names
(list): Список названий признаков (слов) из TF-IDF векторизатора.
num_top_words (int, optional): Количество топ-слов для отображения для каждой
темы. По умолчанию 10.

` def display_topics_with_diversity(model, feature_names, num_top_words=25,
num_display_words=10) `

     Expand source code
    
    
    def display_topics_with_diversity(model, feature_names, num_top_words=25, num_display_words = 10):
        """Выводит наиболее значимые слова для каждой темы и их энтропию.
    
        Аргументы:
            model: Обученная тематическая модель (NMF или LDA).
            feature_names (list): Список названий признаков (слов) из TF-IDF векторизатора.
            num_top_words (int, optional): Количество топ-слов, рассматриваемых для расчета энтропии. По умолчанию 25.
            num_display_words (int, optional): Количество топ-слов для отображения для каждой темы. По умолчанию 10.
        """
        if not hasattr(model, 'components_'):
            print("⚠️ Модель не имеет атрибута components_.")
            return
    
        topic_vectors = model.components_
        if isinstance(topic_vectors, cp.ndarray):
            topic_vectors_np = topic_vectors.get()
        else:
            topic_vectors_np = topic_vectors
    
        for topic_idx, topic in enumerate(topic_vectors_np):
            print(f"   Тема #{topic_idx}. ", end=' ')
            top_word_indices = np.argsort(topic)[::-1][:num_top_words]
            top_words = [feature_names[i] for i in top_word_indices]
            print(f"Топ-{num_display_words} слов: {' '.join(top_words[:num_display_words])}")
            normalized_probabilities = topic[top_word_indices] / np.sum(topic[top_word_indices])
            topic_entropy = entropy(normalized_probabilities, base=2)
            print(f"   Энтропия темы: {topic_entropy:.4f}")
        print()

Выводит наиболее значимые слова для каждой темы и их энтропию.

Аргументы: model: Обученная тематическая модель (NMF или LDA). feature_names
(list): Список названий признаков (слов) из TF-IDF векторизатора.
num_top_words (int, optional): Количество топ-слов, рассматриваемых для
расчета энтропии. По умолчанию 25. num_display_words (int, optional):
Количество топ-слов для отображения для каждой темы. По умолчанию 10.

` def reduce_dataset(df, percentage=0.1) `

     Expand source code
    
    
    def reduce_dataset(df, percentage=0.1):
        """Уменьшает размер DataFrame до указанной доли, отсортированной по убыванию 'estimated_owners'.
    
        Аргументы:
            df (pd.DataFrame): Исходный DataFrame.
            percentage (float): Доля DataFrame, которую нужно оставить (значение от 0 до 1).
    
        Возвращает:
            pd.DataFrame: Уменьшенный DataFrame.
    
        Вызывает ValueError, если percentage не находится в диапазоне [0, 1].
        """
        if not 0 <= percentage <= 1:
            raise ValueError("❌ Процент должен быть в диапазоне от 0 до 1")
    
        print(f"📉 Уменьшение датасета до {percentage * 100}%...")
        df_sorted = df.sort_values(by='estimated_owners', ascending=False)
        num_rows = int(len(df_sorted) * percentage)
        reduced_df = df_sorted.head(num_rows)
        print(f"✅ Датасет уменьшен до {len(reduced_df)} строк.")
        return reduced_df

Уменьшает размер DataFrame до указанной доли, отсортированной по убыванию
'estimated_owners'.

Аргументы: df (pd.DataFrame): Исходный DataFrame. percentage (float): Доля
DataFrame, которую нужно оставить (значение от 0 до 1).

Возвращает: pd.DataFrame: Уменьшенный DataFrame.

Вызывает ValueError, если percentage не находится в диапазоне [0, 1].

` def vectorize_descriptions(df, nmf_params=None, lda_params=None,
vectorizer_cuml=None) `

     Expand source code
    
    
    def vectorize_descriptions(df, nmf_params=None, lda_params=None, vectorizer_cuml=None):
        """Векторизует описания игр, используя TF-IDF и NMF или LDA для тематического моделирования.
    
        Аргументы:
            df (pd.DataFrame): DataFrame, содержащий столбец 'short_description_clean' с очищенными описаниями.
            nmf_params (dict, optional): Параметры для NMF. Если указаны, используется NMF. По умолчанию None.
            lda_params (dict, optional): Параметры для LDA. Если указаны, используется LDA. По умолчанию None.
            vectorizer_cuml (CumlTfidfVectorizer, optional): Обученный CumlTfidfVectorizer. По умолчанию None.
    
        Возвращает:
            tuple: Кортеж, содержащий:
                - np.ndarray: Векторизованные описания (тематические векторы).
                - NMF или LatentDirichletAllocation: Обученная модель NMF или LDA.
    
        Вызывает ValueError, если не предоставлен обученный CumlTfidfVectorizer или не указаны параметры nmf_params или lda_params.
        """
        if vectorizer_cuml is None:
            raise ValueError("❌ Необходимо предоставить обученный CumlTfidfVectorizer.")
        desc_vectorized_cuml = vectorizer_cuml.transform(df['short_description_clean'])
        desc_vectorized_cuml_cpu = desc_vectorized_cuml.get()
        data = cp.asnumpy(desc_vectorized_cuml_cpu.data)
        indices = cp.asnumpy(desc_vectorized_cuml_cpu.indices)
        indptr = cp.asnumpy(desc_vectorized_cuml_cpu.indptr)
        shape = desc_vectorized_cuml_cpu.shape
        desc_vectorized_cpu = csr_matrix((data, indices, indptr), shape=shape)
    
        if nmf_params:
            nmf = NMF(**nmf_params)
            nmf_vectorized = nmf.fit_transform(desc_vectorized_cpu)
            return nmf_vectorized, nmf
        elif lda_params:
            lda = LatentDirichletAllocation(**lda_params)
            lda_vectorized = lda.fit_transform(desc_vectorized_cpu)
            return lda_vectorized, lda
        else:
            raise ValueError("❌ Необходимо указать nmf_params или lda_params")

Векторизует описания игр, используя TF-IDF и NMF или LDA для тематического
моделирования.

Аргументы: df (pd.DataFrame): DataFrame, содержащий столбец
'short_description_clean' с очищенными описаниями. nmf_params (dict,
optional): Параметры для NMF. Если указаны, используется NMF. По умолчанию
None. lda_params (dict, optional): Параметры для LDA. Если указаны,
используется LDA. По умолчанию None. vectorizer_cuml (CumlTfidfVectorizer,
optional): Обученный CumlTfidfVectorizer. По умолчанию None.

Возвращает: tuple: Кортеж, содержащий: \- np.ndarray: Векторизованные описания
(тематические векторы). \- NMF или LatentDirichletAllocation: Обученная модель
NMF или LDA.

Вызывает ValueError, если не предоставлен обученный CumlTfidfVectorizer или не
указаны параметры nmf_params или lda_params.

` def vectorize_owners(df, method='log_scale', scaler=None) `

     Expand source code
    
    
    def vectorize_owners(df, method='log_scale', scaler=None):
        """Векторизует данные о владельцах игр, используя логарифмическое масштабирование или стандартное масштабирование.
    
        Аргументы:
            df (pd.DataFrame): DataFrame, содержащий столбец 'estimated_owners'.
            method (str, optional): Метод векторизации: 'log_scale' или 'standard'. По умолчанию 'log_scale'.
            scaler (CumlMinMaxScaler, optional): Обученный scaler для масштабирования данных. По умолчанию None.
    
        Возвращает:
            np.ndarray: Векторизованные данные о владельцах игр.
    
        Вызывает ValueError, если указан недопустимый метод векторизации владельцев.
        """
        owners = df['estimated_owners'].values.reshape(-1, 1)
        owners = np.array(owners, dtype=float)
        owners = np.nan_to_num(owners, nan=0)
        if method == 'log_scale':
            owners = np.log1p(owners)
            if scaler is not None:
               owners = scaler.transform(owners)
            owners_weighted = owners * (1 + (owners * 2))
            return owners_weighted
        elif method == 'standard':
            if scaler is not None:
               owners = scaler.transform(owners)
            return owners
        else:
            raise ValueError("❌ Недопустимый метод векторизации владельцев.")

Векторизует данные о владельцах игр, используя логарифмическое масштабирование
или стандартное масштабирование.

Аргументы: df (pd.DataFrame): DataFrame, содержащий столбец
'estimated_owners'. method (str, optional): Метод векторизации: 'log_scale'
или 'standard'. По умолчанию 'log_scale'. scaler (CumlMinMaxScaler, optional):
Обученный scaler для масштабирования данных. По умолчанию None.

Возвращает: np.ndarray: Векторизованные данные о владельцах игр.

Вызывает ValueError, если указан недопустимый метод векторизации владельцев.

` def vectorize_tags(df, multilabel_params=None) `

     Expand source code
    
    
    def vectorize_tags(df, multilabel_params=None):
        """Векторизует теги игр, используя MultiLabelBinarizer для преобразования в бинарные векторы.
    
        Аргументы:
            df (pd.DataFrame): DataFrame, содержащий столбец 'all_tags' со списками тегов.
            multilabel_params (dict, optional): Параметры для MultiLabelBinarizer. По умолчанию None.
    
        Возвращает:
            tuple: Кортеж, содержащий:
                - np.ndarray: Векторизованные теги.
                - MultiLabelBinarizer: Обученный объект MultiLabelBinarizer.
        """
        default_params = {'sparse_output': False}
        params = multilabel_params if multilabel_params else default_params
        mlb = MultiLabelBinarizer(**params)
        mlb.fit(df['all_tags'])
        tags_vectorized = mlb.transform(df['all_tags'])
        return tags_vectorized, mlb

Векторизует теги игр, используя MultiLabelBinarizer для преобразования в
бинарные векторы.

Аргументы: df (pd.DataFrame): DataFrame, содержащий столбец 'all_tags' со
списками тегов. multilabel_params (dict, optional): Параметры для
MultiLabelBinarizer. По умолчанию None.

Возвращает: tuple: Кортеж, содержащий: \- np.ndarray: Векторизованные теги. \-
MultiLabelBinarizer: Обученный объект MultiLabelBinarizer.

## Classes

` class CombinedVectorizer (owners_method='log_scale',  
multilabel_params=None,  
nmf_params=None,  
lda_params=None,  
tag_weight=1.0,  
tfidf_cuml_params=None) `

     Expand source code
    
    
    class CombinedVectorizer(BaseEstimator, TransformerMixin):
        """Комбинированный векторизатор для обработки различных типов признаков.
    
        Векторизует данные о владельцах, теги и текстовые описания игр, используя различные методы и векторизаторы.
    
        Аргументы:
            owners_method (str, optional): Метод векторизации для данных о владельцах ('log_scale' или 'standard'). По умолчанию 'log_scale'.
            multilabel_params (dict, optional): Параметры для MultiLabelBinarizer. По умолчанию None.
            nmf_params (dict, optional): Параметры для NMF. Если указаны, используется NMF для векторизации описаний. По умолчанию None.
            lda_params (dict, optional): Параметры для LDA. Если указаны, используется LDA для векторизации описаний. По умолчанию None.
            tag_weight (float, optional): Вес, применяемый к векторизованным тегам. По умолчанию 1.0.
            tfidf_cuml_params (dict, optional): Параметры для CumlTfidfVectorizer. По умолчанию None.
        """
        def __init__(self, owners_method='log_scale', multilabel_params=None, nmf_params=None, lda_params=None, tag_weight=1.0, tfidf_cuml_params=None):
            self.owners_method = owners_method
            self.multilabel_params = multilabel_params
            self.nmf_params = nmf_params
            self.lda_params = lda_params
            self.tag_weight = tag_weight
            self.tfidf_cuml_params = tfidf_cuml_params if tfidf_cuml_params else {}
            self.mlb = None
            self.nmf = None
            self.lda = None
            self.tfidf_feature_names_out_ = None
            self.scaler = CumlMinMaxScaler()
            self.transformed_owners_vectors = None
            self.transformed_tags_vectors = None
            self.transformed_desc_vectors = None
            self.transformed_combined_vectors = None
    
        def fit(self, X, y=None):
            """Обучает векторизатор на предоставленных данных.
    
            Выполняет векторизацию владельцев, тегов и описаний, а также обучает внутренние векторизаторы и скалеры.
    
            Аргументы:
                X (pd.DataFrame): DataFrame, содержащий данные для векторизации ('estimated_owners', 'all_tags', 'short_description_clean').
                y (None): Не используется, нужен для совместимости API scikit-learn.
    
            Возвращает:
                CombinedVectorizer: Обученный векторизатор.
            """
            self.owners_vectors = vectorize_owners(X, method=self.owners_method)
            self.tags_vectors, self.mlb = vectorize_tags(X, multilabel_params=self.multilabel_params)
            if isinstance(self.tags_vectors, cp.sparse.csr_matrix):
                print("ℹ️ Векторы тегов - cupy sparse matrix, преобразование в numpy...")
                self.tags_vectors = np.array(cp.asnumpy(self.tags_vectors.todense()), dtype=np.float64)
            if self.tags_vectors.ndim == 1:
                self.tags_vectors = self.tags_vectors.reshape(-1, 1)
    
            cleaned_descriptions = X['short_description_clean'].str.lower()
            self.tfidf_cuml.fit(cleaned_descriptions)
            self.tfidf_feature_names_out_ = [word for word, index in sorted(self.tfidf_cuml.vocabulary_.to_pandas().items(), key=lambda item: item[1])]
    
            if self.nmf_params and self.lda_params is None:
                self.desc_vectors, self.nmf = vectorize_descriptions(X, nmf_params=self.nmf_params, vectorizer_cuml=self.tfidf_cuml)
                self.lda = None
            elif self.lda_params and self.nmf_params is None:
                 self.desc_vectors, self.lda = vectorize_descriptions(X, lda_params=self.lda_params, vectorizer_cuml=self.tfidf_cuml)
                 self.nmf = None
            else:
                raise ValueError("❌ Необходимо указать nmf_params или lda_params")
            print("✅ Векторизация описаний завершена")
    
            if self.nmf and hasattr(self.nmf, 'components_'):
                 if np.isnan(self.nmf.components_).any():
                    print("⚠️ Обнаружены NaN значения в self.nmf.components_ в fit()!")
            if self.lda and hasattr(self.lda, 'components_'):
                 if np.isnan(self.lda.components_).any():
                     print("⚠️ Обнаружены NaN значения в self.lda.components_ в fit()!")
    
            owners_vectors = vectorize_owners(X, method=self.owners_method)
            self.scaler.fit(owners_vectors)
            return self
    
        def transform(self, X, y=None):
            """Трансформирует входные данные в комбинированные векторы признаков.
    
            Использует обученные векторизаторы и скалеры для преобразования данных о владельцах, тегов и описаний в единое векторное представление.
    
            Аргументы:
                X (pd.DataFrame): DataFrame, содержащий данные для трансформации.
                y (None): Не используется, нужен для совместимости API scikit-learn.
    
            Возвращает:
                np.ndarray: Матрица комбинированных векторов признаков.
            """
            owners_vectors = vectorize_owners(X, method=self.owners_method, scaler=self.scaler)
            owners_vectors = owners_vectors.reshape(owners_vectors.shape[0], -1)
            tags_vectors = self.mlb.transform(X['all_tags'])
    
            tag_weight = self.tag_weight
            tags_vectors_weighted = tags_vectors * tag_weight
            tags_vectors = tags_vectors_weighted
    
            tfidf_transformed_cuml = self.tfidf_cuml.transform(X['short_description_clean'])
            tfidf_transformed_cpu = tfidf_transformed_cuml.get()
            data = cp.asnumpy(tfidf_transformed_cpu.data)
            indices = cp.asnumpy(tfidf_transformed_cpu.indices)
            indptr = cp.asnumpy(tfidf_transformed_cpu.indptr)
            shape = tfidf_transformed_cpu.shape
            tfidf_transformed = csr_matrix((data, indices, indptr), shape=shape)
    
            desc_vectors = None
            if self.nmf_params:
                desc_vectors = self.nmf.transform(tfidf_transformed)
            elif self.lda_params:
                desc_vectors = self.lda.transform(tfidf_transformed)
    
            if desc_vectors is not None and desc_vectors.shape[0] != owners_vectors.shape[0]:
                raise ValueError(f"❌ Несовпадение количества образцов между векторами владельцев и описаний: {owners_vectors.shape[0]} vs {desc_vectors.shape[0]}")
    
            self.transformed_owners_vectors = owners_vectors
            self.transformed_tags_vectors = tags_vectors
            self.transformed_desc_vectors = desc_vectors
    
            combined_vectors = np.hstack([owners_vectors, tags_vectors.toarray() if hasattr(tags_vectors, 'toarray') else tags_vectors, desc_vectors])
            self.transformed_combined_vectors = combined_vectors
    
            return combined_vectors
    
        def get_params(self, deep=True):
            """Возвращает параметры векторизатора.
    
            Возвращает словарь параметров данного векторизатора, включая параметры для всех внутренних векторизаторов и методов.
    
            Аргументы:
                deep (bool, optional): Если True, также возвращает параметры для вложенных объектов, которые являются оценщиками. По умолчанию True.
    
            Возвращает:
                dict: Словарь параметров векторизатора.
            """
            return {
                'owners_method': self.owners_method,
                'multilabel_params': self.multilabel_params,
                 'nmf_params': self.nmf_params,
                'lda_params': self.lda_params,
                'tag_weight': self.tag_weight,
                'tfidf_cuml_params': self.tfidf_cuml_params
            }
    
        def set_params(self, **params):
            """Устанавливает параметры векторизатора.
    
            Позволяет установить параметры векторизатора после инициализации.
    
            Аргументы:
                **params: Параметры векторизатора в виде keyword arguments.
    
            Возвращает:
                CombinedVectorizer: Векторизатор с установленными параметрами.
            """
            if 'owners_method' in params:
                self.owners_method = params['owners_method']
            if 'multilabel_params' in params:
                self.multilabel_params = params['multilabel_params']
            if 'nmf_params' in params:
                self.nmf_params = params['nmf_params']
            if 'lda_params' in params:
                 self.lda_params = params['lda_params']
            if 'tag_weight' in params:
                self.tag_weight = params['tag_weight']
            if 'tfidf_cuml_params' in params:
                self.tfidf_cuml_params = params['tfidf_cuml_params']
                self.tfidf_cuml.set_params(**params['tfidf_cuml_params'])
            return self

Комбинированный векторизатор для обработки различных типов признаков.

Векторизует данные о владельцах, теги и текстовые описания игр, используя
различные методы и векторизаторы.

Аргументы: owners_method (str, optional): Метод векторизации для данных о
владельцах ('log_scale' или 'standard'). По умолчанию 'log_scale'.
multilabel_params (dict, optional): Параметры для MultiLabelBinarizer. По
умолчанию None. nmf_params (dict, optional): Параметры для NMF. Если указаны,
используется NMF для векторизации описаний. По умолчанию None. lda_params
(dict, optional): Параметры для LDA. Если указаны, используется LDA для
векторизации описаний. По умолчанию None. tag_weight (float, optional): Вес,
применяемый к векторизованным тегам. По умолчанию 1.0. tfidf_cuml_params
(dict, optional): Параметры для CumlTfidfVectorizer. По умолчанию None.

### Ancestors

  * sklearn.base.BaseEstimator
  * sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
  * sklearn.utils._metadata_requests._MetadataRequester
  * sklearn.base.TransformerMixin
  * sklearn.utils._set_output._SetOutputMixin

### Methods

` def fit(self, X, y=None) `

     Expand source code
    
    
    def fit(self, X, y=None):
        """Обучает векторизатор на предоставленных данных.
    
        Выполняет векторизацию владельцев, тегов и описаний, а также обучает внутренние векторизаторы и скалеры.
    
        Аргументы:
            X (pd.DataFrame): DataFrame, содержащий данные для векторизации ('estimated_owners', 'all_tags', 'short_description_clean').
            y (None): Не используется, нужен для совместимости API scikit-learn.
    
        Возвращает:
            CombinedVectorizer: Обученный векторизатор.
        """
        self.owners_vectors = vectorize_owners(X, method=self.owners_method)
        self.tags_vectors, self.mlb = vectorize_tags(X, multilabel_params=self.multilabel_params)
        if isinstance(self.tags_vectors, cp.sparse.csr_matrix):
            print("ℹ️ Векторы тегов - cupy sparse matrix, преобразование в numpy...")
            self.tags_vectors = np.array(cp.asnumpy(self.tags_vectors.todense()), dtype=np.float64)
        if self.tags_vectors.ndim == 1:
            self.tags_vectors = self.tags_vectors.reshape(-1, 1)
    
        cleaned_descriptions = X['short_description_clean'].str.lower()
        self.tfidf_cuml.fit(cleaned_descriptions)
        self.tfidf_feature_names_out_ = [word for word, index in sorted(self.tfidf_cuml.vocabulary_.to_pandas().items(), key=lambda item: item[1])]
    
        if self.nmf_params and self.lda_params is None:
            self.desc_vectors, self.nmf = vectorize_descriptions(X, nmf_params=self.nmf_params, vectorizer_cuml=self.tfidf_cuml)
            self.lda = None
        elif self.lda_params and self.nmf_params is None:
             self.desc_vectors, self.lda = vectorize_descriptions(X, lda_params=self.lda_params, vectorizer_cuml=self.tfidf_cuml)
             self.nmf = None
        else:
            raise ValueError("❌ Необходимо указать nmf_params или lda_params")
        print("✅ Векторизация описаний завершена")
    
        if self.nmf and hasattr(self.nmf, 'components_'):
             if np.isnan(self.nmf.components_).any():
                print("⚠️ Обнаружены NaN значения в self.nmf.components_ в fit()!")
        if self.lda and hasattr(self.lda, 'components_'):
             if np.isnan(self.lda.components_).any():
                 print("⚠️ Обнаружены NaN значения в self.lda.components_ в fit()!")
    
        owners_vectors = vectorize_owners(X, method=self.owners_method)
        self.scaler.fit(owners_vectors)
        return self

Обучает векторизатор на предоставленных данных.

Выполняет векторизацию владельцев, тегов и описаний, а также обучает
внутренние векторизаторы и скалеры.

Аргументы: X (pd.DataFrame): DataFrame, содержащий данные для векторизации
('estimated_owners', 'all_tags', 'short_description_clean'). y (None): Не
используется, нужен для совместимости API scikit-learn.

Возвращает: CombinedVectorizer: Обученный векторизатор.

` def get_params(self, deep=True) `

     Expand source code
    
    
    def get_params(self, deep=True):
        """Возвращает параметры векторизатора.
    
        Возвращает словарь параметров данного векторизатора, включая параметры для всех внутренних векторизаторов и методов.
    
        Аргументы:
            deep (bool, optional): Если True, также возвращает параметры для вложенных объектов, которые являются оценщиками. По умолчанию True.
    
        Возвращает:
            dict: Словарь параметров векторизатора.
        """
        return {
            'owners_method': self.owners_method,
            'multilabel_params': self.multilabel_params,
             'nmf_params': self.nmf_params,
            'lda_params': self.lda_params,
            'tag_weight': self.tag_weight,
            'tfidf_cuml_params': self.tfidf_cuml_params
        }

Возвращает параметры векторизатора.

Возвращает словарь параметров данного векторизатора, включая параметры для
всех внутренних векторизаторов и методов.

Аргументы: deep (bool, optional): Если True, также возвращает параметры для
вложенных объектов, которые являются оценщиками. По умолчанию True.

Возвращает: dict: Словарь параметров векторизатора.

` def set_params(self, **params) `

     Expand source code
    
    
    def set_params(self, **params):
        """Устанавливает параметры векторизатора.
    
        Позволяет установить параметры векторизатора после инициализации.
    
        Аргументы:
            **params: Параметры векторизатора в виде keyword arguments.
    
        Возвращает:
            CombinedVectorizer: Векторизатор с установленными параметрами.
        """
        if 'owners_method' in params:
            self.owners_method = params['owners_method']
        if 'multilabel_params' in params:
            self.multilabel_params = params['multilabel_params']
        if 'nmf_params' in params:
            self.nmf_params = params['nmf_params']
        if 'lda_params' in params:
             self.lda_params = params['lda_params']
        if 'tag_weight' in params:
            self.tag_weight = params['tag_weight']
        if 'tfidf_cuml_params' in params:
            self.tfidf_cuml_params = params['tfidf_cuml_params']
            self.tfidf_cuml.set_params(**params['tfidf_cuml_params'])
        return self

Устанавливает параметры векторизатора.

Позволяет установить параметры векторизатора после инициализации.

Аргументы: **params: Параметры векторизатора в виде keyword arguments.

Возвращает: CombinedVectorizer: Векторизатор с установленными параметрами.

` def transform(self, X, y=None) `

     Expand source code
    
    
    def transform(self, X, y=None):
        """Трансформирует входные данные в комбинированные векторы признаков.
    
        Использует обученные векторизаторы и скалеры для преобразования данных о владельцах, тегов и описаний в единое векторное представление.
    
        Аргументы:
            X (pd.DataFrame): DataFrame, содержащий данные для трансформации.
            y (None): Не используется, нужен для совместимости API scikit-learn.
    
        Возвращает:
            np.ndarray: Матрица комбинированных векторов признаков.
        """
        owners_vectors = vectorize_owners(X, method=self.owners_method, scaler=self.scaler)
        owners_vectors = owners_vectors.reshape(owners_vectors.shape[0], -1)
        tags_vectors = self.mlb.transform(X['all_tags'])
    
        tag_weight = self.tag_weight
        tags_vectors_weighted = tags_vectors * tag_weight
        tags_vectors = tags_vectors_weighted
    
        tfidf_transformed_cuml = self.tfidf_cuml.transform(X['short_description_clean'])
        tfidf_transformed_cpu = tfidf_transformed_cuml.get()
        data = cp.asnumpy(tfidf_transformed_cpu.data)
        indices = cp.asnumpy(tfidf_transformed_cpu.indices)
        indptr = cp.asnumpy(tfidf_transformed_cpu.indptr)
        shape = tfidf_transformed_cpu.shape
        tfidf_transformed = csr_matrix((data, indices, indptr), shape=shape)
    
        desc_vectors = None
        if self.nmf_params:
            desc_vectors = self.nmf.transform(tfidf_transformed)
        elif self.lda_params:
            desc_vectors = self.lda.transform(tfidf_transformed)
    
        if desc_vectors is not None and desc_vectors.shape[0] != owners_vectors.shape[0]:
            raise ValueError(f"❌ Несовпадение количества образцов между векторами владельцев и описаний: {owners_vectors.shape[0]} vs {desc_vectors.shape[0]}")
    
        self.transformed_owners_vectors = owners_vectors
        self.transformed_tags_vectors = tags_vectors
        self.transformed_desc_vectors = desc_vectors
    
        combined_vectors = np.hstack([owners_vectors, tags_vectors.toarray() if hasattr(tags_vectors, 'toarray') else tags_vectors, desc_vectors])
        self.transformed_combined_vectors = combined_vectors
    
        return combined_vectors

Трансформирует входные данные в комбинированные векторы признаков.

Использует обученные векторизаторы и скалеры для преобразования данных о
владельцах, тегов и описаний в единое векторное представление.

Аргументы: X (pd.DataFrame): DataFrame, содержащий данные для трансформации. y
(None): Не используется, нужен для совместимости API scikit-learn.

Возвращает: np.ndarray: Матрица комбинированных векторов признаков.

` class TorchLDA (n_topics, n_vocab, device, alpha=0.1, beta=0.01,
max_iterations=100, tolerance=0.0001) `

     Expand source code
    
    
    class TorchLDA(nn.Module):
        """Реализация модели LDA (Latent Dirichlet Allocation) на PyTorch.
    
        Использует нейронную сеть для моделирования LDA, позволяя использовать GPU для ускорения вычислений.
    
        Аргументы:
            n_topics (int): Количество тем для моделирования.
            n_vocab (int): Размер словаря (количество уникальных слов).
            device (torch.device): Устройство, на котором будет выполняться обучение (CPU или GPU).
            alpha (float, optional): Параметр априорного распределения Дирихле для распределения документов по темам. По умолчанию 0.1.
            beta (float, optional): Параметр априорного распределения Дирихле для распределения тем по словам. По умолчанию 0.01.
            max_iterations (int, optional): Максимальное количество итераций EM-алгоритма. По умолчанию 100.
            tolerance (float, optional): Порог сходимости для EM-алгоритма. По умолчанию 1e-4.
        """
        def __init__(self, n_topics, n_vocab, device, alpha=0.1, beta=0.01, max_iterations=100, tolerance=1e-4):
            super().__init__()
            self.n_topics = n_topics
            self.n_vocab = n_vocab
            self.device = device
            self.alpha = alpha
            self.beta = beta
            self.max_iterations = max_iterations
            self.tolerance = tolerance
            self.topic_term_matrix = nn.Parameter(torch.randn(n_topics, n_vocab, device=device).abs())
            self.doc_topic_matrix = None
            self.norm_topic_term_matrix = None
    
        def initialize_parameters(self, docs):
            """Инициализирует параметры модели LDA случайными значениями.
    
            Аргументы:
                docs (torch.Tensor): Матрица документов (не используется в инициализации, но ожидается для совместимости интерфейса).
            """
            self.doc_topic_matrix = torch.rand(docs.shape[0], self.n_topics, device=self.device).abs()
            self.topic_term_matrix.data = torch.randn(self.n_topics, self.n_vocab, device=self.device).abs()
    
        def fit(self, docs, log=False):
            """Обучает модель LDA на основе предоставленных документов, используя EM-алгоритм.
    
            Аргументы:
                docs (torch.Tensor): Матрица документов (размерность: количество документов x размер словаря).
                log (bool, optional): Включает вывод логов во время обучения. По умолчанию False.
    
            Возвращает:
                TorchLDA: Обученная модель LDA.
            """
            if log: print("LDA Fit started")
            self.initialize_parameters(docs)
            docs = docs.to(self.device)
            prev_likelihood = float('-inf')
            for iteration in range(self.max_iterations):
                doc_topic_distribution = self.expect(docs)
                self.topic_term_matrix = self.maximize(docs, doc_topic_distribution)
                current_likelihood = self.likelihood(docs, doc_topic_distribution)
                if log: print(f"Iteration {iteration+1}, Likelihood {current_likelihood:.2f}")
                if abs(current_likelihood - prev_likelihood) < self.tolerance:
                    if log: print("LDA Converged")
                    break
                prev_likelihood = current_likelihood
            self.norm_topic_term_matrix = self.normalize(self.topic_term_matrix)
            if log: print("LDA Fit ended")
            return self
    
        def expect(self, docs):
            """Выполняет E-шаг EM-алгоритма для LDA: оценка распределения документов по темам.
    
            Аргументы:
                docs (torch.Tensor): Матрица документов.
    
            Возвращает:
                torch.Tensor: Матрица распределения документов по темам.
            """
            doc_topic_distribution = torch.matmul(docs, self.topic_term_matrix.T) + self.alpha
            doc_topic_distribution = self.normalize(doc_topic_distribution)
            return doc_topic_distribution
    
        def maximize(self, docs, doc_topic_distribution):
            """Выполняет M-шаг EM-алгоритма для LDA: оценка распределения тем по словам.
    
            Аргументы:
                docs (torch.Tensor): Матрица документов.
                doc_topic_distribution (torch.Tensor): Матрица распределения документов по темам.
    
            Возвращает:
                torch.Tensor: Матрица распределения тем по словам.
            """
            topic_term_matrix = torch.matmul(doc_topic_distribution.T, docs) + self.beta
            return topic_term_matrix
    
        def likelihood(self, docs, doc_topic_distribution):
            """Вычисляет логарифмическое правдоподобие для оценки сходимости EM-алгоритма.
    
            Аргументы:
                docs (torch.Tensor): Матрица документов.
                doc_topic_distribution (torch.Tensor): Матрица распределения документов по темам.
    
            Возвращает:
                float: Значение логарифмического правдоподобия.
            """
            log_likelihood = torch.sum(docs * torch.log(torch.matmul(doc_topic_distribution, self.normalize(self.topic_term_matrix))))
            return log_likelihood.item()
    
        def normalize(self, matrix):
            """Нормализует матрицу, приводя суммы строк к единице.
    
            Аргументы:
                matrix (torch.Tensor): Матрица для нормализации.
    
            Возвращает:
                torch.Tensor: Нормализованная матрица.
            """
            row_sums = matrix.sum(axis=1, keepdim=True)
            return matrix / row_sums
    
        def transform(self, docs):
            """Преобразует новые документы в векторное представление в пространстве тем.
    
            Аргументы:
                docs (torch.Tensor): Матрица новых документов.
    
            Возвращает:
                torch.Tensor: Матрица распределения документов по темам для новых документов.
    
            Вызывает ValueError, если модель LDA еще не обучена.
            """
            if self.norm_topic_term_matrix is None:
                raise ValueError("❌ LDA model has not been fitted yet.")
            docs = docs.to(self.device)
            doc_topic_distribution = torch.matmul(docs, self.norm_topic_term_matrix.T) + self.alpha
            return self.normalize(doc_topic_distribution)

Реализация модели LDA (Latent Dirichlet Allocation) на PyTorch.

Использует нейронную сеть для моделирования LDA, позволяя использовать GPU для
ускорения вычислений.

Аргументы: n_topics (int): Количество тем для моделирования. n_vocab (int):
Размер словаря (количество уникальных слов). device (torch.device):
Устройство, на котором будет выполняться обучение (CPU или GPU). alpha (float,
optional): Параметр априорного распределения Дирихле для распределения
документов по темам. По умолчанию 0.1. beta (float, optional): Параметр
априорного распределения Дирихле для распределения тем по словам. По умолчанию
0.01. max_iterations (int, optional): Максимальное количество итераций EM-
алгоритма. По умолчанию 100. tolerance (float, optional): Порог сходимости для
EM-алгоритма. По умолчанию 1e-4.

Initialize internal Module state, shared by both nn.Module and ScriptModule.

### Ancestors

  * torch.nn.modules.module.Module

### Methods

` def expect(self, docs) `

     Expand source code
    
    
    def expect(self, docs):
        """Выполняет E-шаг EM-алгоритма для LDA: оценка распределения документов по темам.
    
        Аргументы:
            docs (torch.Tensor): Матрица документов.
    
        Возвращает:
            torch.Tensor: Матрица распределения документов по темам.
        """
        doc_topic_distribution = torch.matmul(docs, self.topic_term_matrix.T) + self.alpha
        doc_topic_distribution = self.normalize(doc_topic_distribution)
        return doc_topic_distribution

Выполняет E-шаг EM-алгоритма для LDA: оценка распределения документов по
темам.

Аргументы: docs (torch.Tensor): Матрица документов.

Возвращает: torch.Tensor: Матрица распределения документов по темам.

` def fit(self, docs, log=False) `

     Expand source code
    
    
    def fit(self, docs, log=False):
        """Обучает модель LDA на основе предоставленных документов, используя EM-алгоритм.
    
        Аргументы:
            docs (torch.Tensor): Матрица документов (размерность: количество документов x размер словаря).
            log (bool, optional): Включает вывод логов во время обучения. По умолчанию False.
    
        Возвращает:
            TorchLDA: Обученная модель LDA.
        """
        if log: print("LDA Fit started")
        self.initialize_parameters(docs)
        docs = docs.to(self.device)
        prev_likelihood = float('-inf')
        for iteration in range(self.max_iterations):
            doc_topic_distribution = self.expect(docs)
            self.topic_term_matrix = self.maximize(docs, doc_topic_distribution)
            current_likelihood = self.likelihood(docs, doc_topic_distribution)
            if log: print(f"Iteration {iteration+1}, Likelihood {current_likelihood:.2f}")
            if abs(current_likelihood - prev_likelihood) < self.tolerance:
                if log: print("LDA Converged")
                break
            prev_likelihood = current_likelihood
        self.norm_topic_term_matrix = self.normalize(self.topic_term_matrix)
        if log: print("LDA Fit ended")
        return self

Обучает модель LDA на основе предоставленных документов, используя EM-
алгоритм.

Аргументы: docs (torch.Tensor): Матрица документов (размерность: количество
документов x размер словаря). log (bool, optional): Включает вывод логов во
время обучения. По умолчанию False.

Возвращает: TorchLDA: Обученная модель LDA.

` def initialize_parameters(self, docs) `

     Expand source code
    
    
    def initialize_parameters(self, docs):
        """Инициализирует параметры модели LDA случайными значениями.
    
        Аргументы:
            docs (torch.Tensor): Матрица документов (не используется в инициализации, но ожидается для совместимости интерфейса).
        """
        self.doc_topic_matrix = torch.rand(docs.shape[0], self.n_topics, device=self.device).abs()
        self.topic_term_matrix.data = torch.randn(self.n_topics, self.n_vocab, device=self.device).abs()

Инициализирует параметры модели LDA случайными значениями.

Аргументы: docs (torch.Tensor): Матрица документов (не используется в
инициализации, но ожидается для совместимости интерфейса).

` def likelihood(self, docs, doc_topic_distribution) `

     Expand source code
    
    
    def likelihood(self, docs, doc_topic_distribution):
        """Вычисляет логарифмическое правдоподобие для оценки сходимости EM-алгоритма.
    
        Аргументы:
            docs (torch.Tensor): Матрица документов.
            doc_topic_distribution (torch.Tensor): Матрица распределения документов по темам.
    
        Возвращает:
            float: Значение логарифмического правдоподобия.
        """
        log_likelihood = torch.sum(docs * torch.log(torch.matmul(doc_topic_distribution, self.normalize(self.topic_term_matrix))))
        return log_likelihood.item()

Вычисляет логарифмическое правдоподобие для оценки сходимости EM-алгоритма.

Аргументы: docs (torch.Tensor): Матрица документов. doc_topic_distribution
(torch.Tensor): Матрица распределения документов по темам.

Возвращает: float: Значение логарифмического правдоподобия.

` def maximize(self, docs, doc_topic_distribution) `

     Expand source code
    
    
    def maximize(self, docs, doc_topic_distribution):
        """Выполняет M-шаг EM-алгоритма для LDA: оценка распределения тем по словам.
    
        Аргументы:
            docs (torch.Tensor): Матрица документов.
            doc_topic_distribution (torch.Tensor): Матрица распределения документов по темам.
    
        Возвращает:
            torch.Tensor: Матрица распределения тем по словам.
        """
        topic_term_matrix = torch.matmul(doc_topic_distribution.T, docs) + self.beta
        return topic_term_matrix

Выполняет M-шаг EM-алгоритма для LDA: оценка распределения тем по словам.

Аргументы: docs (torch.Tensor): Матрица документов. doc_topic_distribution
(torch.Tensor): Матрица распределения документов по темам.

Возвращает: torch.Tensor: Матрица распределения тем по словам.

` def normalize(self, matrix) `

     Expand source code
    
    
    def normalize(self, matrix):
        """Нормализует матрицу, приводя суммы строк к единице.
    
        Аргументы:
            matrix (torch.Tensor): Матрица для нормализации.
    
        Возвращает:
            torch.Tensor: Нормализованная матрица.
        """
        row_sums = matrix.sum(axis=1, keepdim=True)
        return matrix / row_sums

Нормализует матрицу, приводя суммы строк к единице.

Аргументы: matrix (torch.Tensor): Матрица для нормализации.

Возвращает: torch.Tensor: Нормализованная матрица.

` def transform(self, docs) `

     Expand source code
    
    
    def transform(self, docs):
        """Преобразует новые документы в векторное представление в пространстве тем.
    
        Аргументы:
            docs (torch.Tensor): Матрица новых документов.
    
        Возвращает:
            torch.Tensor: Матрица распределения документов по темам для новых документов.
    
        Вызывает ValueError, если модель LDA еще не обучена.
        """
        if self.norm_topic_term_matrix is None:
            raise ValueError("❌ LDA model has not been fitted yet.")
        docs = docs.to(self.device)
        doc_topic_distribution = torch.matmul(docs, self.norm_topic_term_matrix.T) + self.alpha
        return self.normalize(doc_topic_distribution)

Преобразует новые документы в векторное представление в пространстве тем.

Аргументы: docs (torch.Tensor): Матрица новых документов.

Возвращает: torch.Tensor: Матрица распределения документов по темам для новых
документов.

Вызывает ValueError, если модель LDA еще не обучена.

  * ### Super-module

    * `[src](index.html "src")`
  * ### Functions

    * `calculate_intra_topic_diversity`
    * `calculate_topic_coherence`
    * `calculate_topic_diversity`
    * `clean_text`
    * `display_topics`
    * `display_topics_with_diversity`
    * `reduce_dataset`
    * `vectorize_descriptions`
    * `vectorize_owners`
    * `vectorize_tags`
  * ### Classes

    * #### `CombinedVectorizer`

      * `fit`
      * `get_params`
      * `set_params`
      * `transform`
    * #### `TorchLDA`

      * `expect`
      * `fit`
      * `initialize_parameters`
      * `likelihood`
      * `maximize`
      * `normalize`
      * `transform`

Generated by [pdoc 0.11.5](https://pdoc3.github.io/pdoc "pdoc: Python API
documentation generator").

