# Module `src.steam_library_analyzer`

## Functions

` def calculate_similarity_and_rank(self, games_data,
combination_method='average') `

     Expand source code
    
    
    def calculate_similarity_and_rank(self, games_data, combination_method='average'):
        """Вычисляет косинусную схожесть и ранжирует игры на основе комбинированных векторов."""
        if not games_data:
            return []
    
        print(f"\n--- ⚙️ Метод calculate_similarity_and_rank: {combination_method} ---")
    
        train_vectors = self.model.transform(self.train_df)
    
        new_df = pd.DataFrame(games_data)
        new_df['short_description_clean'] = new_df['short_description'].apply(clean_text)
        new_df['steam_id'] = [game.get('appid') for game in games_data]
    
        game_vectors = self.model.transform(new_df)
    
        if combination_method == 'average':
            combined_game_vector = np.mean(game_vectors, axis=0).reshape(1, -1)
            method_name = "Обычное усреднение векторов"
        elif combination_method == 'sum':
            combined_game_vector = np.sum(game_vectors, axis=0).reshape(1, -1)
            method_name = "Обычное суммирование векторов"
        else:
            raise ValueError(f"❌ Неизвестный метод комбинирования: {combination_method}")
    
        similarities = cosine_similarity(combined_game_vector, train_vectors)
        game_similarities = similarities[0]
        ranked_indices = np.argsort(game_similarities)[::-1]
    
        game_recommendations = []
    
        input_game_names_processed = {process_game_name(game_data.get('name')) for game_data in games_data if 'name' in game_data}
        print(f"🐞 input_game_names_processed: {input_game_names_processed}")
    
        for j in ranked_indices:
            recommended_game = self.train_df.iloc[j]
            recommended_game_name = recommended_game['name']
            recommended_game_name_processed = process_game_name(recommended_game_name)
    
            print(f"🤔 Рассматриваем рекомендацию: Название: {recommended_game_name}, Processed Name: {recommended_game_name_processed}")
    
            if recommended_game_name_processed in input_game_names_processed:
                print(f"🚫 Исключена игра: {recommended_game_name} (processed name: {recommended_game_name_processed}) так как она есть во входной группе.")
                continue
            else:
                print(f"🔍 Проверка имени: '{recommended_game_name_processed}' не в '{input_game_names_processed}' - игра НЕ из входной группы.")
    
            estimated_owners = recommended_game['estimated_owners']
            positive = recommended_game['positive']
            negative = recommended_game['negative']
    
            if estimated_owners == 0:
                print(f"🚫 Исключена игра: {recommended_game_name} (steam_id: {recommended_game.name}) из-за estimated_owners == 0.")
                continue
    
            if isinstance(positive, (int, float)) and isinstance(negative, (int, float)):
                total_reviews = positive + negative
                if total_reviews > 0:
                    positive_ratio = positive / total_reviews
                    if positive_ratio < 0.7:
                        print(f"🚫 Исключена игра: {recommended_game_name} (steam_id: {recommended_game.name}) из-за positive_ratio < 0.7 ({positive_ratio:.2f}).")
                        continue
    
            if len(game_recommendations) < 10:
                game_recommendations.append(
                    {
                        "name": recommended_game_name,
                        "estimated_owners": estimated_owners,
                        "steam_id": recommended_game.name,
                        "similarity_score": game_similarities[j]
                    }
                )
            if len(game_recommendations) >= 10:
                break
    
        scores = [d['similarity_score'] for d in game_recommendations]
        median_similarity = np.median(scores) if scores else 0
    
        ranked_game_group = {
            "game_data": {"name": f"Группа игр ({method_name})", "appid": f"group_{combination_method}"},
            "recommendations": game_recommendations,
            "median_similarity": median_similarity,
            "combination_method": method_name
        }
        return ranked_game_group

Вычисляет косинусную схожесть и ранжирует игры на основе комбинированных
векторов.

` def load_dataframe(df_path) `

     Expand source code
    
    
    def load_dataframe(df_path):
        """Загружает DataFrame из JSON файла и обрабатывает индекс."""
        df = pd.read_json(df_path)
        if 'steam_id' in df.index.names:
            df['steam_id'] = df.index
            df = df.reset_index(drop=True)
            df = df.set_index('steam_id')
        else:
            if df.index.is_numeric():
                df['steam_id'] = df.index.astype(int)
                df = df.reset_index(drop=True)
                df = df.set_index('steam_id')
        return df

Загружает DataFrame из JSON файла и обрабатывает индекс.

` def load_model(model_path) `

     Expand source code
    
    
    def load_model(model_path):
        """Загружает предварительно обученную модель из указанного файла."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

Загружает предварительно обученную модель из указанного файла.

` def main() `

     Expand source code
    
    
    def main():
        """Главная функция для запуска анализа библиотеки игр пользователя Steam или одиночной игры."""
        analyzer = LibraryAnalyzer()
        parser = argparse.ArgumentParser(description="Анализ библиотеки игр Steam или одиночной игры для получения рекомендаций.")
        group = parser.add_mutually_exclusive_group(required=True) # Группа для взаимоисключающих аргументов
    
        group.add_argument('--library', action='store_true', help='Запустить анализ библиотеки игр пользователя Steam (использует STEAM_USER_URL из .env по умолчанию).')
        group.add_argument('--game', type=str, help='Запустить анализ для одиночной игры. Укажите steamid, название игры или ссылку на игру.')
    
        args = parser.parse_args()
    
        if args.library:
            print("Выбран режим анализа библиотеки.")
            analyzer.run_analysis(STEAM_USER_URL) # Используем STEAM_USER_URL из .env
        elif args.game:
            print(f"Выбран режим анализа одиночной игры для: '{args.game}'.")
            game_recommendations = analyzer.analyze_single_game(args.game)
            if game_recommendations:
                print(f"\n--- 🏆 Рекомендации для игры '{args.game}' ---")
                recommendations = game_recommendations.get("recommendations")
                median_similarity = game_recommendations.get("median_similarity")
                combination_method_name = game_recommendations.get("combination_method")
    
                print(f"\n✨ Метод комбинирования: {combination_method_name}")
                print(f"⭐ Медиана similarity score: {median_similarity:.4f}")
                print("✅ Рекомендации:")
                for rec in recommendations:
                    print(f"  - 🎮 Название: {rec['name']}, 👤 Владельцы: {rec['estimated_owners']}, 🆔 Steam ID: {rec['steam_id']}, 💯 Схожесть: {rec['similarity_score']:.4f}")
                print("---")
            else:
                print("❌ Не удалось получить рекомендации для указанной игры.")

Главная функция для запуска анализа библиотеки игр пользователя Steam или
одиночной игры.

` def process_game_name(name) `

     Expand source code
    
    
    def process_game_name(name):
        """Обрабатывает название игры для целей сравнения."""
        if not isinstance(name, str):
            return ""
        name_lower = name.lower()
        name_latin_only = re.sub(r'[^a-z0-9\s]', '', name_lower)
        return name_latin_only

Обрабатывает название игры для целей сравнения.

## Classes

` class LibraryAnalyzer `

     Expand source code
    
    
    class LibraryAnalyzer:
        """
        Класс для анализа библиотеки игр пользователя Steam и генерации рекомендаций.
        """
        process_game_name = staticmethod(process_game_name)
        calculate_similarity_and_rank = calculate_similarity_and_rank
    
        def __init__(self):
            """
            Инициализирует LibraryAnalyzer.
            """
            self.api_parser = ApiParser()
            self.model = load_model(MODEL_PATH)
            self.train_df = load_dataframe(DF_PROCESSED_JSON_PATH)
            self.train_df['short_description_clean'] = self.train_df['short_description'].apply(clean_text)
            self.data_cleaner = DataCleaner()
    
        def get_games_data_from_dataset(self, games):
            """Извлекает данные об играх из предварительно загруженного датасета."""
            found_games = []
            not_found_games = []
            for game in games:
                app_id = game.get("appid")
                if app_id is not None and app_id in self.train_df.index:
                  found_games.append(self.train_df.loc[app_id])
                else:
                    not_found_games.append(game)
            return found_games, not_found_games
    
        def get_games_data_from_api(self, not_found_games):
            """Получает данные об играх из Steam API и Steam Spy API."""
            games_data = []
            if not not_found_games:
                return games_data
            for game in not_found_games:
                app_id = game.get('appid')
                if app_id:
                    steam_web_api_response = self.api_parser.steam_web_api_client.make_request(
                        request_name="get_app_details",
                        request_params={"appids": app_id}
                    )
    
                    app_data = None
                    if steam_web_api_response['error'] is None and steam_web_api_response['response_json']:
                        app_data = steam_web_api_response['response_json'].get(str(app_id), {}).get('data')
    
                    if app_data:
                        categories = app_data.get("categories", [])
                        if not isinstance(categories, list):
                            categories = []
                        parsed_data = {
                            "name": app_data.get("name", "placeholder"),
                            "release_date": app_data.get("release_date", {}).get("date", "placeholder"),
                            "required_age": app_data.get("required_age", "placeholder"),
                            "price": app_data.get("price_overview", {}).get("final_formatted", "placeholder"),
                            "dlc_count": len(app_data.get("dlc", [])) if isinstance(app_data.get("dlc"), list) else 0,
                            "detailed_description": app_data.get("detailed_description", "placeholder"),
                            "about_the_game": app_data.get("about_the_game", "placeholder"),
                            "short_description": app_data.get("short_description", "placeholder"),
                            "reviews": app_data.get("reviews", "placeholder"),
                            "header_image": app_data.get("header_image", "placeholder"),
                            "website": app_data.get("website", "placeholder"),
                            "support_url": app_data.get("support_info", {}).get("url", "placeholder"),
                            "support_email": app_data.get("support_info", {}).get("email", "placeholder"),
                            "windows": app_data.get("platforms", {}).get("windows", "placeholder"),
                            "mac": app_data.get("platforms", {}).get("mac", "placeholder"),
                            "linux": app_data.get("platforms", {}).get("linux", "placeholder"),
                            "metacritic_score": app_data.get("metacritic", {}).get("score", "placeholder"),
                            "metacritic_url": app_data.get("metacritic", {}).get("url", "placeholder"),
                            "achievements": app_data.get("achievements", {}).get("total", "placeholder"),
                            "recommendations": app_data.get("recommendations", {}).get("total", "placeholder"),
                            "notes": "placeholder",
                            "supported_languages": app_data.get("supported_languages", "placeholder"),
                            "full_audio_languages": app_data.get("full_audio_languages", "placeholder"),
                            "packages": app_data.get("packages", "placeholder"),
                            "developers": app_data.get("developers", "placeholder"),
                            "publishers": app_data.get("publishers", "placeholder"),
                            "categories": [cat.get("description", "placeholder") for cat in categories],
                            "genres": [genre.get("description", "placeholder") for genre in app_data.get("genres", [])],
                            "screenshots": [ss.get("path_thumbnail", "placeholder") for ss in app_data.get("screenshots", [])],
                            "movies": [m.get("webm", {}).get("480", "placeholder") for m in app_data.get("movies", [])],
                            "user_score": "placeholder",
                            "score_rank": "placeholder",
                            "positive": "placeholder",
                            "negative": "placeholder",
                            "estimated_owners": app_data.get("estimated_owners", np.nan),
                            "average_playtime_forever": "placeholder",
                            "average_playtime_2weeks": "placeholder",
                            "median_playtime_forever": "placeholder",
                            "median_playtime_2weeks": "placeholder",
                            "peak_ccu": "placeholder",
                            "all_tags": [cat.get("description", "placeholder") for cat in categories],
                            "steam_id": app_id
                        }
                        if pd.isna(parsed_data["estimated_owners"]):
                            parsed_data["estimated_owners"] = 100000
    
                        cleaned_data = self.data_cleaner.clean_data(pd.DataFrame([parsed_data]))
                        if cleaned_data.shape[0] > 0:
                            games_data.append(cleaned_data.to_dict('records')[0])
                    else:
                        steamspy_api_response = self.api_parser.steamspy_api_client.make_request(
                            request_name="get_app_details",
                            request_params={"appid": app_id}
                        )
                        if steamspy_api_response['error'] is None and steamspy_api_response['response_json']:
                            app_data = steamspy_api_response['response_json']
                            categories = app_data.get("categories", [])
                            if not isinstance(categories, list):
                                categories = []
    
                            parsed_data = {
                                "name": app_data.get("name", "placeholder"),
                                "release_date": "placeholder",
                                "required_age": "placeholder",
                                "price": "placeholder",
                                "dlc_count": "placeholder",
                                "detailed_description": "placeholder",
                                "about_the_game": "placeholder",
                                "short_description": "placeholder",
                                "reviews": "placeholder",
                                "header_image": "placeholder",
                                "website": "placeholder",
                                "support_url": "placeholder",
                                "support_email": "placeholder",
                                "windows": "placeholder",
                                "mac": "placeholder",
                                "linux": "placeholder",
                                "metacritic_score": app_data.get("metacritic", "placeholder"),
                                "metacritic_url": "placeholder",
                                "achievements": "placeholder",
                                "recommendations": "placeholder",
                                "notes": "placeholder",
                                "supported_languages": "placeholder",
                                "full_audio_languages": "placeholder",
                                "packages": "placeholder",
                                "developers": "placeholder",
                                "publishers": "placeholder",
                                "categories": "placeholder",
                                "genres": "placeholder",
                                "screenshots": "placeholder",
                                "movies": "placeholder",
                                "user_score": "placeholder",
                                "score_rank": "placeholder",
                                "positive": app_data.get("positive", "placeholder"),
                                "negative": app_data.get("negative", "placeholder"),
                                "estimated_owners": app_data.get("owners", np.nan),
                                "average_playtime_forever": "placeholder",
                                "average_playtime_2weeks": "placeholder",
                                "median_playtime_forever": "placeholder",
                                "median_playtime_2weeks": "placeholder",
                                "peak_ccu": "placeholder",
                                "all_tags": [cat.get("description", "placeholder") for cat in categories],
                                "steam_id": app_id
                            }
                            if pd.isna(parsed_data["estimated_owners"]):
                                parsed_data["estimated_owners"] = 100000
    
                            cleaned_data = self.data_cleaner.clean_data(pd.DataFrame([parsed_data]))
                            if cleaned_data.shape[0] > 0:
                                games_data.append(cleaned_data.to_dict('records')[0])
                        else:
                            print(f"⚠️ Не удалось получить данные из Steam API и Steam Spy API для app_id: {app_id}. Название игры: {game.get('name', 'Неизвестно')}")
            return games_data
    
        def analyze_single_game(self, game_identifier):
            """Анализирует одиночную игру и возвращает рекомендации."""
            print(f"🚀 Запуск анализа для одиночной игры: {game_identifier}")
    
            game_data_list = []
            app_id = None
    
            # Попытка извлечь appid из ссылки, если это ссылка
            if 'store.steampowered.com/app/' in game_identifier:
                match = re.search(r'/app/(\d+)', game_identifier)
                if match:
                    app_id = match.group(1)
            elif game_identifier.isdigit():
                app_id = game_identifier
    
            if app_id:
                print(f"🆔 Идентифицирован App ID: {app_id}")
                game_data_from_dataset, not_found_dataset = self.get_games_data_from_dataset([{'appid': app_id}])
                if game_data_from_dataset:
                    game_data_list = [fg.to_dict() for fg in game_data_from_dataset]
                else:
                    api_game_data = self.get_games_data_from_api([{'appid': app_id}])
                    if api_game_data:
                        game_data_list = api_game_data
                    else:
                        print(f"❌ Не удалось получить данные для игры с App ID: {app_id}")
                        return None
            else:
                # Поиск игры по названию (менее надежно, может быть несколько игр с похожими названиями)
                print(f"🔍 Поиск игры по названию: '{game_identifier}'")
                found_games_by_name = self.train_df[self.train_df['name'].str.lower() == game_identifier.lower()]
                if not found_games_by_name.empty:
                    print(f"✅ Игра по названию найдена в датасете.")
                    game_data_list = [found_games_by_name.iloc[0].to_dict()] # Берем первую найденную игру, если их несколько
                    app_id = game_data_list[0].get('steam_id')
                else:
                    print(f"⚠️ Игра по названию не найдена в датасете. Попытка поиска через API (может быть медленно).")
                    # Тут можно добавить поиск через API по названию, но это сложнее и выходит за рамки текущей задачи.
                    print(f"❌ Поиск по названию через API не реализован. Пожалуйста, используйте App ID или ссылку на игру.")
                    return None
    
            if not game_data_list:
                print(f"❌ Не удалось получить данные об игре для анализа: {game_identifier}")
                return None
    
            ranked_game_group = self.calculate_similarity_and_rank(game_data_list, combination_method='average')
            return ranked_game_group
    
    
        def analyze_single_game_for_gradio(self, game_identifier):
            """Анализирует одиночную игру и возвращает рекомендации в текстовом формате для Gradio."""
            ranked_game_group = self.analyze_single_game(game_identifier)
            if ranked_game_group:
                recommendations = ranked_game_group.get("recommendations")
                median_similarity = ranked_game_group.get("median_similarity")
                combination_method_name = ranked_game_group.get("combination_method")
    
                output_text = f"🏆 Рекомендации для игры '{game_identifier}':\n"
                output_text += f"✨ Метод комбинирования: {combination_method_name}\n"
                output_text += f"⭐ Медиана similarity score: {median_similarity:.4f}\n"
                output_text += "✅ Рекомендации:\n"
                for rec in recommendations:
                    output_text += f"  - 🎮 Название: {rec['name']}, 👤 Владельцы: {rec['estimated_owners']}, 🆔 Steam ID: {rec['steam_id']}, 💯 Схожесть: {rec['similarity_score']:.4f}\n"
                output_text += "---\n"
                return output_text
            else:
                return "❌ Не удалось получить рекомендации для указанной игры."
    
    
        def run_analysis_for_gradio(self, steam_user_url):
            """Запускает анализ библиотеки игр пользователя Steam и возвращает рекомендации в текстовом формате для Gradio."""
            ranked_games_with_similarity = self.run_analysis_get_results(steam_user_url) # Вызываем новую функцию для получения результатов
    
            if not ranked_games_with_similarity:
                return "❌ Не удалось получить рекомендации для библиотеки пользователя."
    
            output_text = ""
            for group_name in ["recent_games", "most_played_games"]:
                method_results = ranked_games_with_similarity.get(group_name, {})
                if method_results:
                    output_text += f"\n--- 🏆 Рекомендации для группы '{group_name}' ---\n"
                    output_text += f"Игры в группе: {[game_item['name'] for game_item in self.all_games_with_data.get(group_name, [])]}\n" # Используем сохраненные данные
                    for method, ranked_game_group in method_results.items():
                        recommendations = ranked_game_group.get("recommendations")
                        median_similarity = ranked_game_group.get("median_similarity")
                        combination_method_name = ranked_game_group.get("combination_method")
    
                        output_text += f"\n✨ Метод комбинирования: {combination_method_name}\n"
                        output_text += f"⭐ Медиана similarity score: {median_similarity:.4f}\n"
                        output_text += "✅ Рекомендации:\n"
                        for rec in recommendations:
                            output_text += f"  - 🎮 Название: {rec['name']}, 👤 Владельцы: {rec['estimated_owners']}, 🆔 Steam ID: {rec['steam_id']}, 💯 Схожесть: {rec['similarity_score']:.4f}\n"
                        output_text += "---\n"
            return output_text
    
    
        def run_analysis_get_results(self, steam_user_url):
            """
            Запускает анализ библиотеки игр пользователя Steam и возвращает результаты в виде словаря,
            без форматирования вывода для Gradio.
            """
            print(f"🚀 Запуск анализа библиотеки для пользователя с URL: {steam_user_url}")
            # Убираем повторный вызов resolve_vanity_url, предполагаем, что URL уже корректный
            # steam_user_id = self.api_parser.resolve_vanity_url(steam_user_url) # УДАЛЯЕМ ЭТУ СТРОКУ
            steam_user_id_match = re.search(r'/profiles/(\d+)', steam_user_url) # Пытаемся извлечь SteamID64 из URL
            if steam_user_id_match:
                steam_user_id = steam_user_id_match.group(1)
            elif steam_user_url.isdigit() and len(steam_user_url) == 17: # Если передан SteamID64 напрямую
                steam_user_id = steam_user_url
            elif '/id/' in steam_user_url: # Если vanity url, нужно получить id через API
                vanity_name = steam_user_url.split('/')[-1]
                steam_user_id = self.api_parser.resolve_vanity_url(vanity_name)
            else:
                print("❌ Не удалось получить Steam ID пользователя из URL.")
                return None
    
    
            if not steam_user_id:
                print("❌ Не удалось получить Steam ID пользователя.")
                return None
            print(f"👤 Получен Steam ID пользователя: {steam_user_id}")
    
            owned_games = self.api_parser.get_owned_games(steam_user_id)
            if not owned_games:
                print("⚠️ Не удалось получить список игр пользователя.")
                return None
            print(f"🎮 Получено {len(owned_games)} игр от пользователя.")
    
            grouped_games_data = {}
            grouped_games = group_user_games(owned_games)
            print(f"📦 Игры сгруппированы: {grouped_games.keys()}")
    
            self.all_games_with_data = {} # Сохраняем данные о играх для использования в Gradio output
    
            for group_name in ["recent_games", "most_played_games"]:
                games = grouped_games.get(group_name, [])
                if not games:
                    grouped_games_data[group_name] = []
                    continue
    
                all_games_with_data_for_group = []
    
                print(f"🔍 Обработка группы игр: {group_name}")
                found_games, not_found_games = self.get_games_data_from_dataset(games)
                print(f"   ✅ Найдено в датасете: {len(found_games)} игр, ⚠️ не найдено в датасете: {len(not_found_games)} игр")
                api_games_data = self.get_games_data_from_api(not_found_games)
                print(f"   ✅ Получено из API: {len(api_games_data)} игр")
    
                found_game_dict_list = [fg.to_dict() if isinstance(fg, pd.Series) else fg for fg in found_games]
                all_games_with_data_for_group = found_game_dict_list + api_games_data
                self.all_games_with_data[group_name] = all_games_with_data_for_group # Сохраняем тут
    
            ranked_games_with_similarity = {}
            combination_methods_to_test = ['average']
    
            for group_name in ["recent_games", "most_played_games"]:
                games_data = self.all_games_with_data.get(group_name, []) # Используем сохраненные данные
                if games_data:
                    ranked_games_with_similarity[group_name] = {}
                    for method in combination_methods_to_test:
                        print(f"📊 Расчет similarity score для группы '{group_name}' методом '{method}'")
                        ranked_group_results = self.calculate_similarity_and_rank(games_data, combination_method=method)
                        ranked_games_with_similarity[group_name][method] = ranked_group_results
                else:
                    print(f"ℹ️ Нет данных об играх для группы '{group_name}'. Пропускаем расчет similarity.")
    
            return ranked_games_with_similarity

Класс для анализа библиотеки игр пользователя Steam и генерации рекомендаций.

Инициализирует LibraryAnalyzer.

### Static methods

` def process_game_name(name) `

     Expand source code
    
    
    def process_game_name(name):
        """Обрабатывает название игры для целей сравнения."""
        if not isinstance(name, str):
            return ""
        name_lower = name.lower()
        name_latin_only = re.sub(r'[^a-z0-9\s]', '', name_lower)
        return name_latin_only

Обрабатывает название игры для целей сравнения.

### Methods

` def analyze_single_game(self, game_identifier) `

     Expand source code
    
    
    def analyze_single_game(self, game_identifier):
        """Анализирует одиночную игру и возвращает рекомендации."""
        print(f"🚀 Запуск анализа для одиночной игры: {game_identifier}")
    
        game_data_list = []
        app_id = None
    
        # Попытка извлечь appid из ссылки, если это ссылка
        if 'store.steampowered.com/app/' in game_identifier:
            match = re.search(r'/app/(\d+)', game_identifier)
            if match:
                app_id = match.group(1)
        elif game_identifier.isdigit():
            app_id = game_identifier
    
        if app_id:
            print(f"🆔 Идентифицирован App ID: {app_id}")
            game_data_from_dataset, not_found_dataset = self.get_games_data_from_dataset([{'appid': app_id}])
            if game_data_from_dataset:
                game_data_list = [fg.to_dict() for fg in game_data_from_dataset]
            else:
                api_game_data = self.get_games_data_from_api([{'appid': app_id}])
                if api_game_data:
                    game_data_list = api_game_data
                else:
                    print(f"❌ Не удалось получить данные для игры с App ID: {app_id}")
                    return None
        else:
            # Поиск игры по названию (менее надежно, может быть несколько игр с похожими названиями)
            print(f"🔍 Поиск игры по названию: '{game_identifier}'")
            found_games_by_name = self.train_df[self.train_df['name'].str.lower() == game_identifier.lower()]
            if not found_games_by_name.empty:
                print(f"✅ Игра по названию найдена в датасете.")
                game_data_list = [found_games_by_name.iloc[0].to_dict()] # Берем первую найденную игру, если их несколько
                app_id = game_data_list[0].get('steam_id')
            else:
                print(f"⚠️ Игра по названию не найдена в датасете. Попытка поиска через API (может быть медленно).")
                # Тут можно добавить поиск через API по названию, но это сложнее и выходит за рамки текущей задачи.
                print(f"❌ Поиск по названию через API не реализован. Пожалуйста, используйте App ID или ссылку на игру.")
                return None
    
        if not game_data_list:
            print(f"❌ Не удалось получить данные об игре для анализа: {game_identifier}")
            return None
    
        ranked_game_group = self.calculate_similarity_and_rank(game_data_list, combination_method='average')
        return ranked_game_group

Анализирует одиночную игру и возвращает рекомендации.

` def analyze_single_game_for_gradio(self, game_identifier) `

     Expand source code
    
    
    def analyze_single_game_for_gradio(self, game_identifier):
        """Анализирует одиночную игру и возвращает рекомендации в текстовом формате для Gradio."""
        ranked_game_group = self.analyze_single_game(game_identifier)
        if ranked_game_group:
            recommendations = ranked_game_group.get("recommendations")
            median_similarity = ranked_game_group.get("median_similarity")
            combination_method_name = ranked_game_group.get("combination_method")
    
            output_text = f"🏆 Рекомендации для игры '{game_identifier}':\n"
            output_text += f"✨ Метод комбинирования: {combination_method_name}\n"
            output_text += f"⭐ Медиана similarity score: {median_similarity:.4f}\n"
            output_text += "✅ Рекомендации:\n"
            for rec in recommendations:
                output_text += f"  - 🎮 Название: {rec['name']}, 👤 Владельцы: {rec['estimated_owners']}, 🆔 Steam ID: {rec['steam_id']}, 💯 Схожесть: {rec['similarity_score']:.4f}\n"
            output_text += "---\n"
            return output_text
        else:
            return "❌ Не удалось получить рекомендации для указанной игры."

Анализирует одиночную игру и возвращает рекомендации в текстовом формате для
Gradio.

` def calculate_similarity_and_rank(self, games_data,
combination_method='average') `

     Expand source code
    
    
    def calculate_similarity_and_rank(self, games_data, combination_method='average'):
        """Вычисляет косинусную схожесть и ранжирует игры на основе комбинированных векторов."""
        if not games_data:
            return []
    
        print(f"\n--- ⚙️ Метод calculate_similarity_and_rank: {combination_method} ---")
    
        train_vectors = self.model.transform(self.train_df)
    
        new_df = pd.DataFrame(games_data)
        new_df['short_description_clean'] = new_df['short_description'].apply(clean_text)
        new_df['steam_id'] = [game.get('appid') for game in games_data]
    
        game_vectors = self.model.transform(new_df)
    
        if combination_method == 'average':
            combined_game_vector = np.mean(game_vectors, axis=0).reshape(1, -1)
            method_name = "Обычное усреднение векторов"
        elif combination_method == 'sum':
            combined_game_vector = np.sum(game_vectors, axis=0).reshape(1, -1)
            method_name = "Обычное суммирование векторов"
        else:
            raise ValueError(f"❌ Неизвестный метод комбинирования: {combination_method}")
    
        similarities = cosine_similarity(combined_game_vector, train_vectors)
        game_similarities = similarities[0]
        ranked_indices = np.argsort(game_similarities)[::-1]
    
        game_recommendations = []
    
        input_game_names_processed = {process_game_name(game_data.get('name')) for game_data in games_data if 'name' in game_data}
        print(f"🐞 input_game_names_processed: {input_game_names_processed}")
    
        for j in ranked_indices:
            recommended_game = self.train_df.iloc[j]
            recommended_game_name = recommended_game['name']
            recommended_game_name_processed = process_game_name(recommended_game_name)
    
            print(f"🤔 Рассматриваем рекомендацию: Название: {recommended_game_name}, Processed Name: {recommended_game_name_processed}")
    
            if recommended_game_name_processed in input_game_names_processed:
                print(f"🚫 Исключена игра: {recommended_game_name} (processed name: {recommended_game_name_processed}) так как она есть во входной группе.")
                continue
            else:
                print(f"🔍 Проверка имени: '{recommended_game_name_processed}' не в '{input_game_names_processed}' - игра НЕ из входной группы.")
    
            estimated_owners = recommended_game['estimated_owners']
            positive = recommended_game['positive']
            negative = recommended_game['negative']
    
            if estimated_owners == 0:
                print(f"🚫 Исключена игра: {recommended_game_name} (steam_id: {recommended_game.name}) из-за estimated_owners == 0.")
                continue
    
            if isinstance(positive, (int, float)) and isinstance(negative, (int, float)):
                total_reviews = positive + negative
                if total_reviews > 0:
                    positive_ratio = positive / total_reviews
                    if positive_ratio < 0.7:
                        print(f"🚫 Исключена игра: {recommended_game_name} (steam_id: {recommended_game.name}) из-за positive_ratio < 0.7 ({positive_ratio:.2f}).")
                        continue
    
            if len(game_recommendations) < 10:
                game_recommendations.append(
                    {
                        "name": recommended_game_name,
                        "estimated_owners": estimated_owners,
                        "steam_id": recommended_game.name,
                        "similarity_score": game_similarities[j]
                    }
                )
            if len(game_recommendations) >= 10:
                break
    
        scores = [d['similarity_score'] for d in game_recommendations]
        median_similarity = np.median(scores) if scores else 0
    
        ranked_game_group = {
            "game_data": {"name": f"Группа игр ({method_name})", "appid": f"group_{combination_method}"},
            "recommendations": game_recommendations,
            "median_similarity": median_similarity,
            "combination_method": method_name
        }
        return ranked_game_group

Вычисляет косинусную схожесть и ранжирует игры на основе комбинированных
векторов.

` def get_games_data_from_api(self, not_found_games) `

     Expand source code
    
    
    def get_games_data_from_api(self, not_found_games):
        """Получает данные об играх из Steam API и Steam Spy API."""
        games_data = []
        if not not_found_games:
            return games_data
        for game in not_found_games:
            app_id = game.get('appid')
            if app_id:
                steam_web_api_response = self.api_parser.steam_web_api_client.make_request(
                    request_name="get_app_details",
                    request_params={"appids": app_id}
                )
    
                app_data = None
                if steam_web_api_response['error'] is None and steam_web_api_response['response_json']:
                    app_data = steam_web_api_response['response_json'].get(str(app_id), {}).get('data')
    
                if app_data:
                    categories = app_data.get("categories", [])
                    if not isinstance(categories, list):
                        categories = []
                    parsed_data = {
                        "name": app_data.get("name", "placeholder"),
                        "release_date": app_data.get("release_date", {}).get("date", "placeholder"),
                        "required_age": app_data.get("required_age", "placeholder"),
                        "price": app_data.get("price_overview", {}).get("final_formatted", "placeholder"),
                        "dlc_count": len(app_data.get("dlc", [])) if isinstance(app_data.get("dlc"), list) else 0,
                        "detailed_description": app_data.get("detailed_description", "placeholder"),
                        "about_the_game": app_data.get("about_the_game", "placeholder"),
                        "short_description": app_data.get("short_description", "placeholder"),
                        "reviews": app_data.get("reviews", "placeholder"),
                        "header_image": app_data.get("header_image", "placeholder"),
                        "website": app_data.get("website", "placeholder"),
                        "support_url": app_data.get("support_info", {}).get("url", "placeholder"),
                        "support_email": app_data.get("support_info", {}).get("email", "placeholder"),
                        "windows": app_data.get("platforms", {}).get("windows", "placeholder"),
                        "mac": app_data.get("platforms", {}).get("mac", "placeholder"),
                        "linux": app_data.get("platforms", {}).get("linux", "placeholder"),
                        "metacritic_score": app_data.get("metacritic", {}).get("score", "placeholder"),
                        "metacritic_url": app_data.get("metacritic", {}).get("url", "placeholder"),
                        "achievements": app_data.get("achievements", {}).get("total", "placeholder"),
                        "recommendations": app_data.get("recommendations", {}).get("total", "placeholder"),
                        "notes": "placeholder",
                        "supported_languages": app_data.get("supported_languages", "placeholder"),
                        "full_audio_languages": app_data.get("full_audio_languages", "placeholder"),
                        "packages": app_data.get("packages", "placeholder"),
                        "developers": app_data.get("developers", "placeholder"),
                        "publishers": app_data.get("publishers", "placeholder"),
                        "categories": [cat.get("description", "placeholder") for cat in categories],
                        "genres": [genre.get("description", "placeholder") for genre in app_data.get("genres", [])],
                        "screenshots": [ss.get("path_thumbnail", "placeholder") for ss in app_data.get("screenshots", [])],
                        "movies": [m.get("webm", {}).get("480", "placeholder") for m in app_data.get("movies", [])],
                        "user_score": "placeholder",
                        "score_rank": "placeholder",
                        "positive": "placeholder",
                        "negative": "placeholder",
                        "estimated_owners": app_data.get("estimated_owners", np.nan),
                        "average_playtime_forever": "placeholder",
                        "average_playtime_2weeks": "placeholder",
                        "median_playtime_forever": "placeholder",
                        "median_playtime_2weeks": "placeholder",
                        "peak_ccu": "placeholder",
                        "all_tags": [cat.get("description", "placeholder") for cat in categories],
                        "steam_id": app_id
                    }
                    if pd.isna(parsed_data["estimated_owners"]):
                        parsed_data["estimated_owners"] = 100000
    
                    cleaned_data = self.data_cleaner.clean_data(pd.DataFrame([parsed_data]))
                    if cleaned_data.shape[0] > 0:
                        games_data.append(cleaned_data.to_dict('records')[0])
                else:
                    steamspy_api_response = self.api_parser.steamspy_api_client.make_request(
                        request_name="get_app_details",
                        request_params={"appid": app_id}
                    )
                    if steamspy_api_response['error'] is None and steamspy_api_response['response_json']:
                        app_data = steamspy_api_response['response_json']
                        categories = app_data.get("categories", [])
                        if not isinstance(categories, list):
                            categories = []
    
                        parsed_data = {
                            "name": app_data.get("name", "placeholder"),
                            "release_date": "placeholder",
                            "required_age": "placeholder",
                            "price": "placeholder",
                            "dlc_count": "placeholder",
                            "detailed_description": "placeholder",
                            "about_the_game": "placeholder",
                            "short_description": "placeholder",
                            "reviews": "placeholder",
                            "header_image": "placeholder",
                            "website": "placeholder",
                            "support_url": "placeholder",
                            "support_email": "placeholder",
                            "windows": "placeholder",
                            "mac": "placeholder",
                            "linux": "placeholder",
                            "metacritic_score": app_data.get("metacritic", "placeholder"),
                            "metacritic_url": "placeholder",
                            "achievements": "placeholder",
                            "recommendations": "placeholder",
                            "notes": "placeholder",
                            "supported_languages": "placeholder",
                            "full_audio_languages": "placeholder",
                            "packages": "placeholder",
                            "developers": "placeholder",
                            "publishers": "placeholder",
                            "categories": "placeholder",
                            "genres": "placeholder",
                            "screenshots": "placeholder",
                            "movies": "placeholder",
                            "user_score": "placeholder",
                            "score_rank": "placeholder",
                            "positive": app_data.get("positive", "placeholder"),
                            "negative": app_data.get("negative", "placeholder"),
                            "estimated_owners": app_data.get("owners", np.nan),
                            "average_playtime_forever": "placeholder",
                            "average_playtime_2weeks": "placeholder",
                            "median_playtime_forever": "placeholder",
                            "median_playtime_2weeks": "placeholder",
                            "peak_ccu": "placeholder",
                            "all_tags": [cat.get("description", "placeholder") for cat in categories],
                            "steam_id": app_id
                        }
                        if pd.isna(parsed_data["estimated_owners"]):
                            parsed_data["estimated_owners"] = 100000
    
                        cleaned_data = self.data_cleaner.clean_data(pd.DataFrame([parsed_data]))
                        if cleaned_data.shape[0] > 0:
                            games_data.append(cleaned_data.to_dict('records')[0])
                    else:
                        print(f"⚠️ Не удалось получить данные из Steam API и Steam Spy API для app_id: {app_id}. Название игры: {game.get('name', 'Неизвестно')}")
        return games_data

Получает данные об играх из Steam API и Steam Spy API.

` def get_games_data_from_dataset(self, games) `

     Expand source code
    
    
    def get_games_data_from_dataset(self, games):
        """Извлекает данные об играх из предварительно загруженного датасета."""
        found_games = []
        not_found_games = []
        for game in games:
            app_id = game.get("appid")
            if app_id is not None and app_id in self.train_df.index:
              found_games.append(self.train_df.loc[app_id])
            else:
                not_found_games.append(game)
        return found_games, not_found_games

Извлекает данные об играх из предварительно загруженного датасета.

` def run_analysis_for_gradio(self, steam_user_url) `

     Expand source code
    
    
    def run_analysis_for_gradio(self, steam_user_url):
        """Запускает анализ библиотеки игр пользователя Steam и возвращает рекомендации в текстовом формате для Gradio."""
        ranked_games_with_similarity = self.run_analysis_get_results(steam_user_url) # Вызываем новую функцию для получения результатов
    
        if not ranked_games_with_similarity:
            return "❌ Не удалось получить рекомендации для библиотеки пользователя."
    
        output_text = ""
        for group_name in ["recent_games", "most_played_games"]:
            method_results = ranked_games_with_similarity.get(group_name, {})
            if method_results:
                output_text += f"\n--- 🏆 Рекомендации для группы '{group_name}' ---\n"
                output_text += f"Игры в группе: {[game_item['name'] for game_item in self.all_games_with_data.get(group_name, [])]}\n" # Используем сохраненные данные
                for method, ranked_game_group in method_results.items():
                    recommendations = ranked_game_group.get("recommendations")
                    median_similarity = ranked_game_group.get("median_similarity")
                    combination_method_name = ranked_game_group.get("combination_method")
    
                    output_text += f"\n✨ Метод комбинирования: {combination_method_name}\n"
                    output_text += f"⭐ Медиана similarity score: {median_similarity:.4f}\n"
                    output_text += "✅ Рекомендации:\n"
                    for rec in recommendations:
                        output_text += f"  - 🎮 Название: {rec['name']}, 👤 Владельцы: {rec['estimated_owners']}, 🆔 Steam ID: {rec['steam_id']}, 💯 Схожесть: {rec['similarity_score']:.4f}\n"
                    output_text += "---\n"
        return output_text

Запускает анализ библиотеки игр пользователя Steam и возвращает рекомендации в
текстовом формате для Gradio.

` def run_analysis_get_results(self, steam_user_url) `

     Expand source code
    
    
    def run_analysis_get_results(self, steam_user_url):
        """
        Запускает анализ библиотеки игр пользователя Steam и возвращает результаты в виде словаря,
        без форматирования вывода для Gradio.
        """
        print(f"🚀 Запуск анализа библиотеки для пользователя с URL: {steam_user_url}")
        # Убираем повторный вызов resolve_vanity_url, предполагаем, что URL уже корректный
        # steam_user_id = self.api_parser.resolve_vanity_url(steam_user_url) # УДАЛЯЕМ ЭТУ СТРОКУ
        steam_user_id_match = re.search(r'/profiles/(\d+)', steam_user_url) # Пытаемся извлечь SteamID64 из URL
        if steam_user_id_match:
            steam_user_id = steam_user_id_match.group(1)
        elif steam_user_url.isdigit() and len(steam_user_url) == 17: # Если передан SteamID64 напрямую
            steam_user_id = steam_user_url
        elif '/id/' in steam_user_url: # Если vanity url, нужно получить id через API
            vanity_name = steam_user_url.split('/')[-1]
            steam_user_id = self.api_parser.resolve_vanity_url(vanity_name)
        else:
            print("❌ Не удалось получить Steam ID пользователя из URL.")
            return None
    
    
        if not steam_user_id:
            print("❌ Не удалось получить Steam ID пользователя.")
            return None
        print(f"👤 Получен Steam ID пользователя: {steam_user_id}")
    
        owned_games = self.api_parser.get_owned_games(steam_user_id)
        if not owned_games:
            print("⚠️ Не удалось получить список игр пользователя.")
            return None
        print(f"🎮 Получено {len(owned_games)} игр от пользователя.")
    
        grouped_games_data = {}
        grouped_games = group_user_games(owned_games)
        print(f"📦 Игры сгруппированы: {grouped_games.keys()}")
    
        self.all_games_with_data = {} # Сохраняем данные о играх для использования в Gradio output
    
        for group_name in ["recent_games", "most_played_games"]:
            games = grouped_games.get(group_name, [])
            if not games:
                grouped_games_data[group_name] = []
                continue
    
            all_games_with_data_for_group = []
    
            print(f"🔍 Обработка группы игр: {group_name}")
            found_games, not_found_games = self.get_games_data_from_dataset(games)
            print(f"   ✅ Найдено в датасете: {len(found_games)} игр, ⚠️ не найдено в датасете: {len(not_found_games)} игр")
            api_games_data = self.get_games_data_from_api(not_found_games)
            print(f"   ✅ Получено из API: {len(api_games_data)} игр")
    
            found_game_dict_list = [fg.to_dict() if isinstance(fg, pd.Series) else fg for fg in found_games]
            all_games_with_data_for_group = found_game_dict_list + api_games_data
            self.all_games_with_data[group_name] = all_games_with_data_for_group # Сохраняем тут
    
        ranked_games_with_similarity = {}
        combination_methods_to_test = ['average']
    
        for group_name in ["recent_games", "most_played_games"]:
            games_data = self.all_games_with_data.get(group_name, []) # Используем сохраненные данные
            if games_data:
                ranked_games_with_similarity[group_name] = {}
                for method in combination_methods_to_test:
                    print(f"📊 Расчет similarity score для группы '{group_name}' методом '{method}'")
                    ranked_group_results = self.calculate_similarity_and_rank(games_data, combination_method=method)
                    ranked_games_with_similarity[group_name][method] = ranked_group_results
            else:
                print(f"ℹ️ Нет данных об играх для группы '{group_name}'. Пропускаем расчет similarity.")
    
        return ranked_games_with_similarity

Запускает анализ библиотеки игр пользователя Steam и возвращает результаты в
виде словаря, без форматирования вывода для Gradio.

  * ### Super-module

    * `[src](index.html "src")`
  * ### Functions

    * `calculate_similarity_and_rank`
    * `load_dataframe`
    * `load_model`
    * `main`
    * `process_game_name`
  * ### Classes

    * #### `LibraryAnalyzer`

      * `analyze_single_game`
      * `analyze_single_game_for_gradio`
      * `calculate_similarity_and_rank`
      * `get_games_data_from_api`
      * `get_games_data_from_dataset`
      * `process_game_name`
      * `run_analysis_for_gradio`
      * `run_analysis_get_results`

Generated by [pdoc 0.11.5](https://pdoc3.github.io/pdoc "pdoc: Python API
documentation generator").

