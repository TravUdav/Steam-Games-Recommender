import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steam_api_parser import ApiParser
from steam_library_grouper import group_user_games
from steam_constants import all_api_requests
from vectorizer import CombinedVectorizer, clean_text
from dataset_cleaner import DataCleaner
from steam_library_analyzer import LibraryAnalyzer
import gradio as gr


def is_steam_profile_url(input_string):
    """Проверяет, является ли строка ссылкой на профиль Steam (id или profiles)."""
    return "steamcommunity.com/id/" in input_string or "steamcommunity.com/profiles/" in input_string

def is_steam_app_url(input_string):
    """Проверяет, является ли строка ссылкой на игру в Steam Store."""
    return "store.steampowered.com/app/" in input_string

def is_steamid64(input_string):
    """Проверяет, является ли строка SteamID64 (17 цифр)."""
    return input_string.isdigit() and len(input_string) == 17

def analyze_input(user_input):
    """
    Функция для анализа пользовательского ввода и получения рекомендаций.
    Определяет тип ввода (URL профиля, vanity URL, SteamID64, название игры, URL игры)
    и вызывает соответствующую функцию анализатора.
    """
    analyzer = LibraryAnalyzer()
    user_input = user_input.strip()

    if not user_input:
        return "❌ Ввод не может быть пустым. Пожалуйста, введите корректные данные."

    if is_steam_profile_url(user_input):
        print("⚙️ Обнаружен запрос библиотеки пользователя (URL профиля).")
        steam_user_url = user_input
        return analyzer.run_analysis_for_gradio(steam_user_url)

    if is_steam_app_url(user_input):
        print("⚙️ Обнаружен запрос одиночной игры (URL игры).")
        return analyzer.analyze_single_game_for_gradio(user_input)

    if is_steamid64(user_input):
        print("⚙️ Обнаружен запрос библиотеки пользователя (SteamID64).")
        steam_user_url = f"https://steamcommunity.com/profiles/{user_input}"
        return analyzer.run_analysis_for_gradio(steam_user_url)

    print("⚙️ Попытка обработки как названия игры.")
    game_identifier = user_input
    recommendations_output = analyzer.analyze_single_game_for_gradio(game_identifier)
    if recommendations_output != "❌ Не удалось получить рекомендации для указанной игры.":
        print("✅ Распознано как запрос одиночной игры по названию.")
        return recommendations_output

    vanity_resolution_result = validate_vanity_url(user_input, analyzer.api_parser)
    if vanity_resolution_result:
        if isinstance(vanity_resolution_result, str):
            steam_user_url = vanity_resolution_result
            print("⚙️ Обнаружен запрос библиотеки пользователя (vanity URL), подтвержден через API.")
            return analyzer.run_analysis_for_gradio(steam_user_url)
        elif vanity_resolution_result is False:
            pass
        else:
            return "❌ Ошибка при проверке vanity URL через API. Пожалуйста, попробуйте позже."

    return "❌ Не удалось найти пользователя или игру по введенному запросу. Пожалуйста, введите корректную ссылку на профиль Steam, ссылку на игру, Steam ID игры или название игры."


def validate_vanity_url(vanity_url_input, api_parser):
    """
    Проверяет, является ли ввод vanity URL и пытается преобразовать его в URL профиля Steam
    через Steam Web API.
    Возвращает URL профиля, False если vanity URL не найден, или None в случае ошибки API.
    """
    print(f"🔍 Проверка vanity URL: '{vanity_url_input}' через API...")
    try:
        user_id_response = api_parser.resolve_vanity_url(vanity_url_input)
        if user_id_response:
            print(f"✅ Vanity URL '{vanity_url_input}' успешно разрешен в SteamID64: {user_id_response}")
            return f"https://steamcommunity.com/profiles/{user_id_response}"
        else:
            print(f"❌ Vanity URL '{vanity_url_input}' не найден через API.")
            return False
    except Exception as e:
        print(f"⚠️ Ошибка при проверке vanity URL через API: {e}")
        return None


with gr.Blocks(theme=gr.themes.Soft(), title="🎮 Steam Game Recommender") as iface:
    # Заголовок
    gr.Markdown("""
    # 🎮 Персональный рекомендатель игр Steam
    *Получайте персонализированные рекомендации на основе вашей игровой библиотеки или любимых игр*
    """)

    # Блок ввода
    with gr.Row():
        input_box = gr.Textbox(
            label="Введите запрос:",
            placeholder="Ссылка на профиль Steam, SteamID64, название игры...",
            lines=1,
            max_lines=1,
            container=False
        )
        submit_btn = gr.Button("Анализировать", variant="primary")

    # Примеры запросов
    gr.Examples(
        examples=[
            ["Kseoni4"],
            ["76561197992495897"],
            ["Stellaris"],
            ["https://store.steampowered.com/app/730"]
        ],
        inputs=input_box,
        label="Примеры запросов:",
        examples_per_page=3
    )

    # Блок вывода
    output_box = gr.Textbox(
        label="Результаты анализа:",
        interactive=False,
        lines=15,
        container=False,
        show_copy_button=True
    )

    # Инструкция
    with gr.Accordion("ℹ️ Как использовать", open=False):
        gr.Markdown("""
        **Для анализа игровой библиотеки:**
        - Вставьте ссылку на профиль Steam (профиль должен быть публичным)
        - Используйте SteamID64 (17-значное число)
        - Введите имя профиля (vanity URL)

        **Для рекомендаций по конкретной игре:**
        - Введите точное название игры
        - Вставьте ссылку на страницу игры в Steam Store

        **Особенности работы:**
        - Анализ выполняется с помощью ML-модели на основе описаний игр
        - Рекомендации обновляются при каждом новом запросе
        - Для большей точности используйте прямые ссылки
        """)

    # Обработчик событий
    submit_btn.click(
        fn=analyze_input,
        inputs=input_box,
        outputs=output_box,
        api_name="analyze"
    )


if __name__ == "__main__":
    iface.launch()