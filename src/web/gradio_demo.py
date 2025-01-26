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

    # **ПЕРЕДЕРЖИВАЕМ ПРОВЕРКУ НАЗВАНИЯ ИГРЫ ПЕРЕД VANITY URL**
    print("⚙️ Попытка обработки как названия игры.")
    game_identifier = user_input
    recommendations_output = analyzer.analyze_single_game_for_gradio(game_identifier)
    if recommendations_output != "❌ Не удалось получить рекомендации для указанной игры.": # Проверяем, что это не сообщение об ошибке
        print("✅ Распознано как запрос одиночной игры по названию.")
        return recommendations_output

    # Проверка на vanity URL через API (теперь только если не распознано как игра)
    vanity_resolution_result = validate_vanity_url(user_input, analyzer.api_parser)
    if vanity_resolution_result:
        if isinstance(vanity_resolution_result, str): # Успешное разрешение vanity URL
            steam_user_url = vanity_resolution_result
            print("⚙️ Обнаружен запрос библиотеки пользователя (vanity URL), подтвержден через API.")
            return analyzer.run_analysis_for_gradio(steam_user_url)
        elif vanity_resolution_result is False: # Vanity URL не найден, но мы уже попробовали как название игры выше
            pass # Просто проваливаемся дальше, к общему сообщению об ошибке
        else: # vanity_resolution_result is None - ошибка API
            return "❌ Ошибка при проверке vanity URL через API. Пожалуйста, попробуйте позже."

    # Если ни один из типов не распознан, и не найдена игра по названию и vanity URL не валиден
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
            return False # Возвращаем False, если vanity URL не найден
    except Exception as e:
        print(f"⚠️ Ошибка при проверке vanity URL через API: {e}")
        return None # Возвращаем None в случае ошибки API


iface = gr.Interface(
    fn=analyze_input,
    inputs=gr.Textbox(
        lines=1,
        placeholder="Введите ссылку на профиль Steam, SteamID64, vanity URL или название игры",
        label="🔍 Введите запрос для получения рекомендаций:"
    ),
    outputs=gr.Textbox(lines=15, label="✨ Рекомендации:"),
    title="🎮 Персональный рекомендатель игр Steam",
    description="""
    Добро пожаловать в персональный рекомендатель игр Steam! 🚀
    Этот инструмент поможет вам получить рекомендации по играм, основываясь на вашей библиотеке Steam или на основе выбранной вами игры.

    **Как использовать:**

    1.  **Для получения рекомендаций на основе вашей библиотеки Steam:**
        *   Введите **ссылку на ваш профиль Steam**. Убедитесь, что ваш профиль публичный.
            *   Примеры:
                *   `https://steamcommunity.com/id/имя_вашего_профиля/`
                *   `https://steamcommunity.com/profiles/7656119XXXXXXXXXXXX/`
                *   `steamcommunity.com/id/имя_вашего_профиля/`
                *   `steamcommunity.com/profiles/7656119XXXXXXXXXXXX`
        *   Или введите ваш **SteamID64** (17-значное число).
            *   Пример: `7656119XXXXXXXXXXXX`
        *   Или введите ваш **vanity URL** (имя профиля).
            *   Пример: `имя_вашего_профиля`

    2.  **Для получения рекомендаций на основе выбранной игры:**
        *   Введите **название игры**.
            *   Примеры: `Stellaris`, `Cyberpunk 2077`, `Portal 2`
        *   Или введите **ссылку на страницу игры в Steam Store**.
            *   Примеры:
                *   `https://store.steampowered.com/app/730`
                *   `store.steampowered.com/app/440`

    **Примечания:**
    *   Рекомендации основаны на анализе описаний игр и используют модель машинного обучения для определения схожести игр.
    *   Для анализа библиотеки Steam, ваш профиль должен быть публичным, чтобы инструмент мог получить список ваших игр.
    *   Если возникают ошибки при проверке vanity URL, попробуйте ввести SteamID64 или URL профиля напрямую.
    """,
    examples=[
        ["Kseoni4"],
        ["76561197992495897"],
        ["Stellaris"]
    ],
    cache_examples=True
)

if __name__ == "__main__":
    iface.launch()