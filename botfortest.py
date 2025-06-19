
import telebot
from telebot import types
import requests
from io import BytesIO
import pandas as pd
import random
from elasticsearch import Elasticsearch
import io, json, os, re, torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Инициализация бота
bot = telebot.TeleBot('УНИКАЛЬНЫЙ_ТОКЕН')

# Загрузка данных из CSV файла
df = pd.read_csv('joined_trans.csv')

# Инициализация Elasticsearch
es_host = "http://localhost:9200"
es_index = "metadata"
es = Elasticsearch(hosts=[es_host])

# Словарь для хранения уже показанных изображений для каждого пользователя
shown_images = {}

# Словарь для хранения результатов поиска и текущего индекса для каждого пользователя
search_results = {}

# Словарь для хранения результатов поиска похожих картин
similar_results = {}

# Словарь для хранения последнего типа поиска для каждого пользователя
last_search_type = {}

# Словарь для хранения последней показанной картины для каждого пользователя
last_shown_image = {}

# Загрузка данных для поиска по описанию
with open("titleslinks_descs_23_02.json", "r", encoding="utf-8") as file:
    links_descs = json.load(file)

vecs = torch.load("tensor_4.pt")
matrix = np.vstack(vecs)

links_vecs = {}  # Словарь ссылки — векторы
for link, vec in zip(links_descs.keys(), vecs):
    links_vecs[link] = vec

# Инициализация модели для обработки текста
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

# Функция для поиска картины по цвету
def find_image_by_color(color, user_id):
    matching_images = []  # Список для хранения всех подходящих картин
    for index, row in df.iterrows():
        if pd.notna(row['color']) and color.lower() in row['color'].lower():
            if user_id not in shown_images or row['Link.Image'] not in shown_images[user_id]:
                matching_images.append((row['Link.Image'], row))  # Добавляем картину и её данные в список
    if matching_images:
        return random.choice(matching_images)  # Возвращаем случайную картину из списка
    return None, None

# Функция для загрузки изображения по URL
def get_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        photo = BytesIO(response.content)
        return photo
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

# Функция для поиска по автору
def search_by_author(message):
    author = message.text
    chat_id = message.chat.id
    bot.send_message(message.chat.id, f'Ищу экспонат по запросу {message.text}')
    try:
        query = {"match": {"artist": author}}
        res = es.search(index=es_index, query=query)
        results = res["hits"]["hits"]

        if results:
            search_results[chat_id] = {"results": results, "index": 0}
            send_painting(chat_id)
        else:
            bot.send_message(chat_id, f"Автор '{author}' не найден.")

    except Exception as e:
        print(f"Elasticsearch Error: {e}")
        bot.send_message(chat_id, "Произошла ошибка с Elasticsearch.")

# Функция для поиска по названию
def search_by_title(message):
    title = message.text
    chat_id = message.chat.id
    bot.send_message(message.chat.id, f'Ищу экспонат по запросу "{message.text}"')
    try:
        query = {"match": {"title": title}}
        res = es.search(index=es_index, query=query)
        results = res["hits"]["hits"]

        if results:
            search_results[chat_id] = {"results": results, "index": 0}
            send_painting(chat_id)
        else:
            bot.send_message(chat_id, f"Картина с названием '{title}' не найдена.")

    except Exception as e:
        print(f"Elasticsearch Error: {e}")
        bot.send_message(chat_id, "Произошла ошибка с Elasticsearch.")

# Функция для отправки картины
def send_painting(chat_id):
    """Отправляет одну картину с кнопкой 'Ещё' и 'Меню'."""
    if chat_id in search_results:
        results = search_results[chat_id]["results"]
        index = search_results[chat_id]["index"]

        if 0 <= index < len(results):
            hit = results[index]
            painting = hit["_source"]
            object_id = painting.get('object_id')
            title = painting.get('title')
            year = painting.get('year', "Год создания не указан")
            artist = painting.get('artist')
            artist_birth = painting.get('artist_birth', "Дата рождения автора не указана")
            artist_death = painting.get('artist_death', "Дата смерти автора не указана")
            medium = painting.get('medium', "Материал не указан")
            wiki_url = painting.get('wiki_url', None)
            image_url = painting.get('image_url')

            caption = (
                f"<b>{title}</b>\n"
                f"Год создания: {year}\n"
                f"Автор: {artist} ({artist_birth} - {artist_death})\n"
                f"Материал: {medium}\n"
                f"Дополнительные сведения: {wiki_url}"
            )

            photo = get_image_from_url(image_url)

            if photo:
                markup = types.InlineKeyboardMarkup()
                if index < len(results) - 1:
                    button_more1 = types.InlineKeyboardButton('Ещё', callback_data=f'more_{chat_id}')
                    markup.add(button_more1)
                button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
                markup.add(button_menu)

                bot.send_photo(chat_id, photo, caption=caption, parse_mode='html', reply_markup=markup)
                search_results[chat_id]["index"] += 1
                # Сохраняем последнюю показанную картину
                last_shown_image[chat_id] = image_url
            else:
                bot.send_message(chat_id, f"Не удалось загрузить изображение экспоната '{title}'.")
        else:
            bot.send_message(chat_id, "Больше нет результатов.")
            del search_results[chat_id]

    else:
        bot.send_message(chat_id, "Сначала выполните поиск.")

# Функция для поиска по описанию
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Первый элемент model_output содержит все эмбеддинги токенов
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def index_query(query):
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
    else:
        query_str = str(query)
    encoded_input = tokenizer(query_str, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return query_str, embedding

def result_search(corpus, corpus_mtrx_, query_vec_):
    scores = np.dot(corpus_mtrx_, query_vec_.T)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top = np.array(list(corpus.keys()))[sorted_scores_indx.ravel()[:15]]
    return top

# Обработчик команды /help
@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, '<b>Команды бота</b>\n /start\n /help\n /menu', parse_mode='html')

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def menu(message):
    markup = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton('Поиск по названию', callback_data='search1')
    button2 = types.InlineKeyboardButton('Поиск по автору', callback_data='search2')
    button5 = types.InlineKeyboardButton('Поиск по описанию', callback_data='search5')
    button4 = types.InlineKeyboardButton('Поиск по цвету', callback_data='search4')
    markup.row(button1, button2)
    markup.row(button5)
    markup.row(button4)
    bot.send_message(message.chat.id, 'Выберите одну из доступных опций', reply_markup=markup)

# Функция для поиска по цвету
def search_by_color(message):
    color = message.text.strip()
    user_id = message.chat.id
    last_search_type[user_id] = {'type': 'color', 'query': color}  # Сохраняем тип поиска и запрос
    image_url, image_data = find_image_by_color(color, user_id)
    if image_url:
        # Добавляем изображение в список показанных для этого пользователя
        if user_id not in shown_images:
            shown_images[user_id] = []
        shown_images[user_id].append(image_url)
        # Сохраняем последнюю показанную картину
        last_shown_image[user_id] = image_url
        # Отправляем изображение
        response = requests.get(image_url)
        if response.status_code == 200:
            photo = BytesIO(response.content)
            bot.send_photo(message.chat.id, photo)
            # Формируем текст с метаданными
            metadata = f"Название: {image_data['Title_tranlated']}\nДата создания: {image_data['Object.End.Date']}\nАвтор: {image_data['Artist.Alpha.Sort_translated']}\nТехника: {image_data['Medium_translated']}"
            bot.send_message(message.chat.id, metadata)
            # Добавляем кнопки "Еще", "Ввести новый цвет" и "Найти похожие"
            markup = types.InlineKeyboardMarkup()
            button_more1 = types.InlineKeyboardButton('Еще', callback_data='more')
            button_new_color = types.InlineKeyboardButton('Ввести новый цвет', callback_data='new_color')
            button_similar = types.InlineKeyboardButton('Найти похожие', callback_data='find_similar')
            button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
            markup.row(button_more1)
            markup.row(button_new_color)
            markup.row(button_similar)
            markup.row(button_menu)
            bot.send_message(message.chat.id, 'Хотите найти еще одну картину с этим цветом?', reply_markup=markup)
        else:
            bot.send_message(message.chat.id, 'Не удалось загрузить изображение.')
    else:
        # Если картины не найдены, предлагаем ввести другой цвет
        markup = types.InlineKeyboardMarkup()
        button_new_color = types.InlineKeyboardButton('Ввести новый цвет', callback_data='new_color')
        markup.add(button_new_color)
        bot.send_message(message.chat.id, 'Картины с таким цветом не найдены. Хотите ввести другой цвет?', reply_markup=markup)

# Функция для обработки текстового запроса
@bot.message_handler()
def test_photo(message):
    bot.send_message(message.chat.id, f'Ищу картину по запросу {message.text}')
    query = message.text
    user_id = message.chat.id
    last_search_type[user_id] = {'type': 'description', 'query': query}  # Сохраняем тип поиска и запрос
    
    query_string, query_vec = index_query(query)
    query_answer = result_search(links_descs, matrix, query_vec)

    # Проверяем, есть ли результаты поиска
    if len(query_answer) > 0:
        # Фильтруем результаты, чтобы исключить уже показанные изображения
        if user_id not in shown_images:
            shown_images[user_id] = []

        # Оставляем только те изображения, которые еще не были показаны
        available_images = [img for img in query_answer if img not in shown_images[user_id]]

        if len(available_images) > 0:
            # Выбираем случайную картину из доступных
            random_image = random.choice(available_images)
            shown_images[user_id].append(random_image)  # Добавляем изображение в список показанных
            # Сохраняем последнюю показанную картину
            last_shown_image[user_id] = random_image

            # Отправляем изображение
            bot.send_photo(message.chat.id, random_image)

            # Добавляем метаданные
            try:
                # Находим строку с данными о картине
                image_data = df[df['Link.Image'] == random_image].iloc[0]
                metadata = f"Название: {image_data['Title_tranlated']}\nДата создания: {image_data['Object.End.Date']}\nАвтор: {image_data['Artist.Alpha.Sort_translated']}\nТехника: {image_data['Medium_translated']}"
                bot.send_message(message.chat.id, metadata)
            except:
                bot.send_message(message.chat.id, "Метаданные для этой картины не найдены")

            # Добавляем кнопки "Еще", "Ввести новый запрос" и "Найти похожие"
            markup = types.InlineKeyboardMarkup()
            button_more2 = types.InlineKeyboardButton('Еще', callback_data='more')
            button_new_query = types.InlineKeyboardButton('Ввести новый запрос', callback_data='new_query')
            button_similar = types.InlineKeyboardButton('Найти похожие', callback_data='find_similar')
            button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
            markup.row(button_more2)
            markup.row(button_new_query)
            markup.row(button_similar)
            markup.row(button_menu)
            bot.send_message(message.chat.id, 'Хотите найти еще одну картину по этому запросу?', reply_markup=markup)
        else:
            # Если все изображения уже показаны, предлагаем ввести новый запрос
            markup = types.InlineKeyboardMarkup()
            button_new_query = types.InlineKeyboardButton('Ввести новый запрос', callback_data='new_query')
            markup.add(button_new_query)
            bot.send_message(message.chat.id, 'Все подходящие картины уже показаны. Хотите ввести новый запрос?', reply_markup=markup)
    else:
        bot.send_message(message.chat.id, 'Картины по вашему запросу не найдены.')

# Обработчик нажатий на кнопки
@bot.callback_query_handler(func=lambda callback: True)
def callback_message(callback):
    user_id = callback.message.chat.id
    
    if callback.data == 'search1':
        bot.send_message(callback.message.chat.id, 'Введите название картины')
        bot.register_next_step_handler(callback.message, search_by_title)
    elif callback.data == 'search2':
        bot.send_message(callback.message.chat.id, 'Введите автора картины')
        bot.register_next_step_handler(callback.message, search_by_author)
    elif callback.data == 'search5':
        bot.send_message(callback.message.chat.id, 'Введите описание картины')
        bot.register_next_step_handler(callback.message, test_photo)
    elif callback.data == 'more':
        # Обработка кнопки "Еще" - продолжаем предыдущий поиск
        if user_id in last_search_type:
            search_type = last_search_type[user_id]['type']
            query = last_search_type[user_id]['query']
            
            if search_type == 'color':
                # Повторяем поиск по цвету
                image_url, image_data = find_image_by_color(query, user_id)
                if image_url:
                    # Добавляем изображение в список показанных
                    shown_images[user_id].append(image_url)
                    # Сохраняем последнюю показанную картину
                    last_shown_image[user_id] = image_url
                    # Отправляем изображение
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        photo = BytesIO(response.content)
                        bot.send_photo(callback.message.chat.id, photo)
                        # Формируем текст с метаданными
                        metadata = f"Название: {image_data['Title_tranlated']}\nДата создания: {image_data['Object.End.Date']}\nАвтор: {image_data['Artist.Alpha.Sort_translated']}\nТехника: {image_data['Medium_translated']}"
                        bot.send_message(callback.message.chat.id, metadata)
                        # Добавляем кнопки
                        markup = types.InlineKeyboardMarkup()
                        button_more = types.InlineKeyboardButton('Еще', callback_data='more')
                        button_new_color = types.InlineKeyboardButton('Ввести новый цвет', callback_data='new_color')
                        button_similar = types.InlineKeyboardButton('Найти похожие', callback_data='find_similar')
                        button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
                        markup.row(button_more)
                        markup.row(button_new_color)
                        markup.row(button_similar)
                        markup.row(button_menu)
                        bot.send_message(callback.message.chat.id, 'Хотите найти еще одну картину с этим цветом?', reply_markup=markup)
                    else:
                        bot.send_message(callback.message.chat.id, 'Не удалось загрузить изображение.')
                else:
                    # Если больше картин не найдено, предлагаем ввести другой цвет
                    markup = types.InlineKeyboardMarkup()
                    button_new_color = types.InlineKeyboardButton('Ввести другой цвет', callback_data='new_color')
                    markup.add(button_new_color)
                    bot.send_message(callback.message.chat.id, 'Больше картин с таким цветом не найдено. Хотите ввести другой цвет?', reply_markup=markup)
            
            elif search_type == 'description':
                # Повторяем поиск по описанию
                query_string, query_vec = index_query(query)
                query_answer = result_search(links_descs, matrix, query_vec)
                
                if len(query_answer) > 0:
                    # Фильтруем результаты
                    available_images = [img for img in query_answer if img not in shown_images[user_id]]
                    
                    if len(available_images) > 0:
                        # Выбираем случайную картину
                        random_image = random.choice(available_images)
                        shown_images[user_id].append(random_image)
                        # Сохраняем последнюю показанную картину
                        last_shown_image[user_id] = random_image
                        
                        # Отправляем изображение
                        bot.send_photo(callback.message.chat.id, random_image)
                        
                        # Добавляем метаданные
                        try:
                            image_data = df[df['Link.Image'] == random_image].iloc[0]
                            metadata = f"Название: {image_data['Title_tranlated']}\nДата создания: {image_data['Object.End.Date']}\nАвтор: {image_data['Artist.Alpha.Sort_translated']}\nТехника: {image_data['Medium_translated']}"
                            bot.send_message(callback.message.chat.id, metadata)
                        except:
                            bot.send_message(callback.message.chat.id, "Метаданные для этой картины не найдены")
                        
                        # Добавляем кнопки
                        markup = types.InlineKeyboardMarkup()
                        button_more = types.InlineKeyboardButton('Еще', callback_data='more')
                        button_new_query = types.InlineKeyboardButton('Ввести новый запрос', callback_data='new_query')
                        button_similar = types.InlineKeyboardButton('Найти похожие', callback_data='find_similar')
                        button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
                        markup.row(button_more)
                        markup.row(button_new_query)
                        markup.row(button_similar)
                        markup.row(button_menu)
                        bot.send_message(callback.message.chat.id, 'Хотите найти еще одну картину по этому запросу?', reply_markup=markup)
                    else:
                        markup = types.InlineKeyboardMarkup()
                        button_new_query = types.InlineKeyboardButton('Ввести новый запрос', callback_data='new_query')
                        markup.add(button_new_query)
                        bot.send_message(callback.message.chat.id, 'Все подходящие картины уже показаны. Хотите ввести новый запрос?', reply_markup=markup)
                else:
                    bot.send_message(callback.message.chat.id, 'Больше картин по этому запросу не найдено.')
        else:
            bot.send_message(callback.message.chat.id, 'Сначала выполните поиск.')
    elif callback.data == 'search4':
        bot.send_message(callback.message.chat.id, 'Введите цвет для поиска')
        bot.register_next_step_handler(callback.message, search_by_color)
    elif callback.data == 'new_color':
        bot.send_message(callback.message.chat.id, 'Введите новый цвет для поиска')
        bot.register_next_step_handler(callback.message, search_by_color)
    elif callback.data == 'new_query':
        bot.send_message(callback.message.chat.id, 'Введите новый запрос для поиска')
        bot.register_next_step_handler(callback.message, test_photo)
    elif callback.data == 'find_similar':
        if user_id in last_shown_image and last_shown_image[user_id] in links_vecs:
            # Ищем похожие картины для последней показанной
            find_res = result_search(links_descs, matrix, links_vecs[last_shown_image[user_id]])
            if len(find_res) > 1:
                # Сохраняем результаты поиска похожих картин
                similar_results[user_id] = find_res

                # Выбираем случайную картину из результатов (кроме самой картины)
                similar_images = [img for img in find_res if img != last_shown_image[user_id]]
                if len(similar_images) > 0:
                    random_image = random.choice(similar_images[:min(10, len(similar_images))])
                    bot.send_photo(callback.message.chat.id, random_image)
                    
                    # Добавляем метаданные
                    try:
                        image_data = df[df['Link.Image'] == random_image].iloc[0]
                        metadata = f"Название: {image_data['Title_tranlated']}\nДата создания: {image_data['Object.End.Date']}\nАвтор: {image_data['Artist.Alpha.Sort_translated']}\nТехника: {image_data['Medium_translated']}"
                        bot.send_message(callback.message.chat.id, metadata)
                    except:
                        bot.send_message(callback.message.chat.id, "Метаданные для этой картины не найдены")

                    # Добавляем кнопки "Еще" и "Меню"
                    markup = types.InlineKeyboardMarkup()
                    button_more_similar = types.InlineKeyboardButton('Еще', callback_data='more_similar')
                    button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
                    markup.row(button_more_similar)
                    markup.row(button_menu)
                    bot.send_message(callback.message.chat.id, 'Хотите найти еще одну похожую картину?', reply_markup=markup)
                else:
                    bot.send_message(callback.message.chat.id, 'Похожие картины не найдены.')
            else:
                bot.send_message(callback.message.chat.id, 'Похожие картины не найдены.')
        else:
            bot.send_message(callback.message.chat.id, 'Сначала выполните поиск.')
    elif callback.data == 'more_similar':
        # Обработка кнопки "Еще" для поиска похожих картин
        if user_id in similar_results and len(similar_results[user_id]) > 1:
            find_res = similar_results[user_id]

            # Выбираем следующую случайную картину из результатов (кроме самой картины)
            similar_images = [img for img in find_res if img != last_shown_image.get(user_id, '')]
            if len(similar_images) > 0:
                random_image = random.choice(similar_images[:min(10, len(similar_images))])
                bot.send_photo(callback.message.chat.id, random_image)
                
                # Добавляем метаданные
                try:
                    image_data = df[df['Link.Image'] == random_image].iloc[0]
                    metadata = f"Название: {image_data['Title_tranlated']}\nДата создания: {image_data['Object.End.Date']}\nАвтор: {image_data['Artist.Alpha.Sort_translated']}\nТехника: {image_data['Medium_translated']}"
                    bot.send_message(callback.message.chat.id, metadata)
                except:
                    bot.send_message(callback.message.chat.id, "Метаданные для этой картины не найдены")

                # Добавляем кнопки "Еще" и "Меню"
                markup = types.InlineKeyboardMarkup()
                button_more_similar = types.InlineKeyboardButton('Еще', callback_data='more_similar')
                button_menu = types.InlineKeyboardButton('Меню', callback_data='menu')
                markup.row(button_more_similar)
                markup.row(button_menu)
                bot.send_message(callback.message.chat.id, 'Хотите найти еще одну похожую картину?', reply_markup=markup)
            else:
                bot.send_message(callback.message.chat.id, 'Больше похожих картин не найдено.')
        else:
            bot.send_message(callback.message.chat.id, 'Больше похожих картин не найдено.')
    elif callback.data == 'menu':
        menu(callback.message)

# Запуск бота
bot.polling(none_stop=True)
