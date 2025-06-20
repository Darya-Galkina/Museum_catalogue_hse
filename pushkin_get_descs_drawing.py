import os
import pandas as pd
import time
from tqdm import tqdm
import undetected_chromedriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Чтение CSV файла
def read_csv(file_path):
    return pd.read_csv(file_path)

# Добавление данных в существующий CSV файл
def append_to_csv(data, file_path, header=False):
    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', index=False, header=header)

# Функция для получения описания по ссылке
def selenium_crawl(link_to_image):
    res_text = ""
    driver = None
    try:
        # Открываем браузер и сайт
        driver = webdriver.Chrome()
        driver.get("https://visionbot.ru/index.php")
    
        # Находим поле для ввода данных
        input_link = driver.find_element(By.ID, "userlink")
        input_link.send_keys(link_to_image)
        
        # Находим кнопку для отправки
        submit = driver.find_element(By.XPATH, "//*[@id='btn2']")
        ActionChains(driver).move_to_element(submit).perform()
        submit.click()
        submit.click()  # Дважды на всякий случай
        
        # Ожидаем результата
        result = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//*[@id='success1']"))
        )
        res_text = result.text
    except TimeoutException:
        print("Timeout occurred while waiting for the page to load.")
    finally:
        if driver:
            driver.quit()  # Закрываем браузер
        time.sleep(2)  # Пауза перед открытием нового браузера
    return res_text

# Основная функция
def process_links(input_file, output_file):
    # Загружаем исходный файл
    df = read_csv(input_file)

    # Получаем ссылки на изображения
    links_to_images = df['Link.Image'].tolist()

    # Проверяем, существует ли уже файл с результатами
    file_exists = os.path.exists(output_file)

    # Если файл существует, загружаем уже обработанные данные, чтобы избежать повторной обработки
    if file_exists:
        processed_df = read_csv(output_file)
        processed_ids = set(processed_df['Object.ID'].tolist())
    else:
        processed_ids = set()

    # Обрабатываем каждую строку
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        object_id = row['Object.ID']
        link = row['Link.Image']

        # Пропускаем уже обработанные строки
        if object_id in processed_ids:
            continue

        nodata = True
        attempts = 0
        max_attempts = 5
        description = ""

        while nodata and attempts < max_attempts:
            time.sleep(1.5)
            res = selenium_crawl(link)
            if len(res) > 0:
                description = res
                nodata = False
            else:
                attempts += 1
                print(f"No data received for link #{idx + 1}, trying again ({attempts}/{max_attempts})")

        # Если после всех попыток описание не получено
        if nodata:
            print(f"Failed to process link #{idx + 1} after {max_attempts} attempts.")

        # Добавляем результат в файл
        row['Description'] = description
        append_to_csv([row], output_file, header=not file_exists)
        file_exists = True  # Указываем, что файл теперь существует

# Параметры
input_file = 'drawing.csv'
output_file = 'drawing_with_descriptions.csv'

# Запуск обработки для всего файла
process_links(input_file, output_file)

print(f"Готово! Все строки обработаны. Результаты добавляются в файл '{output_file}' во время работы.")
