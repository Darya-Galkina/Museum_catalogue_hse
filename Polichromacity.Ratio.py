import cv2
import numpy as np
import requests
from io import BytesIO
import pandas as pd
import os
import time
from skimage.color import rgb2hsv

def calculate_colorfulness(image_url, num_colors=12, count_threshold=0.05, max_retries=3):
    """Оценивает пестроту изображения, используя 12 невербализованных категорий цветов."""
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()

            image_bytes = BytesIO(response.content)
            img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Не получается декодировать изображение: {image_url}")

            hsv_img = rgb2hsv(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            hue_channel = hsv_img[:, :, 0]
            saturation_channel = hsv_img[:, :, 1]

            num_pixels = np.prod(hsv_img.shape[:2])
            if num_pixels == 0:
                return 0

            hue_channel = hue_channel.flatten()
            saturation_channel = saturation_channel.flatten()

            # Убрать пиксели с низкой насыщенностью, у них сложно определить цвет
            filtered_hue_values = hue_channel[saturation_channel >= 0.2]
            if len(filtered_hue_values) == 0:
                return 0
            
            # Значения hue (от 0 до 1) цикличны, и красные оттенки режутся и оказываются сразу около 0 и около 1. Значение синих оттенов начинаются с 0.55, поэтому если мы сместим круг на полоборота, по синий и красный не будут разрезаться при делении на фрагменты
            shifted_hue_values = (filtered_hue_values + 0.5) % 1
            
            # Разделим спектр на 12 частей (условно получим 12 оттенков)
            color_categories = np.linspace(0, 1, num_colors + 1)
            color_categories = [round(c, 2) for c in color_categories]

            color_counts = [0] * num_colors

            for hue in shifted_hue_values:
                for i in range(num_colors):
                  if color_categories[i] <= hue < color_categories[i+1]:
                      color_counts[i] += 1
                      break # перемещаемся на следующее значение hue

            # оценим долю цветных элементов
            total_pixels_used = len(filtered_hue_values)
            count_threshold_value = total_pixels_used * count_threshold
            
            used_color_count = np.sum(np.array(color_counts) > count_threshold_value)
            colorfulness_score =  used_color_count / num_colors
            
            return colorfulness_score

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Error downloading image from URL: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Error processing image: {e}")
            time.sleep(2)
    return np.nan


def process_images(df, batch_size=10, start_row=0):
    """Обрабатывает изображения и добавляет результаты в DataFrame с сохранением по частям."""
    num_rows = len(df)
    for i in range(start_row, num_rows, batch_size):
        batch_end = min(i + batch_size, num_rows)
        print(f"Processing rows: {i} to {batch_end-1}")

        for index in range(i, batch_end):
          if df.at[index, "Object.Name"] == "Photograph":
            df.at[index, "Polichromacity.Ratio"] = np.nan
            print(f"Skipping Photograph at row: {index}")
          elif pd.isna(df.at[index, "Polichromacity.Ratio"]):
            url = df.at[index, "Link.Image"]
            if pd.notna(url):
              colorfulness = calculate_colorfulness(url)
              df.at[index, "Polichromacity.Ratio"] = colorfulness
          else:
              print(f"Skipping row {index} because it has Polichromacity.Ratio = {df.at[index, 'Polichromacity.Ratio']}")

        df.to_csv(os.path.join(work_dir, 'joined.csv'), index=False)
        print(f"Saved updated CSV, completed {batch_end} out of {num_rows}")
        time.sleep(1) 
    return df


if __name__ == '__main__':

    work_dir = r"D:\Desktop\PUSHKA\csvs"
    os.chdir(work_dir)


    try:
        df = pd.read_csv('joined.csv')
        
  
        if 'Polichromacity.Ratio' not in df.columns:
           df['Polichromacity.Ratio'] = None


        start_row = 0
        if 'Polichromacity.Ratio' in df.columns and df['Polichromacity.Ratio'].notna().any():
          start_row = df[df['Polichromacity.Ratio'].notna()].index.max() + 1
          print(f"Resuming processing from row: {start_row}")
           
    except FileNotFoundError:
        print(f"Error: 'joined.csv' not found in {work_dir}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()


    df = process_images(df, batch_size=10, start_row=start_row)

    print("Finished processing all images.")