import cv2
import numpy as np
import requests
from io import BytesIO
import pandas as pd
import os
import time

def calculate_contrast_std(image_url, max_retries=3):
    """Вычисляет контрастность изображения по URL с механизмом повторных попыток."""
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, stream=True, timeout=10) 
            response.raise_for_status()

            image_bytes = BytesIO(response.content)
            img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError(f"Could not decode image from URL: {image_url}")
            contrast = np.std(img)
            return contrast

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
          url = df.at[index, "Link.Image"]
          if pd.notna(url):
            contrast = calculate_contrast_std(url)
            df.at[index, "Contrast.Ratio"] = contrast

        df.to_csv(os.path.join(work_dir, 'joined.csv'), index=False)
        print(f"Saved updated CSV, completed {batch_end} out of {num_rows}")
        time.sleep(1) 

    return df

if __name__ == '__main__':

    work_dir = r"D:\Desktop\PUSHKA\csvs"
    os.chdir(work_dir)

    try:
        df = pd.read_csv('joined.csv')
        
        if 'Contrast.Ratio' not in df.columns:
           df['Contrast.Ratio'] = None

        start_row = 0
        if 'Contrast.Ratio' in df.columns and df['Contrast.Ratio'].notna().any():
          start_row = df[df['Contrast.Ratio'].notna()].index.max() + 1
          print(f"Resuming processing from row: {start_row}")
           
    except FileNotFoundError:
        print(f"Error: 'joined.csv' not found in {work_dir}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()
      
    df = process_images(df, batch_size=10, start_row=start_row)

    print("Finished processing all images.")