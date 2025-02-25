# -*- coding: utf-8 -*-

import os
import pandas as pd
import time
import torch
from transformers import pipeline

# Определяем устройство: GPU, если доступно, иначе CPU
device = 0 if torch.cuda.is_available() else -1

print("Загружаю модель distilgpt2...")
try:
    # Используем distilgpt2 вместо GPT-Neo
    generator = pipeline("text-generation", model="distilgpt2", device=device)
    print(f"Модель '{generator.model.config._name_or_path}' успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit()

print(f"Используемая модель: {generator.model.config._name_or_path}")
print(f"Используемый девайс: {'GPU' if device == 0 else 'CPU'}")

def generate_text(product_name):
    """
    Генерирует Title и Description для товара с учётом новых требований:
    - Длина description: 140-180 символов
    - Ключевые слова в начале
    - Уникальность, призыв к действию
    - Без лишних восклицаний, заглавных букв и "сложных" оборотов
    """
    prompt = (
        f"Ты - профессиональный копирайтер, специализирующийся на описаниях для промышленных товаров. "
        f"Тебе нужно написать Title и Description для товара с названием: \"{product_name}\".\n\n"
        f"Title (до 70 символов) и Description (140-180 символов), "
        f"учитывая ключевые слова в начале и призыв к действию, "
        f"без необоснованных обещаний и рекламных лозунгов. "
        f"Формат ответа:\nTitle: <текст>\nDescription: <текст>"
    )

    for attempt in range(3):
        try:
            print(f"Попытка {attempt + 1} для товара: '{product_name}'")
            result = generator(
                prompt,
                max_new_tokens=120,     # немного увеличили, чтобы хватило на description
                num_return_sequences=1,
                truncation=True,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )[0]["generated_text"]

            # Проверяем наличие требуемых ключевых слов "Title:" и "Description:"
            if "Title:" in result and "Description:" in result:
                # Извлекаем Title и Description
                title_part = result.split("Title:")[1].split("Description:")[0].strip()
                description_part = result.split("Description:")[1].strip()
                return title_part, description_part

            print(f"Формат ответа некорректен, попытка {attempt + 1}/3...")
            time.sleep(1)
        except Exception as e:
            print(f"Ошибка генерации (попытка {attempt + 1}): {e}")
            time.sleep(1)

    # Если не удалось получить корректный ответ
    return "", ""

def process_products(df):
    titles = []
    descriptions = []
    total = len(df)

    for idx, product in enumerate(df["NAME_"]):
        print(f"Обработка товара {idx+1}/{total}: {product}")
        title, desc = generate_text(product)
        if not title and not desc:
            print(f"Не удалось обработать товар: {product}. Прерываю процесс.")
            break
        titles.append(title)
        descriptions.append(desc)
        time.sleep(0.5)

    # Возвращаем только ту часть DataFrame, которая успешно обработана
    df = df.iloc[:len(titles)]
    df["Title"] = titles
    df["Description"] = descriptions
    return df

def main():
    input_file = "provided_file.csv"
    output_file = "products_with_generated_text.csv"

    if not os.path.exists(input_file):
        print(f"Ошибка: Файл '{input_file}' не найден!")
        return

    df = pd.read_csv(input_file, header=0, encoding="utf-8-sig", dtype=str, sep=";")
    if "NAME_" not in df.columns:
        print("Ошибка: Нет столбца 'NAME_'! Проверьте структуру данных.")
        return

    print(f"Файл загружен, количество товаров: {len(df)}")
    df_processed = process_products(df)
    df_processed.to_csv(output_file, index=False, encoding="utf-8-sig", sep=";")
    print(f"Обработка завершена. Результат сохранён в '{output_file}'.")

if __name__ == "__main__":
    main()
