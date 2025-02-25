import os
import pandas as pd
import time
import torch
from transformers import pipeline

# Определяем, есть ли доступ к CUDA (GPU); если нет — используем CPU
device = 0 if torch.cuda.is_available() else -1

print("Загружаю модель GPT-Neo 2.7B...")
try:
    # Загружаем модель GPT-Neo 2.7B
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=device)
    print(f"Модель '{generator.model.config._name_or_path}' успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")

print(f"Используемая модель: {generator.model.config._name_or_path}")
print(f"Используемый девайс: {'GPU' if device == 0 else 'CPU'}")

def generate_text(product_name):
    """
    Генерация «Тега» и «Описание» на основе названия товара.
    Возвращает кортеж (tag, description). Если за 3 попытки не удалось получить корректный формат, возвращает ('', '').
    """
    prompt = (
        f"Ты — профессиональный копирайтер с опытом создания продающих описаний для промышленного оборудования. "
        f"Напиши привлекательный и уникальный тег и описание для товара, название которого: \"{product_name}\".\n\n"
        f"Тег должен быть коротким (не более 15 слов), подчёркивать главное преимущество товара и быть понятным для покупателя.\n"
        f"Описание должно содержать от 20 до 50 слов, давать конкретику по применению, качеству и особенностям товара, "
        f"без клише и завышенных обещаний.\n\n"
        f"Ответ должен быть в формате:\nТег: <текст>\nОписание: <текст>"
    )

    for attempt in range(3):
        try:
            print(f"▶️ Попытка {attempt + 1} для товара: '{product_name}'")
            result = generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                truncation=True,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )[0]["generated_text"]

            if "Тег:" in result and "Описание:" in result:
                tag = result.split("Тег:")[1].split("Описание:")[0].strip()
                description = result.split("Описание:")[1].strip()
                return tag, description

            print(f"⚠️ Некорректный формат ответа (попытка {attempt + 1}/3). Пробуем ещё раз.")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Ошибка генерации (попытка {attempt + 1}): {e}")
            time.sleep(1)

    return "", ""

def process_products(df):
    """
    Обрабатывает все товары из DataFrame, создавая столбцы 'Tag' и 'Description'.
    Возвращает обновлённый DataFrame.
    """
    tags = []
    descriptions = []
    total = len(df)

    for idx, product in enumerate(df["NAME_"]):
        print(f"⚙️ Обработка товара {idx + 1}/{total}: {product}")
        tag, desc = generate_text(product)
        if tag == "" and desc == "":
            print(f"❌ Не удалось обработать товар: {product}. Прерываю процесс.")
            break
        tags.append(tag)
        descriptions.append(desc)
        time.sleep(0.5)

    df = df.iloc[:len(tags)]
    df["Tag"] = tags
    df["Description"] = descriptions
    return df

def main():
    """
    Основной процесс:
      1. Считывает CSV-файл 'provided_file.csv'.
      2. Генерирует теги и описания.
      3. Сохраняет результат в 'products_with_generated_text.csv'.
    """
    input_file = "provided_file.csv"
    output_file = "products_with_generated_text.csv"

    if not os.path.exists(input_file):
        print(f"❌ Ошибка: Файл '{input_file}' не найден!")
        return

    df = pd.read_csv(input_file, header=0, encoding="utf-8-sig", dtype=str, sep=";")
    if "NAME_" not in df.columns:
        print("❌ Ошибка: В файле нет столбца 'NAME_'! Проверьте структуру данных.")
        return

    print(f"📂 Файл загружен, количество товаров для обработки: {len(df)}")
    df_processed = process_products(df)
    df_processed.to_csv(output_file, index=False, encoding="utf-8-sig", sep=";")
    print(f"✅ Обработка завершена. Результат сохранён в '{output_file}'.")

    # Для локального запуска на Windows можно вручную открыть проводник и найти файл,
    # либо запустить команду, если у вас настроены утилиты для скачивания:
    # from google.colab import files
    # files.download(output_file)

if __name__ == "__main__":
    main()
