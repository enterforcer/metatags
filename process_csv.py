import os
import pandas as pd
import time
import torch
from transformers import pipeline

# ------------------------- Параметры -------------------------
csv_path = "/content/metatags1/provided_file.csv"  # Обновите, если нужно
output_file = "/content/products_with_generated_text.csv"
model_id = "distilgpt2"  # Используем модель distilgpt2

# ------------------------- Проверка наличия CSV-файла -------------------------
if not os.path.exists(csv_path):
    print(f"❌ Ошибка: CSV-файл не найден: {csv_path}")
    exit()
else:
    print(f"✅ CSV-файл найден: {csv_path}")

# ------------------------- Загрузка модели distilgpt2 -------------------------
device = 0 if torch.cuda.is_available() else -1
print("Используемый девайс:", "GPU" if device == 0 else "CPU")
print("Загружаю модель distilgpt2 из Hugging Face Hub...")

try:
    generator = pipeline("text-generation", model=model_id, device=device)
    print(f"✅ Модель '{generator.model.config._name_or_path}' успешно загружена.")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")
    exit()

print(f"Используемая модель: {generator.model.config._name_or_path}")
print(f"Используемый девайс: {'GPU' if device == 0 else 'CPU'}")

# ------------------------- Обновленный промпт для генерации Title и Description -------------------------
def generate_text(product_name):
    """
    Генерирует Title и Description для товара.
    
    Требования:
      - Title: не более 70 символов, кратко отражает суть товара и содержит ключевые слова, без непроверяемых обещаний.
      - Description: от 140 до 180 символов, релевантное содержимое, содержит ключевые слова в начале, уникальное, понятное, с призывом к действию.
    
    Формат ответа:
      Title: <текст>
      Description: <текст>
      
    Если за 3 попытки не удалось получить корректный формат, возвращает ('', '').
    """
    prompt = (
Ты — профессиональный копирайтер с опытом написания эффективных описаний для промышленных товаров. Твоя задача: создать корректные Title и Description для товара, который называется: "<название товара>".

**Важно**: 
1) Title (70 символов максимум) — должен кратко отражать суть товара, упомянуть ключевые слова и не содержать лестных обещаний.
2) Description (140–180 символов). Соблюдай:
   • Релевантность: чёткое соответствие реальным свойствам товара.
   • Объём: 140–180 символов без выдумок и неподтверждённых фраз.
   • Ключевые слова: употребляй в начале. Используй только проверенную информацию.
   • Уникальность: избегай заимствованных оборотов с чужих сайтов.
   • Призыв к действию: побуждай пользователя узнать подробности.
   • Понятность: избегай лишних спецсимволов и громких лозунгов.

**Финальный формат ответа**:
Title: <краткий заголовок без вводящих в заблуждение обещаний>

Description: <описание в 140–180 символов с ключевыми словами >

    )
    
    for attempt in range(3):
        try:
            print(f"▶️ Попытка {attempt + 1} для товара: '{product_name}'")
            result = generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                truncation=True,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )[0]["generated_text"]
            
            if "Title:" in result and "Description:" in result:
                title = result.split("Title:")[1].split("Description:")[0].strip()
                description = result.split("Description:")[1].strip()
                return title, description
            
            print(f"⚠️ Некорректный формат ответа, попытка {attempt + 1}.")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Ошибка генерации для '{product_name}' (попытка {attempt + 1}): {e}")
            time.sleep(1)
    return "", ""

# ------------------------- Обработка товаров -------------------------
def process_products(df):
    """
    Обрабатывает товары из DataFrame пакетами, генерируя Title и Description.
    """
    titles = []
    descriptions = []
    total = len(df)
    
    for idx, product in enumerate(df["NAME_"]):
        print(f"⚙️ Обработка товара {idx + 1}/{total}: {product}")
        title, desc = generate_text(product)
        if not title and not desc:
            print(f"❌ Не удалось сгенерировать данные для товара: {product}")
            titles.append("")
            descriptions.append("")
        else:
            titles.append(title)
            descriptions.append(desc)
        time.sleep(0.5)
    
    df["Title"] = titles
    df["Description"] = descriptions
    return df

# ------------------------- Основной процесс -------------------------
def main():
    """
    1. Считывает CSV-файл.
    2. Генерирует Title и Description для каждого товара.
    3. Сохраняет результат в CSV.
    """
    df = pd.read_csv(csv_path, header=0, encoding="utf-8-sig", sep=";")
    if "NAME_" not in df.columns:
        print("❌ Ошибка: В CSV-файле отсутствует столбец 'NAME_'.")
        return
    
    print(f"📂 Файл загружен, товаров: {len(df)}")
    df_processed = process_products(df)
    df_processed.to_csv(output_file, index=False, encoding="utf-8-sig", sep=";")
    print(f"✅ Обработка завершена. Результат сохранён в '{output_file}'.")

if __name__ == "__main__":
    main()
