import argparse
import cv2
import torch
from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def preprocess_image(image_path):
    """
    Предобработка изображения для улучшения читаемости текста
    """
    # Загрузка изображения через OpenCV
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Конвертация OpenCV BGR в PIL RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Увеличение резкости
    sharpened = pil_image.filter(ImageFilter.UnsharpMask(
        radius=2, 
        percent=150, 
        threshold=3
    ))
    
    # Увеличение контрастности
    contrast_enhancer = ImageEnhance.Contrast(sharpened)
    enhanced = contrast_enhancer.enhance(1.3)
    
    # Увеличение резкости
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    final_image = sharpness_enhancer.enhance(1.5)
    
    return final_image

def extract_text_with_moondream2(image, model, tokenizer, device):
    """
    Извлечение текста с помощью Moondream2
    """
    # Кодирование изображения
    image_embeds = model.encode_image(image)
    
    # Запрос для точного извлечения текста
    question = "Read all text in Russian visible in this image exactly as it appears. Copy the text verbatim without translation interpretation, explanation, or addition. Preserve exact spelling, numbers, and formatting."
    
    # Генерация ответа
    answer = model.answer_question(
        image_embeds=image_embeds,
        question=question,
        tokenizer=tokenizer
    )
    
    return answer.strip()

def main():
    parser = argparse.ArgumentParser(description='Распознавание текста с изображения с помощью Moondream2')
    parser.add_argument('image_path', help='Путь к файлу изображения')
    parser.add_argument('--no-preprocess', action='store_true', help='Не выполнять предобработку изображения')
    
    args = parser.parse_args()
    
    # Проверка существования файла
    if not os.path.exists(args.image_path):
        print(f"Ошибка: файл '{args.image_path}' не существует")
        return
    
    try:
        # Загрузка модели Moondream2
        print("Загрузка модели Moondream2...")
        model_id = "vikhyatk/moondream2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print("Модель загружена успешно!")
        
        # Предобработка изображения
        if args.no_preprocess:
            print("Предобработка отключена, загружаем оригинальное изображение...")
            image = Image.open(args.image_path)
        else:
            print("Выполняется предобработка изображения...")
            image = preprocess_image(args.image_path)
        
        # Распознавание текста
        print("Распознавание текста...")
        text = extract_text_with_moondream2(image, model, tokenizer, device)
        
        # Вывод результата
        print("\n" + "="*60)
        print("РАСПОЗНАННЫЙ ТЕКСТ:")
        print("="*60)
        print(text)
        print("="*60)
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()
