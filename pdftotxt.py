import fitz  # PyMuPDF
import re

def extract_presentation_slides(pdf_path):
    doc = fitz.open(pdf_path)
    structured_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
#        structured_text.append(f"\n{'='*50}")
#        structured_text.append(f"СЛАЙД {page_num + 1}")
#        structured_text.append(f"{'='*50}")
        
        # Метод 1: Извлечение с сортировкой по позиции (вертикальной)
        text_blocks = page.get_text("blocks")
        text_blocks.sort(key=lambda block: block[1])  # Сортировка по Y координате
        
        for block in text_blocks:
            text = block[4].strip()
            if text and len(text) > 2:  # Фильтруем мусор
                structured_text.append(text)
        
        structured_text.append("")  # Разделитель
    
    return "\n".join(structured_text)

def post_process_text(text):
    """Постобработка извлеченного текста"""

    # Удаление номеров страниц в середине текста
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Восстановление переносов слов
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Удаление лишних пустых строк
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Очистка специальных символов
    text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?;:()\/\-–—><\n]', '', text)

    return text.strip()

# Использование
#text = extract_presentation_slides("communications_merged.pdf")
#text = extract_presentation_slides("merged.pdf")
#text = extract_presentation_slides("leadership/workbook.pdf")
#text = extract_presentation_slides("energy/merged.pdf")
#text = extract_presentation_slides("analytics/merged.pdf")
text = extract_presentation_slides("problems/1.pdf")
with open("problems/structured_presentation1.txt", "w", encoding="utf-8") as f:
    f.write(post_process_text(text))
