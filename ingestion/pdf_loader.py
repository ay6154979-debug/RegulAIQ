from typing import List, Dict
import fitz  # PyMuPDF

def load_pdf(file_path: str) -> List[Dict]:
    document = fitz.open(file_path)
    pages = []

    for page_num in range(len(document)):
        page = document[page_num]
        text = page.get_text()
        pages.append({
            "page": page_num + 1,
            "text": text
        })

    return pages
