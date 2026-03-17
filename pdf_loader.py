import os
from pypdf import PdfReader


def load_all_pdfs(folder_path):

    text = ""

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            reader = PdfReader(os.path.join(folder_path, file))

            for page in reader.pages:
                text += page.extract_text()

    return text