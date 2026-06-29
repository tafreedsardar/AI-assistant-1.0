from PyPDF2 import PdfReader

reader = PdfReader("Fall 2025 Course Schedule PDF (3).pdf")
number_of_pages = len(reader.pages)
page = reader.pages[7]
text = page.extract_text()
print(text)