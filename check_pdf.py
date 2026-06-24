import fitz

pdf_path = "data/swiggy_annual_report.pdf"
doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    if text.strip():
        print(f"Page {page_num+1}: {len(text)} characters")
    else:
        print(f"Page {page_num+1}: **EMPTY**")