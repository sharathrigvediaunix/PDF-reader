import pdfplumber


with pdfplumber.open("tests/fixtures/GP204.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")

    for i, page in enumerate(pdf.pages):
        print(f"\n--- PAGE {i + 1} ---")
        table = page.extract_table()
        if table:
            print(f"Total rows: {len(table)}")
            for row in table:
                print(row)