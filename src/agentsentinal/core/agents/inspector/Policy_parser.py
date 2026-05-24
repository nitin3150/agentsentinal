import pdfplumber
 
def parse_policy_pdf(pdf_path: str) -> str:
    """
    Takes a PDF file path and returns its text content as a single string.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"--- Page {i} ---\n{text}")
 
    return "\n\n".join(pages)
 
 
if __name__ == "__main__":
    print(parse_policy_pdf(r"src\agentsentinal\core\agents\inspector\SAMPLE-EMPLOYEE-POLICY-HANDBOOK.pdf"))
 