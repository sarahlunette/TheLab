import json
import streamlit as st
import markdown
from weasyprint import HTML

st.set_page_config(page_title="ChatGPT JSON Viewer", layout="wide")
st.title("📄 JSON Viewer — ChatGPT Style + Styled PDF Export")

# --- Cleaning function (same as before) ---
def clean_llm_markdown(text: str) -> str:
    import re
    text = text.replace("\\n", "\n").replace("\\t", "    ").strip('"')
    text = re.sub(r"\n\s+(\- |\* |\d+\. |#)", r"\n\1", text)
    text = re.sub(r"\n\s*(#{1,6})\s*", r"\n\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- PDF Generator using Markdown → HTML → PDF ---
def generate_pdf_from_markdown(markdown_text, output_path):
    # Convert Markdown to HTML
    html = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])

    # Embed HTML inside a styled template
    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica', sans-serif;
                margin: 2rem;
                line-height: 1.6;
            }}
            h1, h2, h3, h4 {{
                color: #333;
                margin-top: 1.5rem;
            }}
            ul {{
                margin-left: 1.2rem;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 4px 6px;
                border-radius: 4px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 8px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    # Generate PDF
    HTML(string=html_template).write_pdf(output_path)


# --- Streamlit File Upload ---
uploaded_file = st.file_uploader("Upload JSON", type=["json"])

if uploaded_file:
    data = json.load(uploaded_file)
    raw_text = data.get("answer", json.dumps(data, ensure_ascii=False))

    clean_text = clean_llm_markdown(raw_text)

    st.markdown("### 👁️ Cleaned Output (Streamlit-style Markdown)")
    st.markdown(clean_text)

    # --- PDF Export Button ---
    if st.button("📥 Export Styled PDF"):
        pdf_path = "report.pdf"
        generate_pdf_from_markdown(clean_text, pdf_path)

        with open(pdf_path, "rb") as pdf:
            st.download_button(
                "Download PDF",
                pdf,
                file_name="report.pdf",
                mime="application/pdf",
            )
