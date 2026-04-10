#!/usr/bin/env python3
from pathlib import Path

import markdown
from weasyprint import HTML


def main():
    base = Path(__file__).resolve().parent
    markdown_path = base / "REPORT.md"
    image_path = base / "learning_curves.png"
    pdf_path = base / "REPORT.pdf"

    markdown_text = markdown_path.read_text(encoding="utf-8")
    body = markdown.markdown(markdown_text, extensions=["extra"])

    html = f"""
    <html>
      <head>
        <meta charset=\"utf-8\">
        <style>
          body {{
            font-family: Georgia, serif;
            margin: 36px;
            line-height: 1.45;
            color: #111;
          }}
          h1, h2 {{
            color: #111;
          }}
          code {{
            background: #f3f3f3;
            padding: 2px 4px;
            border-radius: 4px;
          }}
          ul {{
            margin-top: 0.3rem;
          }}
          img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
          }}
          .figure {{
            margin: 24px 0;
          }}
        </style>
      </head>
      <body>
        {body}
        <div class=\"figure\">
          <h2>Learning Curves</h2>
          <img src=\"{image_path.as_uri()}\" alt=\"Learning curves\">
        </div>
      </body>
    </html>
    """

    HTML(string=html, base_url=base.as_uri()).write_pdf(str(pdf_path))
    print(pdf_path)


if __name__ == "__main__":
    main()