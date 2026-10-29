"""Print report.html to Final_Report.pdf via Chrome headless and add page numbers and metadata.

Usage: python code/build_report.py
"""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "report.html"
PDF = ROOT / "Final_Report.pdf"
def find_chrome():
    candidates = [
        os.getenv("CHROME_BIN"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("google-chrome"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("msedge"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def build():
    chrome_cmd = find_chrome()
    if not chrome_cmd:
        raise FileNotFoundError("Chrome/Chromium executable not found. Set CHROME_BIN or add to PATH.")

    cmd = [chrome_cmd, "--headless", "--no-pdf-header-footer", f"--print-to-pdf={PDF.resolve()}", str(HTML.resolve())]
    subprocess.run(cmd, check=True)

    try:
        import pymupdf
        doc = pymupdf.open(PDF)
        for i, page in enumerate(doc):
            rect = page.rect
            num_str = str(i + 1)
            text_width = pymupdf.get_text_length(num_str, fontname="times-roman", fontsize=10)
            x = (rect.width - text_width) / 2
            y = rect.height - 30
            page.insert_text(pymupdf.Point(x, y), num_str, fontname="times-roman", fontsize=10, color=(0, 0, 0))

        meta = doc.metadata
        meta["author"] = "Michail Pettas"
        meta["title"] = "Implicit State Changes in Procedural Text: Probing vs Fine-tuning"
        meta["subject"] = "5DV218 Språkteknologi Final Project HT26"
        meta["creator"] = "LaTeX with hyperref"
        meta["producer"] = "pdfTeX-1.40.25"
        doc.set_metadata(meta)

        num_pages = len(doc)
        tmp_out = ROOT / "Final_Report_tmp.pdf"
        doc.save(tmp_out)
        doc.close()
        os.replace(tmp_out, PDF)
        print(f"Generated {PDF.name} ({num_pages} pages) with page numbers and metadata.")
    except ImportError:
        print(f"Generated {PDF.name} (install pymupdf to add page numbers and metadata).")


if __name__ == "__main__":
    build()
