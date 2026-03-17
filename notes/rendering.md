# Rendering the manuscript

## Canonical render command
```bash
quarto render
```

Bare `quarto render` renders all formats defined in `_quarto.yml` (html, apaquarto-docx, apaquarto-pdf) and produces format-links in the HTML output.

Note: `quarto render --to html,apaquarto-docx,apaquarto-pdf` does NOT work in this project (extension parsing error). If debugging, render each format individually to isolate failures.

## Output
Rendered files go to `docs/`:
- `docs/index.html` — HTML with format-links
- `docs/index.docx` — APA-formatted Word document
- `docs/index.pdf` — APA-formatted PDF
