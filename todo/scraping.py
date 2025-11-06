import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import os
import sys
sys.path.append('../../data/')
sys.path.append('../../data/scraping/')
from src.data.todo.scraping import pages_to_scrape

def fetch_html(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def parse_normal_page(url, html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else ''

    # sections — basé sur h2 (ou h3 selon page)
    sections = []
    for hdr in soup.find_all(["h2", "h3"]):
        htext = hdr.get_text(strip=True)
        content = []
        for sib in hdr.next_siblings:
            if sib.name in ["h2", "h3"]:
                break
            if sib.name == "p":
                content.append(sib.get_text(strip=True))
        sections.append({"heading": htext, "paragraphs": content})

    # liens
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(url, href)
        text = a.get_text(strip=True)
        links.append({"text": text, "href": full})

    return {
        "url": url,
        "title": title_text,
        "sections": sections,
        "links": links
    }

def parse_glossary_page(url, html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else ''

    # On suppose que les mots sont des liens <a> dans une liste
    terms = []
    for a in soup.find_all("a", href=True):
        term = a.get_text(strip=True)
        href = urljoin(url, a["href"])
        if term:
            # On pourrait fetch ce href pour obtenir sa définition, mais ici on met le lien
            terms.append({"term": term, "href": href})

    return {
        "url": url,
        "title": title_text,
        "terms": terms
    }

def parse_term_page(url, html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else ''

    # définition (généralement dans <p> sous le titre)
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)
    # on peut extraire annotation si existe (souvent dans <em> ou <small>)
    # et les liens associés
    related = []
    for a in soup.find_all("a", href=True):
        related.append({"text": a.get_text(strip=True), "href": urljoin(url, a["href"])})

    return {
        "url": url,
        "title": title_text,
        "definition_paragraphs": paragraphs,
        "related": related
    }

def build_json():

    result = {"pages": []}

    for url in pages_to_scrape:
        html = fetch_html(url)
        # décider quel parser utiliser
        if "drr-glossary/terminology" in url:
            gloss = parse_glossary_page(url, html)
            result["pages"].append(gloss)
        elif "terminology/build-back-better" in url:
            term = parse_term_page(url, html)
            result["pages"].append(term)
        else:
            norm = parse_normal_page(url, html)
            result["pages"].append(norm)

    # Pour chaque terme du glossaire, on peut aussi “suivre le lien” et récupérer la définition
    # Exemple : pour page 3 (index), la liste des termes
    for pg in result["pages"]:
        if "terms" in pg:
            new_terms = []
            for t in pg["terms"]:
                try:
                    h = fetch_html(t["href"])
                    parsed = parse_term_page(t["href"], h)
                    new_terms.append({
                        "term": t["term"],
                        "href": t["href"],
                        "definition_paragraphs": parsed["definition_paragraphs"],
                        "related": parsed["related"]
                    })
                except Exception as e:
                    # si échec, on se contente du lien
                    new_terms.append(t)
            pg["terms"] = new_terms

    return result

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def save_json_to_pdf(data, output_path):
    """
    Sauvegarde le JSON dans un fichier PDF lisible.
    """
    # création du PDF
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    textobject = c.beginText(2 * cm, height - 2 * cm)
    textobject.setFont("Helvetica", 10)

    # convertir le JSON en texte lisible
    json_text = json.dumps(data, ensure_ascii=False, indent=2)

    # découper en lignes pour ne pas dépasser la largeur
    for line in json_text.split("\n"):
        textobject.textLine(line)
        if textobject.getY() < 2 * cm:  # si on atteint le bas de la page
            c.drawText(textobject)
            c.showPage()
            textobject = c.beginText(2 * cm, height - 2 * cm)
            textobject.setFont("Helvetica", 10)

    c.drawText(textobject)
    c.save()

if __name__ == "__main__":
    j = build_json()
    output_dir = os.path.abspath("../../app/docs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scraped_pages.pdf")

    save_json_to_pdf(j, output_file)

    print(f"✅ PDF sauvegardé avec succès dans : {output_file}")
