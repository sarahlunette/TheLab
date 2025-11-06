import json
import os

# 🧩 adapter ce chemin selon ton OS
BOOKMARKS_PATH = os.path.expanduser(
    r"C:\Users\sarah\AppData\Local\Google\Chrome\User Data\Default\Bookmarks"  # Windows
)

OUTPUT_JSON = "chrome_favorites.json"
FOLDER_NAME = "Projets"  # ← nom du dossier de favoris que tu veux extraire

def find_folder(node, folder_name):
    """Recherche récursive du dossier."""
    if node.get("type") == "folder" and node.get("name") == folder_name:
        return node
    if "children" in node:
        for child in node["children"]:
            found = find_folder(child, folder_name)
            if found:
                return found
    return None

def extract_urls(node):
    """Extract all URLs from a folder."""
    urls = []
    if node.get("type") == "url":
        urls.append(node["url"])
    elif node.get("type") == "folder":
        for child in node.get("children", []):
            urls.extend(extract_urls(child))
    return urls

def main():
    with open(BOOKMARKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    roots = data.get("roots", {})
    all_urls = []

    # recherche du dossier dans toutes les racines
    for root in roots.values():
        folder = find_folder(root, FOLDER_NAME)
        if folder:
            all_urls = extract_urls(folder)
            break

    if not all_urls:
        print(f"⚠️ Dossier '{FOLDER_NAME}' introuvable dans les favoris.")
        return

    # sauvegarde en JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_urls, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(all_urls)} liens extraits et enregistrés dans {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
