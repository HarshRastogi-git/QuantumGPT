"""
Data preparation script for QuantumGPT v2.
Downloads Shakespeare + Project Gutenberg texts for a richer corpus.
"""
import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(DATA_DIR, "raw.txt")

SOURCES = [
    # Shakespeare complete works
    (
        "https://www.gutenberg.org/files/100/100-0.txt",
        "Shakespeare Complete Works"
    ),
    # Jane Austen - Pride and Prejudice
    (
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "Pride and Prejudice"
    ),
    # Charles Dickens - A Tale of Two Cities
    (
        "https://www.gutenberg.org/files/98/98-0.txt",
        "A Tale of Two Cities"
    ),
    # Arthur Conan Doyle - Adventures of Sherlock Holmes
    (
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        "Sherlock Holmes"
    ),
    # Mark Twain - Adventures of Huckleberry Finn
    (
        "https://www.gutenberg.org/files/76/76-0.txt",
        "Huckleberry Finn"
    ),
]


def download_text(url: str, title: str) -> str:
    print(f"  Downloading: {title}...")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        # Strip Project Gutenberg header/footer
        start_markers = ["*** START OF", "***START OF", "CHAPTER I", "CHAPTER 1", "ACT I", "ACT 1"]
        end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]
        start_idx = 0
        end_idx = len(raw)
        for marker in start_markers:
            idx = raw.find(marker)
            if idx != -1:
                start_idx = raw.find("\n", idx) + 1
                break
        for marker in end_markers:
            idx = raw.rfind(marker)
            if idx != -1:
                end_idx = idx
                break
        text = raw[start_idx:end_idx].strip()
        print(f"    ✓ {len(text):,} characters")
        return text
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return ""


def prepare_corpus():
    print("=" * 60)
    print("QuantumGPT v2 — Data Preparation")
    print("=" * 60)

    if os.path.exists(RAW_PATH):
        size = os.path.getsize(RAW_PATH)
        print(f"Found existing corpus: {size / 1024 / 1024:.2f} MB")
        ans = input("Re-download? [y/N]: ").strip().lower()
        if ans != "y":
            print("Using existing corpus.")
            return

    all_text = []
    for url, title in SOURCES:
        text = download_text(url, title)
        if text:
            all_text.append(f"\n\n{'='*60}\n{title.upper()}\n{'='*60}\n\n")
            all_text.append(text)

    if not all_text:
        print("No texts downloaded. Creating minimal fallback corpus...")
        all_text = ["The quick brown fox jumps over the lazy dog. " * 10000]

    corpus = "".join(all_text)
    # Clean: normalize whitespace, remove non-ASCII control chars
    import re
    corpus = re.sub(r"\r\n", "\n", corpus)
    corpus = re.sub(r"\r", "\n", corpus)
    corpus = re.sub(r"[^\x20-\x7E\n\t]", "", corpus)
    corpus = re.sub(r"\n{4,}", "\n\n\n", corpus)

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    mb = len(corpus.encode("utf-8")) / 1024 / 1024
    print(f"\n✓ Corpus saved: {RAW_PATH}")
    print(f"  Total size: {mb:.2f} MB")
    print(f"  Total chars: {len(corpus):,}")


if __name__ == "__main__":
    prepare_corpus()
