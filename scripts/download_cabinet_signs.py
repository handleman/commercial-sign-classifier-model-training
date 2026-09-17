"""
Download 200 "cabinet signs" images into data/cabinet/.

Dependencies:
    pip install icrawler duckduckgo-search requests
"""

import os
import sys
import time
import requests

DDG_KEYWORDS = [
    "cabinet signs",
    "illuminated cabinet sign",
    "LED cabinet sign",
    "outdoor cabinet sign",
    "backlit cabinet sign",
    "custom cabinet sign",
    "sign cabinet storefront",
]


def count_images(directory):
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))
    ])


def next_filename(directory):
    """Return next sequential filename like 000148.jpg"""
    existing = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))
    ])
    if not existing:
        return 1
    last = existing[-1]
    try:
        return int(os.path.splitext(last)[0]) + 1
    except ValueError:
        return len(existing) + 1


def download_via_ddg(output_dir, keywords, target):
    """Use DuckDuckGo image search to download images up to target count."""
    from duckduckgo_search import DDGS

    downloaded = count_images(output_dir)
    seen_urls = set()

    # Collect all URLs first
    all_results = []
    for kw in keywords:
        if downloaded + len(all_results) >= target:
            break
        need = target - downloaded - len(all_results)
        print(f"  DDG search: '{kw}' (need {need} more)...")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(kw, max_results=need + 20))
            for r in results:
                url = r.get("image", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(url)
        except Exception as e:
            print(f"  DDG error for '{kw}': {e}")
        time.sleep(1)

    # Download them
    counter = next_filename(output_dir)
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in all_results:
        if count_images(output_dir) >= target:
            break
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:
                ext = "jpg"
                ct = resp.headers.get("Content-Type", "")
                if "png" in ct:
                    ext = "png"
                elif "gif" in ct:
                    ext = "gif"
                elif "webp" in ct:
                    ext = "webp"
                fname = os.path.join(output_dir, f"{counter:06d}.{ext}")
                with open(fname, "wb") as f:
                    f.write(resp.content)
                print(f"  image #{count_images(output_dir)}  {url[:80]}")
                counter += 1
        except Exception:
            pass


def main():
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        os.system(f"{sys.executable} -m pip install icrawler --break-system-packages")
        from icrawler.builtin import BingImageCrawler

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cabinet")
    os.makedirs(output_dir, exist_ok=True)

    target = 200

    # --- Primary: Bing ---
    print("Searching 'cabinet signs' via Bing ...")
    BingImageCrawler(
        feeder_threads=4,
        parser_threads=4,
        downloader_threads=8,
        storage={"root_dir": output_dir},
    ).crawl(keyword="cabinet signs", max_num=target, min_size=(100, 100))

    downloaded = count_images(output_dir)
    print(f"After Bing: {downloaded} images.")

    # --- Top up: DuckDuckGo ---
    if downloaded < target:
        print(f"Topping up via DuckDuckGo ({target - downloaded} more needed)...")
        download_via_ddg(output_dir, DDG_KEYWORDS, target)
        downloaded = count_images(output_dir)

    print(f"Done. {downloaded} images saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
