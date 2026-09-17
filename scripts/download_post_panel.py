"""
Download 200 "post and panel signs" images into data/post_panel/.

Dependencies:
    pip install icrawler duckduckgo-search requests
"""

import os
import sys
import time
import requests

DDG_KEYWORDS = [
    "post and panel signs",
    "post panel sign outdoor",
    "post panel sign business",
    "double post panel sign",
    "post panel wayfinding sign",
    "post mounted panel sign",
    "aluminum post panel sign",
]


def count_images(directory):
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))
    ])


def next_filename(directory):
    existing = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))
    ])
    if not existing:
        return 1
    try:
        return int(os.path.splitext(existing[-1])[0]) + 1
    except ValueError:
        return len(existing) + 1


def download_via_ddg(output_dir, keywords, target):
    from duckduckgo_search import DDGS

    seen_urls = set()
    all_results = []

    for kw in keywords:
        if count_images(output_dir) + len(all_results) >= target:
            break
        need = target - count_images(output_dir) - len(all_results)
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
            print(f"  DDG error for '{kw}': {e} — waiting 15s before next...")
            time.sleep(15)
            continue
        time.sleep(3)

    counter = next_filename(output_dir)
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in all_results:
        if count_images(output_dir) >= target:
            break
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:
                ct = resp.headers.get("Content-Type", "")
                ext = "png" if "png" in ct else "gif" if "gif" in ct else "webp" if "webp" in ct else "jpg"
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

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "post_panel")
    os.makedirs(output_dir, exist_ok=True)

    target = 200

    # --- Primary: Bing ---
    print("Searching 'post and panel signs' via Bing ...")
    BingImageCrawler(
        feeder_threads=4,
        parser_threads=4,
        downloader_threads=8,
        storage={"root_dir": output_dir},
    ).crawl(keyword="post and panel signs", max_num=target, min_size=(100, 100))

    downloaded = count_images(output_dir)
    print(f"After Bing: {downloaded} images.")

    # --- Top up: more Bing keyword variations ---
    bing_extra_keywords = [
        "post panel sign outdoor",
        "post and panel sign business",
        "double post panel sign",
        "post panel wayfinding sign",
        "aluminum post panel sign",
        "post and panel real estate sign",
    ]
    for kw in bing_extra_keywords:
        if downloaded >= target:
            break
        remaining = target - downloaded
        print(f"Bing top-up: '{kw}' ({remaining} more needed)...")
        BingImageCrawler(
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": output_dir},
        ).crawl(keyword=kw, max_num=remaining, min_size=(100, 100))
        downloaded = count_images(output_dir)
        print(f"  → {downloaded} total images now.")

    # --- Final top up: DuckDuckGo (if still needed) ---
    if downloaded < target:
        print(f"Topping up via DuckDuckGo ({target - downloaded} more needed)...")
        download_via_ddg(output_dir, DDG_KEYWORDS, target)
        downloaded = count_images(output_dir)

    print(f"Done. {downloaded} images saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
