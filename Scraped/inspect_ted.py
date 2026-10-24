import json
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def check_robots_txt():
    print("=" * 60)
    print("1. CHECKING https://www.ted.com/robots.txt")
    print("=" * 60)
    try:
        resp = requests.get("https://www.ted.com/robots.txt", headers=HEADERS, timeout=15)
        print(f"HTTP Status Code: {resp.status_code}")
        lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
        print(f"Total non-empty lines in robots.txt: {len(lines)}")
        print("Sample rules (first 15 lines):")
        for line in lines[:15]:
            print(f"  {line}")
    except requests.RequestException as e:
        print(f"[ERROR] Could not fetch robots.txt: {e}")


def inspect_talks_page():
    print("\n" + "=" * 60)
    print("2. INSPECTING https://www.ted.com/talks HTML & NEXT.JS STRUCTURE")
    print("=" * 60)
    url = "https://www.ted.com/talks"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        print(f"HTTP Status Code: {resp.status_code}")
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Show why old HTML scraping selectors fail (return 0 records)
        legacy_cards = soup.select("div.talk-link, div.media__message, a.ga-link")
        print(f"Legacy HTML selector match count ('div.talk-link'): {len(legacy_cards)} records")

        # 2. Check for Next.js __NEXT_DATA__ script tag
        next_data_script = soup.find("script", id="__NEXT_DATA__")
        if next_data_script and next_data_script.string:
            print("Found <script id='__NEXT_DATA__'>! Parsing embedded JSON...")
            data = json.loads(next_data_script.string)
            top_keys = list(data.keys())
            print(f"Top-level JSON keys: {top_keys}")
            page_props_keys = list(data.get("props", {}).get("pageProps", {}).keys())
            print(f"pageProps keys: {page_props_keys[:10]}")
        else:
            print("No __NEXT_DATA__ script tag found (page may use React Server Components or client-side GraphQL).")

        # 3. Count any <a href="/talks/..."> links visible in initial HTML
        talk_links = {
            a.get("href")
            for a in soup.find_all("a", href=True)
            if "/talks/" in a["href"] and not a["href"].endswith("/talks")
        }
        print(f"Unique '/talks/...' links visible in initial HTML response: {len(talk_links)}")
        for sample_link in list(talk_links)[:5]:
            print(f"  Sample link: {sample_link}")

    except requests.RequestException as e:
        print(f"[ERROR] Failed to inspect {url}: {e}")


if __name__ == "__main__":
    check_robots_txt()
    inspect_talks_page()