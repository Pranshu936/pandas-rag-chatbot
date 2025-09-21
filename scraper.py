import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

# Idhar hum pandas User Guide ka base URL set kar rahe hain
BASE_URL = "https://pandas.pydata.org/docs/user_guide/index.html"

# idhar improved version jo thoda flexible hai navigation dhoondhne mein
def get_all_user_guide_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = set()

        # Pehla try: original selector
        nav_section = soup.find('div', class_='bd-toc-item-container')

        # Agar nahi milti, alternative try karo
        if not nav_section:
            # Example: koi div jiski class mein "toc" ho (case-insensitive)
            nav_section = soup.find('div', class_=lambda x: x and "toc" in x.lower())

        if not nav_section:
            print("⚠️ Navigation section nahi mila using known selectors.")
            return []

        for a_tag in nav_section.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(url, href)
            links.add(full_url)
        print(f"🔍 Found {len(links)} unique links in the User Guide.")
        return list(links)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the main page: {e}")
        return []


def scrape_page_content(url):
    """
    Yeh function ek page ka main text scrape karega
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Idhar main content dhundh rahe hain <main> tag ke andar
        main_content = soup.find('main', {'id': 'main-content'})
        
        if main_content:
            # Text ko saaf karke return kar rahe hain
            text = main_content.get_text(separator=' ', strip=True)
            return {
                "url": url,
                "text": text
            }
        else:
            print(f"⚠️ Main content nahi mila: {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page content from {url}: {e}")
        return None

def main():
    """
    Yahan se poora scraper chal raha hai
    """
    print("🚀 Starting the pandas documentation scraper...")
    
    # Step 1: Idhar hum sare links nikal rahe hain User Guide ke
    all_links = get_all_user_guide_links(BASE_URL)
    
    if not all_links:
        print("Koi link nahi mila bhai. Exiting...")
        return
        
    scraped_data = []
    
    # Step 2: Ab har link pe ghoom ghoom ke data scrape karenge
    for i, link in enumerate(all_links):
        print(f"Scraping ({i+1}/{len(all_links)}): {link}")
        content = scrape_page_content(link)
        if content:
            scraped_data.append(content)
        # Idhar thoda ruk rahe hain taaki server ko load na lage
        time.sleep(0.5) 

    # Step 3: Ab sab data JSON file mein save kar rahe hain
    output_filename = 'scraped_pandas_docs.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Success! {len(scraped_data)} pages scrape ho gaye.")
    print(f"Data save ho gaya idhar -> '{output_filename}'")


if __name__ == "__main__":
    main()
