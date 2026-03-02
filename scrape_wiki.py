import urllib.request
import re
import json

url = "https://ko.wikipedia.org/wiki/위키백과:자주_틀리는_한국어/외래어"

# Wikipedia URLs must be quoted for non-ASCII
req = urllib.request.Request(
    urllib.parse.quote(url, safe=":/"), 
    headers={'User-Agent': 'Mozilla/5.0'}
)
html = urllib.request.urlopen(req).read().decode('utf-8')

# Extract all rows from wikitable
foreign_dict = {}

tables = re.findall(r'<table class="wikitable.*?>(.*?)</table>', html, re.DOTALL)
count = 0
for table in tables:
    rows = re.findall(r'<tr.*?>(.*?)</tr>', table, re.DOTALL)
    for row in rows[1:]:  # Skip header
        cols = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.DOTALL)
        if len(cols) >= 2:
            count += 1
            # Remove HTML tags inside columns
            wrong = re.sub(r'<[^>]+>', '', cols[0])
            correct = re.sub(r'<[^>]+>', '', cols[1])
            
            # Clean up footnote references [1] or similar
            wrong = re.sub(r'\[.*?\]', '', wrong).strip()
            correct = re.sub(r'\[.*?\]', '', correct).strip()
            # Clean up HTML entities
            wrong = wrong.replace('&#160;', ' ').replace('&amp;', '&')
            correct = correct.replace('&#160;', ' ').replace('&amp;', '&')
            
            # Split by commas or slashes if multiple
            wrongs = [w.strip() for w in re.split(r'[,/]', wrong) if w.strip()]
            corrects = [c.strip() for c in re.split(r'[,/]', correct) if c.strip()]
            
            for c in corrects:
                if not c: continue
                # Remove parenthetical remarks e.g., (표준어)
                c_clean = re.sub(r'\(.*?\)', '', c).strip()
                if not c_clean: continue
                
                if c_clean not in foreign_dict:
                    foreign_dict[c_clean] = []
                for w in wrongs:
                    w_clean = re.sub(r'\(.*?\)', '', w).strip()
                    if w_clean and w_clean not in foreign_dict[c_clean] and w_clean != c_clean:
                        foreign_dict[c_clean].append(w_clean)

try:
    with open('error_generation/resources/foreign_words.json', 'r', encoding='utf-8') as f:
        existing = json.load(f)

    for c, ws in existing.items():
        if c not in foreign_dict:
            foreign_dict[c] = []
        for w in ws:
            if w not in foreign_dict[c]:
                foreign_dict[c].append(w)
except Exception:
    pass

# filter empty lists
foreign_dict = {k: v for k, v in foreign_dict.items() if v}

print(f"Total dictionary size: {len(foreign_dict)} correct words mapped to typos.")

with open('error_generation/resources/foreign_words.json', 'w', encoding='utf-8') as f:
    json.dump(foreign_dict, f, ensure_ascii=False, indent=4)
