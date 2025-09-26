# exercise1_webscraping.py
# COMP262 - Assignment 1 - Exercise 1
# Student: Parichit Upadhayay

import requests
from bs4 import BeautifulSoup
import csv

# Step 1: Fetch webpage
url = "https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence-software-engineering-technology-online-optional-co-op"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Extract required information

# Website title
title = soup.title.string.strip() if soup.title else "Not Found"

highlights = []
highlights_section = soup.find("ul")
if highlights_section:
    highlights = [li.get_text(strip=True) for li in highlights_section.find_all("li")]
else:
    highlights.append("Highlights not found")

overview_text = []
all_paragraphs = soup.find_all("p")
if all_paragraphs and len(all_paragraphs) >= 2:
    overview_text = [p.get_text(strip=True) for p in all_paragraphs[:2]]
else:
    overview_text.append("Overview paragraphs not found")

# Step 3: Export results to CSV
filename = "Parichit_my_future.csv"
with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Website Title", title])
    writer.writerow([])
    writer.writerow(["Program Highlights"])
    for h in highlights:
        writer.writerow(["", h])
    writer.writerow([])
    writer.writerow(["Program Overview"])
    for p in overview_text:
        writer.writerow(["", p])

print(f"Data scraped and saved to {filename}")
