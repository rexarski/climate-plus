from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# Create a new instance of the Firefox driver
driver = webdriver.Firefox()

# Load the page with the table you want to scrape
url = "https://www.fsb-tcfd.org/example-disclosures/"
driver.get(url)

# Wait for the page to load and the table to render
# (replace "table_id" with the ID of the table you want to scrape)
table = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "bbg-cpt-feed__table"))
)

# Extract the HTML content of the table
table_html = table.get_attribute("innerHTML")

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(table_html, "html.parser")


# Get the table headers
headers = []
for th in soup.find_all("th"):
    headers.append(th.text.strip())

# Get the table rows
rows = []
for tr in soup.find_all("tr")[1:]:
    cells = []
    links = []
    for td in tr.find_all("td"):
        cells.append(td.text.strip())
    for a in tr.find_all("a"):
        links.append(a["href"])
    rows.append(cells + links)

headers = headers + ["TCFD Report URL", "Report URL"]

df = pd.DataFrame(rows, columns=headers)
df = df.drop(["Relevant TCFD Report", "TCFD Report URL"], axis=1)
df.to_csv("../data/tcfd.csv", index=False)

# Close the browser
driver.quit()
