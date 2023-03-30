# climate-plus

<p align="center">
  <img src="logo.png" width="40%" alt="A placeholder logo of climate-plus">
</p>

## poetry

```bash
# To initiate the environment
poetry shell

# To add new dependencies
poetry add <package>

# To remove dependencies
poetry remove <package>

# To quit current environment
exit
```

## Web scraping and pdf parsing

- In `/utils`, `scrape.py` uses selenium to scrape the main table of Example Disclosures from [Task Force on Climate-related Financial Disclosures (TCFD)](https://www.fsb-tcfd.org/example-disclosures/) website.
  - The tabular data is stored in `/data/tcfd.csv`
- Also in `/utils`, `parse_pdf.ipynb`:
  - scrapes corresponding pdfs and store them in `/data/pdf` directory (unparsed)
  - parses the pdfs to extract the text and store them in `/data/txt` directory
  - Note: some urls are invalid and some pdfs are not readable (as the urls are linked to an actual website, or files in other formats.) Those text files are dropped.

## Todos

- [ ] Prepare datasets compatible with ClimateBERT
  - [ ] climate-fever
  - [ ] TCFD data
    - [x] Data scraping
    - [ ] Generate sentence-based data using [`chitchat`](https://github.com/rexarski/chitchat) (QA)
- [ ] Train ClimateBERT for downstream task 1 (factchecking)
- [ ] Train ClimateBERT for downstream task 2 (TCFD classification)
- [ ] Generate a documentation website for `climate-plus`
