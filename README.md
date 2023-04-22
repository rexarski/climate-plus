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

- In `/utils`, `scraper.py` uses selenium to scrape the main table of Example Disclosures from [Task Force on Climate-related Financial Disclosures (TCFD)](https://www.fsb-tcfd.org/example-disclosures/) website.
  - The tabular data is stored in `/data/tcfd.csv`
- ~~Also in `/utils`, `pdf_parser.ipynb`:~~
  - ~~scrapes corresponding pdfs and store them in `/data/pdf` directory (unparsed)~~
  - ~~parses the pdfs to extract the text and store them in `/data/txt` directory~~
  - ~~Note: some urls are invalid and some pdfs are not readable (as the urls are linked to an actual website, or files in other formats.) Those text files are dropped.~~

## Retrieve key sentences using gpt-3.5

- Run `python chitchat_wip/tcfd_preparer.py` to generate sentence-based data for ClimateBERT, where each line is a key sentence from one of the TCFD's example disclosure reports, and the corresponding disclosure is one of the four TCFD's recommendations: *Risk Management*, *Strategy*, *Metrics and Targets*, *Governance*.
  - This question answering in context is adapted from [`chitchat`](https://github.com/rexarski/chitchat), which is still a work in progress.
  - Originally, we let the **"PDF to txt"** conversion to filter out those unreadable pdfs from invalid urls. However, the issue is that the conversion is not perfect, which retains the majority of the messy contents. Therefore, the number of tokens and its processing time are significantly increased.
  - And we decide to manually modify the pdf files and only keep the corresponding pages.

## Push the dataset to Hugging Face 🤗

- Upload the dataset to Hugging Face.
  - Run `huggingface-cli login` in terminal to log in to Hugging Face account.
  - Run `python chitchat_wip/tcfd_uploader.py` to upload the dataset object to Hugging Face.

> Caution ⚠️: The whole process of generating classification-defining sentences is still a long-shot which assumes that
>
> 1. the TCFD's example disclosures are representative of the whole population of the climate-related financial disclosures;
> 2. those sentences are distinguishable enough to be used as a classification boundary;
> 3. the page numbers provided are accurate and complete.

## Restructure `climate_fever`

- The original [`climate_fever` dataset](https://huggingface.co/datasets/climate_fever) needs some refinement for training.
  - For each `claim`, the `evidence` is a list of sentences. Our tweak here is to expand the list so that each claim-evidence pair only has two sentences (1 claim and 1 evidence).
  - The updated dataset is named after `climate_fever_fixed` ("fixed-length") and is available [here](https://huggingface.co/datasets/rexarski/climate_fever_fixed).

## Install the minimal python package

```bash
# cd to the root directory of the project
pip install -e ./minipackage

# unit tests of the package
pytest ./minipackage/tests
```

## Todos

- [ ] Prepare datasets compatible with ClimateBERT
  - [x] TCFD data
    - [x] Data scraping
    - [x] Generate sentence-based data using [`chitchat`](https://github.com/rexarski/chitchat) (QA)
      - [x] pdf parser
    - [x] Convert to a datasets object
    - [x] Update model card of [`TCFD_disclosure`](https://huggingface.co/datasets/rexarski/TCFD_disclosure)
  - [x] [`climate-fever`](https://huggingface.co/datasets/rexarski/climate_fever_fixed)
    - [ ] Update model card
- [x] Train ClimateBERT for downstream task 1 (factchecking)
- [ ] Train ClimateBERT for downstream task 2 (TCFD classification)
- [x] Generate a documentation website for `climate-plus`
