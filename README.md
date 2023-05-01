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

## Model training

|        Downstream task         |               File                |      Base model      |                    Data set                    | Accuracy | F1 score |
| :----------------------------: | :-------------------------------: | :------------------: | :--------------------------------------------: | :------: | :------: |
|          Factchecking          | `trainer/factchecking_trainer.py` | `bert-base-uncased`  |             `climate_fever_fixed`              |  0.7158  |  0.6015  |
| TCFD disclosure classification |   `trainer/tcfd_trainer.ipynb`    | `distilroberta-base` |       `TCFD_disclosure` (11 subclasses)        |  0.3667  |  0.3333  |
| TCFD disclosure classification |   `trainer/tcfd_trainer.ipynb`    | `distilroberta-base` | `TCFD_disclosure` (11 subclasses -> 4 classes) |  0.8333  |  0.7771  |
| TCFD disclosure classification |   `trainer/tcfd_11_trainer.py`    | `distilroberta-base` |    `training_data.json` (4 classes + None)     |  0.8075  |  0.6268  |

## Demo

- Factchecking
  - Example 1
    - **Claim**: Sea ice has diminished much faster than scientists and climate models anticipated.
    - **Evidence**: Past models have underestimated the rate of Arctic shrinkage and underestimated the rate of precipitation increase.
    - **Label**: `SUPPORTS`
    - **Prediction**: `SUPPORTS` ✅
  - Example 2
    - **Claim**: Climate Models Have Overestimated Global Warming
    - **Evidence**: The 2017 United States-published National Climate Assessment notes that "climate models may still be underestimating or missing relevant feedback processes".
    - **Label**: `SUPPORTS`
    - **Prediction**: `REFUTES` ❌
  - Example 3
    - **Claim**: Climate skeptics argue temperature records have been adjusted in recent years to make the past appear cooler and the present warmer, although the Carbon Brief showed that NOAA has actually made the past warmer, evening out the difference.
    - **Evidence**: Reconstructions have consistently shown that the rise in the instrumental temperature record of the past 150 years is not matched in earlier centuries, and the name "hockey stick graph" was coined for figures showing a long-term decline followed by an abrupt rise in temperatures.
    - **Label**: `NOT_ENOUGH_INFO`
    - **Prediction**: `NOT_ENOUGH_INFO` ✅
  - Example 4
    - **Claim**: Humans are too insignificant to affect global climate.
    - **Evidence**: Human impact on the environment or anthropogenic impact on the environment includes changes to biophysical environments and ecosystems, biodiversity, and natural resources caused directly or indirectly by humans, including global warming, environmental degradation (such as ocean acidification), mass extinction and biodiversity loss, ecological crisis, and ecological collapse.
    - **Label**: `REFUTES`
    - **Prediction**: `REFUTES` ✅
- TCFD disclosure classification
  - Example 1
    - **Text**: As a global provider of transport and logistics services, we are often called on for expert input and industry insights by government representatives.
    - **Label**: `Risk Management a)`
    - **Prediction**: `Metrics and Targets` ❌
  - Example 2
    - **Text**: There are no sentences in the provided excerpts that disclose Scope 1 and Scope 2, and, if appropriate Scope 3 GHG emissions. The provided excerpts focus on other metrics and targets related to social impact investing, assets under management, and carbon footprint calculations.
    - **Label**: `Metrics and Targets b)`
    - **Prediction**: `Metrics and Targets a)`  🔧
  - Example 3
    - **Text**: Our strategy needs to be resilient under a range of climate-related scenarios. This year we have undertaken climate-related scenario testing of a select group of customers in the thermal coal supply chain. We assessed these customers using two of the International Energy Agency’s scenarios; the ‘New Policies Scenario’ and the ‘450 Scenario’. Our reporting reflects the Financial Stability Board’s (FSB) Task Force on Climate-Related Disclosures (TCFD) recommendations. Using the FSB TCFD’s disclosure framework, we have begun discussions with some of our customers in emissions-intensive industries. The ESG Committee is responsible for reviewing and approving our climate change-related objectives, including goals and targets. The Board Risk Committee has formal responsibility for the overview of ANZ’s management of new and emerging risks, including climate change-related risks.
    - **Label**: `Strategy c)`
    - **Prediction**: `Risk Management a)` ❌
  - Example 4
    - **Text**: AXA created a Group-level Responsible Investment Committee (RIC), chaired by the Group Chief Investment Officer, and including representatives from AXA Asset Management entities, representatives of Corporate Responsibility (CR), Risk Management and Group Communication.
    - **Label**: `Goverance b)`
    - **Prediction**: `Goverance b)` ✅

## References

- [ClimateBert](https://climatebert.ai/), AI powered climate-related corporate disclosure analytics
- [ClimateBert: A Pretrained Language Model for Climate-Related Text](https://arxiv.org/abs/2110.12010)
- [`chitchat`](https://github.com/rexarski/chitchat)
- Datasets
  - [`climate_fever_fixed`](https://huggingface.co/datasets/rexarski/climate_fever_fixed)
  - [`TCFD_disclosure`](https://huggingface.co/datasets/rexarski/TCFD_disclosure)
- Model
  - [`bert-base-climate-fever-fixed`](https://huggingface.co/rexarski/bert-base-climate-fever-fixed)
  - [`distilroberta-tcfd-disclosure`](https://huggingface.co/rexarski/distilroberta-tcfd-disclosure)
  - [`distilroberta-tcfd-disclosure-5`](https://huggingface.co/rexarski/distilroberta-tcfd-disclosure-5)
