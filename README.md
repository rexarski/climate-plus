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

|        Downstream task         |                                 File                                 |      Base model      |                                                    Data set                                                    |                                              End model                                               | Accuracy | Weighted F1 score |
| :----------------------------: | :------------------------------------------------------------------: | :------------------: | :------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: | :------: | :---------------: |
|          Factchecking          | [`trainer/factchecking_trainer.py`](trainer/factchecking_trainer.py) | `bert-base-uncased`  |             [`climate_fever_fixed`](https://huggingface.co/datasets/rexarski/climate_fever_fixed)              |   [`bert-base-climate-fever-fixed`](https://huggingface.co/rexarski/bert-base-climate-fever-fixed)   |  0.7087  |      0.7144       |
| TCFD disclosure classification |      [`trainer/tcfd_11_trainer.py`](trainer/tcfd_11_trainer.py)      | `distilroberta-base` |         [`TCFD_disclosure`](https://huggingface.co/datasets/rexarski/TCFD_disclosure) (11 subclasses)          |   [`distilroberta-tcfd-disclosure`](https://huggingface.co/rexarski/distilroberta-tcfd-disclosure)   |  0.3333  |      0.3144       |
| TCFD disclosure classification |      [`trainer/tcfd_11_trainer.py`](trainer/tcfd_11_trainer.py)      | `distilroberta-base` | [`TCFD_disclosure`](https://huggingface.co/datasets/rexarski/TCFD_disclosure)[^1] (11 subclasses -> 4 classes) |   [`distilroberta-tcfd-disclosure`](https://huggingface.co/rexarski/distilroberta-tcfd-disclosure)   |  0.8333  |      0.8246       |
| TCFD disclosure classification |       [`trainer/tcfd_5_trainer.py`](trainer/tcfd_5_trainer.py)       | `distilroberta-base` |                    [`training_data.json`](data/training_data.json)[^2] (4 classes + `None`)                    | [`distilroberta-tcfd-disclosure-5`](https://huggingface.co/rexarski/distilroberta-tcfd-disclosure-5) |  0.8075  |      0.8013       |

[^1]: Essentially, this is the same model as the previous one, but the evaluation metrics are calculated based on a "loose" version of "correct prediction". Basically, if the prediction of a subcategory falls into the same category as the true label, then it is considered as a correct one.

[^2]: `training_data.json` contains 50k text sequences annotated with 5 classes (4 TCFD categories + "None"). It was used by the ClimateBERT team in their draft notebook [`training-example`](https://github.com/ClimateBert/training-example).

## Demo

- Factchecking
  - Example 1
    - **Claim**: there is no relationship between temperature and carbon dioxide emissions by Â­humans[...]
    - **Evidence**: Human activities are now causing atmospheric concentrations of greenhouse gases—including carbon dioxide, methane, tropospheric ozone, and nitrous oxide—to rise well above pre-industrial levels ... Increases in greenhouse gases are causing temperatures to rise ...
    - **Label**: `REFUTES`
    - **Prediction**: `REFUTES` ✅
  - Example 2
    - **Claim**: The late 1970s marked the end of a 30-year cooling trend.
    - **Evidence**: During the last 20-30 years, world temperature has fallen, irregularly at first but more sharply over the last decade..
    - **Label**: `NOT_ENOUGH_INFO`
    - **Prediction**: `NOT_ENOUGH_INFO` ✅
  - Example 3
    - **Claim**: Even during a period of long term warming, there are short periods of cooling due to climate variability.
    - **Evidence**: El Niño events cause short-term (approximately 1 year in length) spikes in global average surface temperature while La Niña events cause short term cooling.
    - **Label**: `SUPPORTS`
    - **Prediction**: `SUPPORTS` ✅
  - Example 4
    - **Claim**: Humans are too insignificant to affect global climate.
    - **Evidence**: Human impact on the environment or anthropogenic impact on the environment includes changes to biophysical environments and ecosystems, biodiversity, and natural resources caused directly or indirectly by humans, including global warming, environmental degradation (such as ocean acidification), mass extinction and biodiversity loss, ecological crisis, and ecological collapse.
    - **Label**: `REFUTES`
    - **Prediction**: `NOT_ENOUGH_INFO` ❌
- TCFD disclosure classification
  - Example 1
    - **Text**: 1. Should our products fail to meet energy-efficiency standards and regulations, we will risk losing sales opportunities.
    - **Label**: `Strategy a)`
    - **Prediction**: `Strategy a)` ✅
  - Example 2
    - **Text**: There are no sentences in the provided excerpts that describe the targets the company uses to manage climate-related risks or opportunities.
    - **Label**: `Metrics and Targets c)`
    - **Prediction**: `Metrics and Targets b)`  🔧
  - Example 3
    - **Text**: Describe how processes for identifying, assessing, and managing climate-related risks are integrated into the organization’s overall risk management.
    - **Label**: `Risk Management c)`
    - **Prediction**: `Risk Management b)` 🔧
  - Example 4
    - **Text**: Reporting on such risks and opportunities is provided to.
    - **Label**: `Governance a)`
    - **Prediction**: `Risk Management a)` ❌

## Limitation

- Limited number of training data, especially for TCFD disclosure classification (fewer than 600 samples)
- For TCFD's task, the model is trained on a dataset without any non-climate related data, which is kind of unrealistic in real-world scenarios.
  - **Future improvement**: populate the dataset with non-climate related data (resembling the `None` label in `training_data.json`.)

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
