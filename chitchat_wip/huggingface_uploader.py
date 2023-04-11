import datasets

tcfd_dataset = datasets.load_dataset("csv", data_files="data/tcfd_output.csv")
tcfd_dataset.push_to_hub("rexarski/TCFD_disclosure")
