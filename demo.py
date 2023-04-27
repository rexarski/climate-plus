from minipackage import factcheck, tcfd_classify

text1 = [
    "Aker BP’s carbon price assumptions are significantly higher than the prices assumed in the IEA’s scenarios.",
    "Strategy c)",
]

text2 = [
    "Aker BP’s assumed carbon price reaches USD 235/tCO2 in 2030, assumed flat thereafter.",
    "Metrics and Targets a)",
]


def check_factcheck(text):
    print(
        f"[INPUT] {text[0]}\n[OUTPUT] {tcfd_classify.tcfd_classify(text[0])[0]}\n[TRUTH] {text[1]}\n"
    )


check_factcheck(text1)
check_factcheck(text2)

claim = "Obama administration's Clean Power Plan would have little or no effect on carbon dioxide emissions."

evidence = "In his announcement, Obama stated that the plan includes the first standards on carbon dioxide emissions from power plants ever proposed by the Environmental Protection Agency."

print(f"[CLAIM] {claim}\n[EVIDENCE] {evidence}")

print(f"[PREDICTION] {factcheck.factcheck(claim, evidence)[0]}")
