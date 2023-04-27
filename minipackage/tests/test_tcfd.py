from minipackage import tcfd_classify

text1 = "Locally, our representatives are in dialogue with governments and authorities as part of the ongoing interaction between authorities and the business community."


def test_tcfd():
    labels = tcfd_classify.tcfd_classify(text1)
    assert labels[0] in [
        "Governance a)",
        "Governance b)",
        "Metrics and Targets a)",
        "Metrics and Targets b)",
        "Metrics and Targets c)",
        "Risk Management a)",
        "Risk Management b)",
        "Risk Management c)",
        "Strategy a)",
        "Strategy b)",
        "Strategy c)",
    ]
