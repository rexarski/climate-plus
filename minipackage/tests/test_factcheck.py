from minipackage import factcheck

text1 = "says Sweet, who has authored several sea-level rise studies."
text2 = "Over the 21st century, the IPCC projects that in a very high emissions scenario the sea level could rise by 61–110 cm."


def test_factcheck():
    labels = factcheck.factcheck(text1, text2)
    assert labels in (["SUPPORTS"], ["REFUTES"], ["NOT_ENOUGH_INFO"])
