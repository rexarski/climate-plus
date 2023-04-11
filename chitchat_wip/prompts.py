from langchain.prompts import PromptTemplate

template = """Extract 5 most representative sentences from the given source context to reflect its TCFD classification, which is Strategy. The sentences should be exactly the same as original documents.
===========
[SOURCE]: STRESS TESTING OUR PORTFOLIO AGAINST THIRD-PARTY CLIMATE SCENARIOS
STRESS TESTING OUR PORTFOLIO AGAINST THIRD-PARTY CLIMATE SCENARIOS

To stress test our portfolio, AES identified third-party scenarios covering varying degrees of transition and physical risk. We ultimately selected the International Energy Agency’s (IEA) 2017 World Energy Outlook (WEO) for transition risk scenarios, and for physical risk scenarios, we selected the Representative Concentration Pathways (RCPs) established by the Intergovernmental Panel on Climate Change’s (IPCC) Fifth Assessment Report (AR5). The temperature ranges indicated represent the projected increase in global surface temperatures from pre-industrial levels. As these two sets of scenarios are not formally harmonized, we have grouped them into the scenario conventions for purposes of the stress test and this report. Please see Building the Scenarios for more information.

The TCFD and other proponents of scenario planning for climate change impacts have highlighted the importance of using recognized third-party scenarios. While the scenarios may not be aligned with AES’ view of the future, our stress test uses the assumptions and outputs of third-party frameworks referenced by the TCFD as directly as possible so that investors can more easily compare companies.

Our climate resilience stress test is fundamentally an in-depth financial analysis assessing the sensitivity of gross margin across our entire business – from every individual plant, up through to our strategic business units. Our effort was guided by a steering group consisting of members from our financial planning and analysis, corporate risk and strategy, sustainability, legal, operations and other teams.

OUR PORTFOLIO IS NOT ONLY RESILIENT, BUT POSITIONED FOR GROWTH

Given our modest exposure to direct carbon risk and our pivot toward Clean Energy Growth Platforms, transition risk can enhance our upside potential.

The stress test highlights the effectiveness of our efforts to mitigate climate change risk given the decreasing portion of our margin that is directly exposed to carbon pricing as we move from the simulated Business as Usual Scenario (3-6°C) to a 1.5-2°C Scenario. In the results that follow, direct carbon exposed margin refers to margin that has the potential to be directly and negatively affected by, or has been subject to, a price on carbon. Even in the 1.5-2°C Scenario, where carbon prices reach $125/tonne for emerging economies and $140/ tonne for advanced economies by 2040, our direct carbon exposed margin is virtually zero. Under this scenario, our existing thermal plants are considered to be retired at the end of their anticipated useful life or contracted for reliability with the off-taker bearing the cost of carbon. The majority of the margin from these plants comes from capacity payments, which are not directly carbon exposed. These payments are essentially for availability and are received regardless of the amount of energy generated. However, these plants have indirect carbon exposure if the credit quality of our off-takers deteriorates due to carbon pricing. Please see Focusing on Reliability and De-Risking our Thermal Assets and Transition Risk Resilience for more information.
===========
[ANSWER]:
To stress test our portfolio, AES identified third-party scenarios covering varying degrees of transition and physical risk.
We ultimately selected the International Energy Agency’s (IEA) 2017 World Energy Outlook (WEO) for transition risk scenarios, and for physical risk scenarios, we selected the Representative Concentration Pathways (RCPs) established by the Intergovernmental Panel on Climate Change’s (IPCC) Fifth Assessment Report (AR5).
The TCFD and other proponents of scenario planning for climate change impacts have highlighted the importance of using recognized third-party scenarios.
Our effort was guided by a steering group consisting of members from our financial planning and analysis, corporate risk and strategy, sustainability, legal, operations and other teams.
However, these plants have indirect carbon exposure if the credit quality of our off-takers deteriorates due to carbon pricing.

----------

Extract 5 most representative sentences from the given source context to reflect its TCFD classification, which is {disclosure}. The sentences should be exactly the same as original documents.
===========
{summaries}
===========
[ANSWER]:
"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "disclosure"]
)
