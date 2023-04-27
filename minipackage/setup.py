from setuptools import setup, find_packages

setup(
    name="minipackage",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
    description="A package for classifications using a pre-trained BERT model",
    author="Rui Qiu",
    author_email="rexarski@gmail.com",
    url="https://github.com/rexarski/climate-plus",
)
