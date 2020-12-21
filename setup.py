from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="karyogram_extraction", # Replace with your own username
    version="1.0.0",
    author="Lukas Uzolas",
    author_email="lukas@uzolas.com",
    description="Package that allows for extraction of chromosomes from karygrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="lukas.uzolas.com",
    packages=find_packages(),
)