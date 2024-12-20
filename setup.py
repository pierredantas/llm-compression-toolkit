from setuptools import setup, find_packages

setup(
    name="llm-compression-toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.0.0",
    ],
    author="Pierre Dantas",
    author_email="pierre.dantas@gmail.com",
    description="A toolkit for compressing and quantizing language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pierredantas/llm-compression-toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
