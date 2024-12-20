from setuptools import setup, find_namespace_packages

setup(
    name="mixed-precision-quantization",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
    ],
    author="Pierre Dantas",
    author_email="pierre.dantas@gmail.com",
    description="Mixed Precision Quantization for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pierredantas/llm-compression-toolkit/tree/main/quantization/MixedPrecisionQuantization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
