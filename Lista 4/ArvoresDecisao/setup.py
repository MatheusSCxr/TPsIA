from setuptools import setup, find_packages

setup(
    name="ArvoreDecisao",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    author="Matheus S. C.",
    description="Implementações recursivas das árvores de decisão: ID3, C4.5 e CART",
    python_requires=">=3.8",
)