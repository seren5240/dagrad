from setuptools import setup, find_packages

setup(
    name="dagrad",
    version="0.0.1",
    python_requires=">=3.7",
    description="A project for DAG learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Chang Deng",
    author_email="changdeng@uchicago.edu",
    keywords=[
        "dagma", "notears", "topo", "causal discovery", 
        "bayesian network", "structure learning"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "dagma",
        "scipy",
        "python-igraph",
        "torch",
        "scikit-learn",
        "notears @ git+https://github.com/xunzheng/notears.git"
    ],
    url="https://github.com/Duntrain/dagrad",
    project_urls={
        "Homepage": "https://github.com/Duntrain/dagrad",
        "Documentation": "https://readthedocs.org",
        "Repository": "https://github.com/Duntrain/dagrad",
        "Issues": "https://github.com/Duntrain/dagrad/issues"
    },
)
