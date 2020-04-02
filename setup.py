import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="SeqDesign",
    version="0.0.1",
    author="Aaron Kollasch",
    author_email="awkollasch@gmail.com",
    description=("Autoregressive protein models to "
                 "predict mutation effects and generate novel sequences"),
    license="MIT",
    keywords="autoregressive protein model generative",
    url="https://github.com/debbiemarkslab/seqdesign",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
    ],
    python_requires='==2.7',
)
