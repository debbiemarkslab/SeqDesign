import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="seqdesign",
    version="1.0.0",
    author="Aaron Kollasch",
    author_email="awkollasch@gmail.com",
    description="Protein design and variant prediction using autoregressive generative models",
    license="MIT",
    keywords="autoregressive protein sequence design deep learning generative",
    url="https://github.com/debbiemarkslab/seqdesign",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    scripts=[
        "bin/calc_logprobs_seqs_fr",
        "bin/calc_logprobs_seqs_nanobody",
        "bin/generate_sample_seqs_fr",
        "bin/library_selection_birch",
        "bin/process_output_sequences",
        "bin/run_autoregressive_fr",
        "bin/make_birch_feature_matrix",
    ],
)