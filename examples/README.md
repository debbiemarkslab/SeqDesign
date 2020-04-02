# SeqDesign Examples

## Downloading Data


## Predicting mutation effects


## Generating libraries

### Relevant scripts and their description:

- `calc_logprobs_seqs_nanobody.py` <br>&emsp;
    calculates log probabilities for all nanobody sequences in a user-provided fasta file 

- `calc_logprobs_seqs_fr.py` <br>&emsp;
    calculates log probabilities for all sequences in a user-provided fasta file 

- `calc_logprobs_seqs_nanobody.py` <br>&emsp;
    calculates log probabilities for all nanobody sequences in a user-provided fasta file 

- `./code/*` <br>&emsp;
    scripts required to run `calc_logprobs_seqs_nanobody.py`

- `./library_selection/birch_funcs.py` <br>&emsp;
    BIRCH clustering code used to cluster
    the nanobody sequences. This code was originally derived from sklearn, but
    it was modified to run on extremely large datasets. 

- `./library_selection/generate_sample_seqs_fr.py` <br>&emsp;
    code for ancestral sampling of
    CDR3 sequences from the germline nanobody sequence

- `./library_selection/helper_birch.py` <br>&emsp;
    code for running BIRCH clustering on the
    processed nanobody k-mer counts after processing by process_output_sequences.py

- `./library_selection/process_output_sequences.py` <br>&emsp;
    script to process all of the
    randomly generated sequences from generate_sample_seqs_fr.py. This code makes
    sure that these sequences were not in the starting library, have the correct
    final framework region, have the right other library constraints, and finally
    processes these sequences into k-mer vectors for downstream processing.



### Used packages and their licenses:
- tensorflow - Apache License
- numpy - New BSD License
- scipy - New BSD License
- sklearn - New BSD License
