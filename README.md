# SeqDesign

Biological sequence design for antibodies using deep learning.

## Installation

We recommend using SeqDesign with a GPU that supports CUDA, especially for training.
If a GPU is available, install the [TensorFlow GPU dependencies](https://www.tensorflow.org/install/gpu), 
then install the SeqDesign dependencies with:
```shell script
pip install -r requirements_gpu.txt
```

Using the [linux_setup.sh](linux_setup.sh) script, 
installation on Ubuntu 18.04 LTS took 5 minutes.

If no GPU is available, use:  
```shell script
pip install -r requirements.txt
```

Then install SeqDesign:
```shell script
python setup.py install
```

### Used software and versions tested:
- python - 2.7
- tensorflow - 1.12  
- numpy - 1.15  
- scipy - 0.19  
- sklearn - 0.18  

Tested on Ubuntu 18.04 LTS and CentOS 7.2

## Examples

See the [examples](examples) directory.


## Usage
Run each script with the `-h` argument to see additional arguments.

### Training

Given a fasta file of training sequences, run:
```shell script
run_autoregressive_fr <your_dataset>.fa
```

### Mutation effect prediction
Deterministic:
```shell script
calc_logprobs_seqs_fr --sess <your_sess> --dropout-p 1.0 --num-samples 1 --input <input>.fa --output <output>.csv
```

Average of 500 samples:
```shell script
calc_logprobs_seqs_fr --sess <your_sess> --dropout-p 0.5 --num-samples 500 --input <input>.fa --output <output>.csv
```

### Sequence generation
```shell script
generate_sample_seqs_fr --sess <your_sess>
```
