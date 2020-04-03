# SeqDesign Examples

## Downloading example data  
```shell script
./download_example_data.sh
```

This script will download the following files for 
training, mutation prediction, and sequence generation:  

- `datasets/sequences/BLAT_ECOLX_1_b0.5_lc_weights.fa`
- `datasets/nanobodies/Manglik_filt_seq_id80_id90.fa`
- `datasets/nanobodies/Manglik_labelled_nanobodies.txt`
- `calc_logprobs/input/BLAT_ECOLX_r24-286_Ranganathan2015.fa`
- `sess/BLAT_ECOLX_v2_channels-48_rseed-11_19Aug16_0626PM.ckpt-250000*`
- `sess/nanobody.ckpt-250000*`

## Training the model
```shell script
./demo_train.sh
```

This script will run 100 training iterations on the β-Lactamase sequence dataset
(the full model runs 250,000 iterations).
The final model checkpoint will appear as three files in
`sess/BLAT_ECOLX_elu_channels-48_rseed-11_<timestamp>.ckpt-100*`

On an AWS p2.xlarge instance, this demonstration took 6 minutes.

## Predicting mutation effects
```shell script
./demo_calc_logprobs.sh
```

This script will use the pretrained model weights in 
`sess/BLAT_ECOLX_v2_channels-48_rseed-11_19Aug16_0626PM.ckpt-250000*`
to make mutation effect predictions for the β-Lactamase mutational scan from
[Stiffler et al., Cell, 2015](https://doi.org/10.1016/j.cell.2015.01.035).

The final predictions are the average of 10 predictions 
(500 are used in the full test).
These predictions will appear in
`calc_logprobs/output/demo_BLAT_ECOLX_r24-286_Ranganathan2015_rseed-11_channels-48_dropoutp-0.5.csv`

On an AWS p2.xlarge instance, this demonstration took 7 minutes.

## Generating nanobody libraries
```shell script
./demo_generate.sh
```

This will generate nanobody CDR3 and FRA4 sequences given a preceding VH sequence.
The full nanobody sequences will be output in
`generated/nanobody.ckpt-250000_temp-1.0_rseed-42.fa` 

On an AWS p2.xlarge instance, this demonstration took 6.5 minutes.
