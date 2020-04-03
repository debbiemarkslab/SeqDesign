# SeqDesign Examples

## Downloading Data  
> ./download_example_data.sh

## Training the model
> ./demo_train.sh

This script will run 100 training iterations on the beta-lactamase dataset (the full model runs 250,000).
The final model checkpoint will appear as three files in
`sess/BLAT_ECOLX_elu_channels-48_rseed-11_<timestamp>.ckpt-100*`

On an AWS p2.xlarge instance, this demonstration took 6 minutes.

## Predicting mutation effects
> ./demo_calc_logprobs.sh

This script will use the pretrained model weights in 
`sess/BLAT_ECOLX_v2_channels-48_rseed-11_19Aug16_0626PM.ckpt-250000*`
to make mutation effect predictions for the beta-lactamase mutational scan from
[Stiffler et al., Cell, 2015](https://doi.org/10.1016/j.cell.2015.01.035).

The final predictions are the average of 10 predictions (500 are used in the full test).
These predictions will appear in
`calc_logprobs/output/demo_BLAT_ECOLX_r24-286_Ranganathan2015_rseed-11_channels-48_dropoutp-0.5.csv`

On an AWS p2.xlarge instance, this demonstration took 7 minutes.

## Generating nanobody libraries
> ./demo_generate.sh


