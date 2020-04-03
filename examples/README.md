# SeqDesign Examples

## Downloading Data  
> ./download_example_data.sh

## Training the model
> ./demo_train.sh

This will run 100 training iterations on the beta-lactamase dataset (the full model runs 250,000).
The final model checkpoint will appear as three files in
`sess/BLAT_ECOLX_elu_channels-48_rseed-11_<timestamp>.ckpt-100*`
On an AWS p2.xlarge instance, this demonstration takes about 6 minutes.

## Predicting mutation effects
> ./demo_calc_logprobs.sh



## Generating nanobody libraries


