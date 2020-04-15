#!/bin/bash
cd calc_logprobs || exit
calc_logprobs_seqs_fr --sess BLAT_ECOLX_v2_channels-48_rseed-11_19Aug16_0626PM --checkpoint 250000 \
  --channels 48 --r-seed 11 --dropout-p 0.5 --num-samples 10 \
  --input input/BLAT_ECOLX_r24-286_Ranganathan2015.fa \
  --output output/demo_BLAT_ECOLX_r24-286_Ranganathan2015_rseed-11_channels-48_dropoutp-0.5.csv