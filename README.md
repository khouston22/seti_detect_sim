# seti_detect_sim: Spectrum Normalizatin and Thresholding Simulation

This repository contains Jupyter notebooks for evaluating normalization algorithms for narrowband SETI.
These algorithms estimate the local noise level and create detection thresholds.

A problem was noticed in seticore whereby SNR did not increase as expected with increasing averaging
time.  The simulations in this repository were used to evaluate the problem and develop an 
improved normalization algorithm which was implemented in seticore2.

Reference: K. M. Houston, "Fine-Tuning the Narrowband SETI Signal Processing Pipeline", 
75th International Astronautical Congress (IAC), Milan, Italy, 14-18 October 2024.

## Notebooks description

Notebook "noncoh_avg.ipynb" was used to evaluate the root cause of problem: the polyphase
filter bank (PFB), which creates each coarse channel, has a non-flat frequency response.  However, the 
original seticore baseline implicitly assumes a flat frequency response.  When an overall coarse channel 
spectrum mean and standard deviation are computed, it is found that after a certain number of averages, 
the standard deviation does not decrease with more averaging.  This is because the standard 
deviation is a measure of the "average" (root-mean-square) deviation from a constant mean value, which is
non-zero even for zero noise (infinite averaging) due to the PFB curvature.  This causes the detection threshold to
stop decreasing, resulting in lost sensitivity and an inaccurate SNR estimate.  A similar effect
happens with very large DC offsets.  See the above reference.

Proper normalization effectively flattens the spectrum.  In "noncoh_avg.ipynb", a well-established method called 
split-window two-pass normalization is applied which results in desired behavior.

Notebook "sg_norm_eval.ipynb" goes further and compares a simpler method of normalization, where 
the coarse channel is divided into a large number of subbands.  Statistics are estimated for each subband,
and a separate detection threshold is determined for each subband.  With a smooth PFB response and a narrow
subband bandwidth, the spectrum is essentially flat in each subband.  The detection threshold descreases 
as expected with more averaging.  The SNR estimate also increases as expected with more averaging.

Related repositories include:

https://github.com/khouston22/seti_end_to_end

https://github.com/khouston22/seticore2 (a fork of seticore https://github.com/lacker/seticore)

https://github.com/khouston22/seti_test_file_gen,

https://github.com/UCBerkeleySETI/rawspec, and





