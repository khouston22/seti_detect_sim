import numpy as np
from numpy.random import randn, rand, randint, seed

import scipy as sp
from scipy.fft import fft, fftshift


def gen_simple_coarse_sg(n_time,n_freq,n_sti=1,H_edge_db=0.,n_pol=2,dc_offset=0.,tone_freq=[],tone_snr_db=[],seed_value=22):

	"""
	[sg,freq,Hsq]=gen_simple_coarse_sg(n_time,n_freq,n_sti,H_edge_db,n_pol,dc_offset,tone_freq,tone_snr_db,seed_value)

	function to generate simple spectrogram for a coarse channel
	assumes baseband (complex) input noise
	Note: simple parabolic PFB response, does not consider aliasing from adjacent PFB channels

	inputs

	n_time		    			Number of rows in original spectrogram
	n_freq		    			Number of frequency points spectrogram
	n_sti		    			Number of successsive lines to be averaged in spectrogram
	H_edge_db					PFB Crossover value in dB at freq=-.5 and .5
	n_pol						Number of polarizations added together
	dc_offset	            	DC offset to apply
	tone_freq	(n_tone)	    normalized tone frequency, -.5 < tone_freq[i] < .5
	tone_snr_db (n_tone)	    tone snr in dB after combining polarizations
	seed_value	            	rng seed value

	outputs

	sg		    (nt,n_freq)     normalized matrix of n_time//n_sti row vectors of length n_freq
	freq		(n_freq)        normalized frequency vector -.5 <= freq < .5
	Hsq 		(n_freq)        PFB mag squared frequency response (simple parabolic model)

	"""

	seed(seed_value)

	nt = n_time//n_sti

	sg = np.zeros((nt,n_freq),dtype=float)

	freq = np.linspace(-.5,.5,n_freq,endpoint=False,dtype=float)
	
	if np.isscalar(tone_freq):
		tone_freq = np.asarray([tone_freq])
	else:
		tone_freq = np.asarray(tone_freq)
	n_tone = len(tone_freq)

	if np.isscalar(tone_snr_db):
		tone_snr_db = np.asarray([tone_snr_db])
	else:
		tone_snr_db = np.asarray(tone_snr_db)
	if (len(tone_snr_db)==1):
		tone_snr_db = tone_snr_db*np.ones(n_tone)
	elif (len(tone_snr_db)==0):
		n_tone = 0
	

	print(f"{tone_freq=}\n")
	print(f"{tone_snr_db=}\n")

	#
	# generate raw noise spectrogram
	#

	for it in range(nt):

		for i_sti in range(n_sti):
			v0x = (randn(n_freq) + 1j*randn(n_freq))/np.sqrt(2.) + dc_offset
			if (n_pol==1):
				v0y = np.zeros((n_freq)) + dc_offset
			else:
				v0y = (randn(n_freq) + 1j*randn(n_freq))/np.sqrt(2.) + dc_offset

			#
			# generate spectrogram
			#

			sg[it,:] += fftshift((abs(fft(v0x,n_freq))**2 + abs(fft(v0y,n_freq))**2)/(n_freq*n_sti))

	#
	# apply parabolic model for PFB frequency response
	#

	if (abs(H_edge_db)>0.):
		Hsq_delta = 1 - 10**(.1*H_edge_db)
		Hsq = 1 - Hsq_delta*abs(freq/.5)**2
		sg = sg*np.outer(np.ones((nt,1)),Hsq)
	else:
		Hsq = np.ones((n_freq),dtype=float)

	#
	# add in tones
	#

	if n_tone>0:
		ii_freq = np.rint((tone_freq/.5)*n_freq/2).astype(int) + n_freq//2
		tone_level = n_pol*Hsq[ii_freq]*10**(.1*tone_snr_db)
		sg[:,ii_freq] += tone_level


	return sg, freq, Hsq


