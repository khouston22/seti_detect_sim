import numpy as np
import pandas as pd
import std_fns as s

def detect1D(norm_psd,f0,df,n_subband,norm_mean,norm_std,z_det=10.,n_nearest_det=101, verbose=False):
	"""
	[n_det,det_out]=def detect1D(norm_psd,f0,df,n_subband,norm_mean,norm_std,z_det,n_nearest_det)

	simple detection function with normalized power spectral density vector

	inputs:

	norm_psd	(nf)     					normalized spectrum vector of length nf from sw2pm
	f0, df	                 				freq of norm_psd[0], freq increment
	n_subband                 				number of freqency subbands for mean std calc
												n_subband must evenly divide nf
	norm_mean	(n_subband) or scalar		mean of norm_psd with sheared tones removed
	norm_std	(n_subband) or scalar		std dev of norm_psd with sheared tones removed
	z_det	                 				number of std deviations at threshold
	n_nearest_det                 			minimum separation in freq bins between detections

	outputs:

	n_det	                 				number of detections
	det_out		DataFrame					DataFrame of new detections, or []

    Columns of det_out:

	f_index				index for detection frequency [0,nf)
	margin_db			SNR margin above threshold = snr_db - threshold_snr_db
	snr_db				net SNR estimate in dB
	snr_coh_db			SNR estimate in dB at input (prior to noncoherent integration)
	noncoh_gain_db		estimated SNR gain in dB for noncoh integration
	peak_db				norm peak value10*log10(peak_value)
	norm_mean_db		10*log10(norm_mean_value) 
	threshold_snr_db	snr corresponding to threshold =  
	peak_value			peak value o fnorm
	norm_mean_value		norm_mean value of subband (close to 1.0)
	norm_std_value		norm_std value of subband
	threshold_value		threshold value of subband 

	"""

	norm_mean = s.make_ndarray(norm_mean)
	norm_std = s.make_ndarray(norm_std)

	if (s.length(norm_mean)!=n_subband):
		raise Exception(f'Error in detect1D, {s.length(norm_mean)=} not equal to {n_subband=}\n')
	if (s.length(norm_std)!=n_subband):
		raise Exception(f'Error in detect1D, {s.length(norm_std)=} not equal to {n_subband=}\n')
	
	nf = len(norm_psd)
	
	if verbose: print(f'In detect1D')

	n_det = 0
	det_out = []

	nf_seg = nf//n_subband
		
	for i_seg in range(n_subband):
		
		ii = range(i_seg*nf_seg,(i_seg+1)*nf_seg)
		work = np.copy(norm_psd[ii])

		threshold_value = norm_mean[i_seg] + z_det*norm_std[i_seg]
		
		ii_det = np.nonzero(work > threshold_value)[0]

		if verbose: 
			print(f'{i_seg=} of {n_subband}')
			print(f'ii_det={ii_det}')

		while (len(ii_det)>0):
			n_det += 1
			norm_peaks = work[ii_det]
			i_peak = np.argmax(norm_peaks)
			f_index = i_seg*nf_seg + ii_det[i_peak]
			f_peak = f0 + df*f_index
			peak_value = norm_peaks[i_peak]
			norm_mean_value = norm_mean[i_seg]
			norm_std_value = norm_std[i_seg]
			norm_mean_db = 10.*np.log10(norm_mean_value)
			snr_coh_db = 10.*np.log10(peak_value/norm_mean_value - 1.)
			noncoh_gain_db = 10.*np.log10(norm_mean_value/norm_std_value)
			snr_db = snr_coh_db + noncoh_gain_db
			peak_db	= 10.*np.log10(peak_value)
			threshold_db = 10.*np.log10(threshold_value)
			threshold_snr_db = 10.*np.log10(threshold_value/norm_mean_value - 1.) + noncoh_gain_db
			margin_db = snr_db - threshold_snr_db

			# add detection to table

			det_out=add_detection(det_out,f_peak,f_index,margin_db,snr_db,snr_coh_db,noncoh_gain_db,peak_db,
				  threshold_db,peak_value,threshold_value,norm_mean_value,norm_std_value)
			
			if verbose: 
				print(f'Detection # {n_det}')
				print(f'{norm_peaks=}')
				print(f'{f_index=},{margin_db=:.3f},{snr_db=:.3f},{snr_coh_db=:.3f},{noncoh_gain_db=:.3f},{peak_db=:.3f}')
				print(f'{threshold_db=:.3f},{peak_value=:.3f}')
				print(f'{norm_mean_value=:.3f},{norm_std_value=:.3f},{threshold_value=:.3f}')
				print(f'det_out=')
				print(det_out)
				
			# remove region above threshold near peak

			ii_det_min = ii_det[i_peak] - n_nearest_det
			ii_det_max = ii_det[i_peak] + n_nearest_det
			ii_keep = np.nonzero((ii_det>ii_det_max)|(ii_det<ii_det_min))[0]
			ii_det = ii_det[ii_keep]

			if verbose: print(f'Secondary ii_det={ii_det}')
		

	if (s.length(det_out)==0):
		n_det = 0
	else:
		n_det = det_out.shape[0]

	return n_det,det_out


def add_detection(det_in,f_peak,f_index,margin_db,snr_db,snr_coh_db,noncoh_gain_db,peak_db,
				  threshold_db,peak_value,threshold_value,norm_mean_value,norm_std_value):
	"""
	det_out=add_detection(det_in,f_peak,f_index,margin_db,snr_db,snr_coh_db,noncoh_gain_db,peak_db,
				  threshold_db,peak_value,threshold_value,norm_mean_value,norm_std_value)

	adds detection to det_in dataframe

	inputs:

	det_in		DataFrame		input DataFrame to be appended, or []
	Columns of det_in and det_out:
	f_peak						detection frequency
	f_index						index for detection frequency [0,nf)
	margin_db					SNR margin above threshold = snr_db - threshold_snr_db
	snr_db						net SNR estimate in dB
	snr_coh_db					SNR estimate in dB at input (prior to noncoherent integration)
	noncoh_gain_db				estimated SNR gain in dB for noncoh integration
	peak_db						norm peak value 10*log10(peak_value)
	threshold_db				det threshold value 10*log10(threshold_value)  
	peak_value					peak value of norm
	threshold_value				threshold value of subband 
	norm_mean_value				norm_mean value of subband (close to 1.0)
	norm_std_value				norm_std value of subband

	outputs:

	n_det	                 				total number of detections
	det_out		DataFrame					input DataFrame to be appended, or []

	"""

	if (s.length(det_in)==0):	# if empty
		n_det = 1;
	else:
		n_det = det_in.shape[0] + 1

	det = pd.DataFrame({
				"freq": [f_peak],"f_index": [f_index],"margin_db": [margin_db],"snr_db": [snr_db],"snr_coh_db": [snr_coh_db],
				"noncoh_gain_db": [noncoh_gain_db],"peak_db": [peak_db],"threshold_db": [threshold_db],
				"peak_value": [peak_value],"threshold_value": [threshold_value],"norm_mean_value": [norm_mean_value],
				"norm_std_value": [norm_std_value],},
			index=[n_det-1],)

	if (n_det>1):
		det_out = pd.concat([det_in,det], axis=0)
	else:
		det_out = det

	return det_out
