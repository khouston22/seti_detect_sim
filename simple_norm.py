import numpy as np
import pandas as pd
import std_fns as s

def simple_norm(x,shear=2.3,n_subband=1):
	"""
	[norm_psd, bkgnd, norm_mean, norm_std]=simple_norm(x,shear,n_subband)

	function to implement simple local-mean spectrum normalizer

	inputs

	x		    (nt,nf) or (nf)		Input vector/matrix of nt row vectors of length nf
	shear                 			shearing threshold
	n_subband                 		number of freqency subbands for mean std calc
										n_subband must evenly divide nf

	outputs

	norm_psd	(nt,nf) or (nf)     normalized vector/matrix of nt row vectors of length nf
	bkgnd		(nt,nf) or (nf)		background vector/matrix of of nt row vectors of length nf
	norm_mean	(nt,n_subband) or (n_subband) or scalar		mean of norm_psd with sheared tones removed
	norm_std	(nt,n_subband) or (n_subband)or scalar		std dev of norm_psd with sheared tones removed

	"""

	xdim = np.shape(x)
	ndim = len(xdim)
	if (ndim==2):
		nt = np.shape(x)[0]
		nf = np.shape(x)[1]
		norm_psd = np.zeros((nt,nf))
		bkgnd = np.zeros((nt,nf))
		norm_mean = np.zeros(nt,n_subband)
		norm_std = np.zeros(nt,n_subband)
	elif (ndim==1):
		nt = 1
		nf = np.shape(x)[0]
		norm_psd = np.zeros(nf)
		bkgnd = np.zeros(nf)
		if (n_subband>1):
			norm_mean = np.zeros(n_subband)
			norm_std = np.zeros(n_subband)
		else:
			norm_mean = 0
			norm_std = 0
	else:
		raise Exception(f'Error in simple_norm(), x not a matrix {xdim=}\n')

	nf_seg = nf//n_subband
	if (nf_seg*n_subband != nf):
		raise Exception(f'Error in simple_norm(), {n_subband=} does not evenly divide {nf=}\n')
	
	for it in range(nt):

		if ndim==2:
			x1 = x[it,:]
		else:
			x1 = x
		
		#
		# calculate stats on subbands of x1
		#
		
		for i_seg in range(n_subband):

			ii = range(i_seg*nf_seg,(i_seg+1)*nf_seg)
			if ndim==2:
				work = np.copy(x1[it,ii])
			else:
				work = np.copy(x1[ii])

			#
			# do simple 3-pass mean and std over frequency before normalization
			# with simple replacement by shear threshold
			#

			mean1 = np.mean(work)
			std1 = np.std(work)
			shear_limit = mean1 + std1*shear
			
			work = np.minimum(work,shear_limit)
			mean1 = np.mean(work)
			std1 = np.std(work)
			shear_limit = mean1 + std1*shear
			
			work = np.minimum(work,shear_limit)
			mean1 = np.mean(work)
			std1 = np.std(work)
			
			#
			# normalize spectrogram by subband mean
			# and compute mean and std of normalized subbands
			#

			if ndim==2:
				norm_psd[it,ii] = x1[it,ii]/mean1
				bkgnd[it,ii] = mean1*np.ones(nf_seg)
				norm_mean[it,i_seg] = 1. 
				norm_std[it,i_seg] = std1/mean1
			else:
				norm_psd[ii] = x1[ii]/mean1
				bkgnd[ii] = mean1*np.ones(nf_seg)
				if (n_subband>1):
					norm_mean[i_seg] = 1.
					norm_std[i_seg] = std1/mean1
				else:
					norm_mean = 1.
					norm_std = std1/mean1
		
	return norm_psd, bkgnd, norm_mean, norm_std
	
	

