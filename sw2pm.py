import numpy as np
import pandas as pd
import std_fns as s

def sw2pm(x,n_window=201,n_gap=5,shear=2.3,calc_stats=False,n_subband=1):
	"""
	[norm_psd, bkgnd, norm_mean, norm_std]=sw2pm(x,n_window,n_gap,shear,calc_stats,n_subband)

	function to implement split-window 2-pass mean

	inputs

	x		    (nt,nf) or (nf)		Input vector/matrix of nt row vectors of length nf
	n_window	            		total width of split window in samples (odd#)
	n_gap	                		gap width in samples (odd#)
									total number of zero samples within gap
	shear                 			shearing threshold
	calc_stats                 		=True to calculate norm_mean, norm_std
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
		raise Exception(f'Error in sw2pm, x not a matrix {xdim=}\n')

	if (nf < n_window):
		raise Exception(f'Error in sw2pm, {nf=} < {n_window=}\n')

	n1 = n_gap//2
	n2 = n_window//2

	scale1 = shear/(2*(n2 - n1))
	scale2 = 1/(2*(n2 - n1))

	l1a = 0
	l1b = nf
	l2a = l1a + n2 - n1
	l2b = l1b + n2 - n1
	l3a = l1a + n2 + n1 + 1
	l3b = l1b + n2 + n1 + 1
	l4a = l1a + n2 + n2 + 1
	l4b = l1b + n2 + n2 + 1

	for it in range(nt):

		if ndim==2:
			x1 = x[it,:]
		else:
			x1 = x

		#
		# determine shear threshold = first sliding average*shear
		# use cumulative sum for split window averaging filter
		# bkgnd array = shear threshold = shear*1st sliding average
		#

		x_cum = np.cumsum(np.concatenate(	
			([0.],np.ones(n2)*np.mean(x1[0:n2]),x1,np.ones(n2)*np.mean(x1[-n2:-1]))))
		
		bkgnd1 = scale1*(	x_cum[l4a:l4b] - x_cum[l3a:l3b] 
						+ 	x_cum[l2a:l2b] - x_cum[l1a:l1b])
		
		#
		# shear data - replace data by background if 
		# x value exceeds its shear threshold
		#
		# work array = sheared x vector
		
		work = np.copy(x1)

		ii_shear = np.nonzero(x1 > bkgnd1)
		work[ii_shear] = bkgnd1[ii_shear]/shear
		
		#
		# do second sliding average on sheared x vector
		#

		x_cum = np.cumsum(np.concatenate(	
			([0.],np.ones(n2)*np.mean(work[0:n2]),work,np.ones(n2)*np.mean(work[-n2:-1]))))
		
		bkgnd1 = scale2*(	x_cum[l4a:l4b] - x_cum[l3a:l3b] 
						+ 	x_cum[l2a:l2b] - x_cum[l1a:l1b])
		
		#
		# normalize data
		#

		work=np.copy(bkgnd1)
		
		ii_z = np.nonzero(bkgnd1<=0.)
		work[ii_z] = 1.

		if ndim==2:
			norm_psd[it,:] = x1/work
			bkgnd[it,:] = bkgnd1
		else:
			norm_psd = x1/work
			bkgnd = bkgnd1
		
		#
		# calculate stats on sheared norm
		#

		if calc_stats:

			nf_seg = nf//n_subband
			
			for i_seg in range(n_subband):

				ii = range(i_seg*nf_seg,(i_seg+1)*nf_seg)
				if ndim==2:
					work = np.copy(norm_psd[it,ii])
				else:
					work = np.copy(norm_psd[ii])

				#
				# do simple 3-pass mean and std over frequency with simple replacement by shear threshold
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
				
				if ndim==2:
					norm_mean[it,i_seg] = mean1 
					norm_std[it,i_seg] = std1
				else:
					if (n_subband>1):
						norm_mean[i_seg] = mean1
						norm_std[i_seg] = std1
					else:
						norm_mean = mean1
						norm_std = std1
		
	if calc_stats:
		return norm_psd, bkgnd, norm_mean, norm_std
	else:
		return norm_psd, bkgnd
	

