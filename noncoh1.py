import numpy as np

#
# generate spectrogram for coarse channel
#

n_sti=1
dc = dc_offset*dc_phase

if (i_case==1):
    n_pol=1
else:
    n_pol=2

[sg,freq,Hsq]=gen_simple_coarse_sg(n_time,n_freq,n_sti,H_edge_db,n_pol,dc,tone_freq,tone_snr_db)

#
# set up for variable averaging
#

if dc_reject:
    sg[:,n_freq//2]=0
  
col_sum0 = np.mean(sg,0)

col_sum0_db = db(col_sum0)
  
z_threshold = 10.
mean0 = np.mean(col_sum0)
std0 = np.std(col_sum0)
mean0_db = db(mean0)
threshold0_db = db(mean0 + z_threshold*std0)

log2_n_time = np.log2(n_time).astype(int)

n_avg_array = np.nan*np.ones(log2_n_time)
mean1_array = np.nan*np.ones(log2_n_time)
std1_array = np.nan*np.ones(log2_n_time)
mean_std_ratio = np.nan*np.ones(log2_n_time)
mean_std_true = np.nan*np.ones(log2_n_time)

std2_std_ratio = np.nan*np.ones(log2_n_time)
std2_std_true = np.nan*np.ones(log2_n_time)
std_expected = np.nan*np.ones(log2_n_time)

noncoh_snr_gain_db = np.nan*np.ones(log2_n_time)
noncoh_snr_gain_db_expected = np.nan*np.ones(log2_n_time)

#
# apply averaging over successive powers of 2
#

for i_avg in range(log2_n_time):
    n_avg = 2**(i_avg+1)

    sum1 = np.mean(sg[0:n_avg,:],0)

    #
    # apply split window 2 pass mean normalization
    #

    if sw2pm_enable:
        [norm1, Hsq_est] = sw2pm(sum1,201,11,shear_threshold)
        local_mean_est = 1
    else:
        norm1 = sum1
        Hsq_est = np.nan*np.ones(n_freq)
        local_mean_est = 0

    sum1_db = db(sum1)
    norm1_db = db(norm1)
    Hsq_db = db(Hsq)
    Hsq_est_db = db(Hsq_est)

    mean1 = np.mean(norm1)
    std1 = np.std(norm1)
   
    if shear_enable:
        shear_limit = mean1 + std1*shear_threshold
        if (1):
            # tracks expected snr gain slightly better for low # avgs
            norm_shear = np.minimum(norm1,shear_limit)
        else:
            norm_shear = norm1
            ii_shear = norm1>shear_limit
            norm_shear[ii_shear] = mean1
        std1 = np.std(norm_shear)
    
    mean1_db = db(mean1)
    threshold1_db = db(mean1 + z_threshold*std1)

    if (n_avg==2):
        std1_2 = std1

    n_avg_array[i_avg] = n_avg
    mean1_array[i_avg] = mean1
    std1_array[i_avg] = std1
    mean_std_ratio[i_avg] = mean1/std1
    mean_std_true[i_avg] = np.sqrt(x_avg*n_avg)
    
    std2_std_ratio[i_avg] = std1_2/std1
    std2_std_true[i_avg] = np.sqrt(n_avg/2)
    std_expected[i_avg] = std1_2/np.sqrt(n_avg/2)
    
    noncoh_snr_gain_db[i_avg] = db(mean_std_ratio[i_avg])
    noncoh_snr_gain_db_expected[i_avg] = 5*np.log10(x_avg*n_avg)  # approx
    
    result_str1 = f'n-avg={n_avg:4d} mean={mean1:.4f} std-dev={std1:.3f} mean/std={mean_std_ratio[i_avg]:6.3f} vs. {mean_std_true[i_avg]:6.3f} std2/std={std2_std_ratio[i_avg]:6.3f} vs. {std2_std_true[i_avg]:6.3f}'
    print(f'{result_str1}')

    result_str2 = f'n-avg={n_avg:4d} mean={mean1:.4f} std-dev={std1:.3f}\nmean/std={mean_std_ratio[i_avg]:6.3f} vs. {mean_std_true[i_avg]:6.3f} std2/std={std2_std_ratio[i_avg]:6.3f} vs. {std2_std_true[i_avg]:6.3f}'

    if special_plot_case:
        # plot psd original and normalized for this value of n_avg
        exec(open("./noncoh_plot1.py").read())
    
