import matplotlib.pyplot as plt
params = {'legend.fontsize': 'medium',
          'figure.figsize': (10,6),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)

fig = plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.loglog(n_avg_array,mean1_array,'-*',label='Mean Over Freq.')
plt.loglog(n_avg_array,std1_array,'-*',label='StdDev Over Freq.')
plt.loglog(n_avg_array,mean_std_ratio,'-*',label='Mean/Std Ratio')
plt.loglog(n_avg_array,mean_std_true,'--',label='Expected Mean/Std')
plt.loglog(n_avg_array,std_expected,'--',label='Expected StdDev')
plt.xlim(.1,1000)
plt.ylim(.01,100)
# plt.xlabel('Number of Spectrum Averages')
plt.ylabel('Relative Values')
plt.title(case_str)
plt.legend(loc='upper left')
plt.grid()

plt.subplot(2,1,2)
plt.semilogx(n_avg_array,noncoh_snr_gain_db,'-*',label='Actual')
plt.semilogx(n_avg_array,noncoh_snr_gain_db_expected,'-*',label='Expected')
plt.xlim(.1,1000)
plt.ylim(0,20)
plt.xlabel('Number of Spectrum Averages')
plt.ylabel('Non-Coherent Gain dB')
# plt.title('Non-Coherent SNR Gain',fontsize='medium')
plt.figtext(.30,.42,'Non-Coherent SNR Gain',fontsize='large')
plt.legend(loc='upper left')
plt.grid()

plt.savefig(output_dir+f'02-NonCoh-gain-{i_case}'+'.png',bbox_inches='tight')

if display_figs:
    plt.show()
else:
    plt.close(fig)

fig = plt.figure(figsize=(10, 6))
plt.semilogx(n_avg_array,noncoh_snr_gain_db,'-*',label='Actual')
plt.semilogx(n_avg_array,noncoh_snr_gain_db_expected,'-*',label='Expected')
plt.xlim(1,1000)
plt.ylim(0,20)
plt.xlabel('Number of Spectrum Averages')
plt.ylabel('Non-Coherent Gain dB')
plt.title(case_str)
# plt.title('Expected and Observed Non-Coherent SNR Gain')
plt.legend(loc='upper left')
plt.grid()

plt.savefig(output_dir+f'03-NonCoh-gain-{i_case}'+'.png',bbox_inches='tight')

if display_figs:
    plt.show()
else:
    plt.close(fig)
