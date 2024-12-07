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
plt.plot(freq,sum1_db,'-',label='Mean Spectrum')
plt.plot(freq,Hsq_db,'-',label='Coarse Channel Hsq(f)')
if local_mean_est:
    plt.plot(freq,Hsq_est_db,'-r',label='Local Mean Estimate')

plt.xlim(-.5,.5)
plt.ylim(-5.,20.)
#plt.xlabel('Normalized Frequency within Coarse Channel')
plt.ylabel('Amplitude dB')
plt.title(case_str)
plt.figtext(.15,.82,result_str2,fontsize=10)
plt.legend(loc='upper right')
plt.grid()

plt.subplot(2,1,2)
plt.plot(freq,norm1_db,'-',label='Normalized Spectrum')
plt.plot([-.5,.5],mean1_db*np.ones(2),'--',label='Mean',linewidth=2.5)
plt.plot([-.5,.5],threshold1_db*np.ones(2),'-r',label=f'Threshold z={z_threshold:.0f}',linewidth=2.5)
plt.xlim(-.5,.5)
plt.ylim(-5.,20.)
plt.xlabel('Normalized Frequency within Coarse Channel')
plt.ylabel('Amplitude dB')
# plt.title('After Normalization')
plt.figtext(.15,.42,'After Normalization',fontsize='large')
plt.legend(loc='upper right')
plt.grid()

if special_plot_case:
    plt.savefig(output_dir+f'04-NonCoh-normalization-{i_case}-{i_avg}'+'.png',bbox_inches='tight')
else:
    plt.savefig(output_dir+f'01-NonCoh-normalization-{i_case}'+'.png',bbox_inches='tight')

if display_figs:
    plt.show()
else:
    plt.close(fig)
