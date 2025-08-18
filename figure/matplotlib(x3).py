import matplotlib.pyplot as plt
methods = ['VDSR', 'LapSRN', 'EDSR-baseline', 'IMDN',  'ShuffleMixer', 'DPSAN', 'SCPFAN(ours)']
flops = [612.6, 115, 160, 72, 43, 29.8, 1.01]  # FLOPs (G)
psnr = [33.66, 33.81, 34.37, 34.36, 34.40, 34.19,34.88]  # PSNR (dB)
params = [6.65, 2.9, 15.55, 7.03, 4.15, 1.95, 2.34]
circle_sizes = [param * 200 for param in params]
custom_colors = ['purple', 'blue', 'green', 'cyan', 'orange', 'red', 'lime']
plt.figure(figsize=(8, 8))
scatter = plt.scatter(flops, psnr, s=circle_sizes, c=custom_colors, alpha=0.8, edgecolors="w")
for i, method in enumerate(methods):
    x_offset = 50
    if method == 'SCPFAN(ours)':
        plt.text(flops[i], psnr[i] - 0.05, method, fontsize=18, va='top', ha='center', color='red')
    elif method in ['LapSRN', 'VDSR','ShuffleMixer']:
        plt.text(flops[i], psnr[i]+0.04 , method, fontsize=18, va='bottom', ha='center')
    elif method in ['EDSR-baseline']:
        plt.text(flops[i]+x_offset, psnr[i]-0.05 , method, fontsize=18, va='bottom', ha='left')
    else:
        plt.text(flops[i], psnr[i]-0.05 , method, fontsize=18, va='top', ha='center')
plt.title('PSNR vs. FLOPs vs. Params', fontsize=22)
plt.xlabel('Flops (G)', fontsize=19)
plt.ylabel('PSNR (dB)', fontsize=19)
plt.grid(alpha=0.5)
#plt.tight_layout()
plt.show()
