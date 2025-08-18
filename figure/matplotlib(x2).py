import matplotlib.pyplot as plt
methods = ['VDSR', 'LapSRN', 'EDSR-baseline', 'IMDN',  'ShuffleMixer', 'DPSAN', 'SCPFAN(ours)']
flops = [613, 30, 316, 161, 91, 65.32, 1.00]  # FLOPs (G)
psnr = [37.53, 37.52, 37.99, 38.00, 38.01, 37.87,38.02]  # PSNR (dB)
params = [6.66, 2.51, 13.70, 6.94, 3.94, 1.95, 2.34]
circle_sizes = [param * 200 for param in params]
custom_colors = ['purple', 'blue', 'green', 'cyan', 'orange', 'red', 'lime']
plt.figure(figsize=(8, 8))
scatter = plt.scatter(flops, psnr, s=circle_sizes, c=custom_colors, alpha=0.8, edgecolors="w")
for i, method in enumerate(methods):
    if method == 'SCPFAN(ours)':
        plt.text(flops[i], psnr[i] - 0.02, method, fontsize=18, va='top', ha='center', color='red')
    elif method in ['LapSRN', 'VDSR']:
        plt.text(flops[i], psnr[i]+0.03 , method, fontsize=18, va='bottom', ha='center')
    elif method in ['ShuffleMixer']:
        plt.text(flops[i], psnr[i] + 0.01, method, fontsize=16, va='center', ha='center')
    else:
        plt.text(flops[i], psnr[i]-0.03 , method, fontsize=18, va='top', ha='center')
plt.title('PSNR vs. FLOPs vs. Params', fontsize=22)
plt.xlabel('Flops (G)', fontsize=19)
plt.ylabel('PSNR (dB)', fontsize=19)
plt.grid(alpha=0.5)
#plt.tight_layout()
plt.show()
