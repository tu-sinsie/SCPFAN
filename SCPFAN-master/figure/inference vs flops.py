import matplotlib.pyplot as plt
methods = ['CARN', 'SwinIR-light', 'SRFormer-light', 'BSRN', 'CAMixerSR', 'SCPFAN(Ours)']
flops = [91, 49.6, 62.8, 19.4, 44.6, 1.01]
inference_time = [904, 109, 159, 149, 133, 43]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(inference_time[:-1], flops[:-1], color='yellow', label="Other methods", edgecolors='black', alpha=0.8, s=60)
ax.scatter(inference_time[-1], flops[-1], color='red', marker='*', s=200, label='SCPFAN(Ours)', edgecolors='black')
for i, method in enumerate(methods):
    offset_x, offset_y = -1, -4
    if method == 'SwinIR-light':
        offset_x,offset_y = 10,0
    elif method == 'SRFormer-light':
        offset_y = 2
    fontsize = 18 if i < len(methods) - 1 else 16
    fontweight = 'bold' if i == len(methods) - 1 else 'normal'
    color = 'red' if i == len(methods) - 1 else 'black'
    ax.text(inference_time[i] + offset_x, flops[i] + offset_y, method,
            fontsize=fontsize, fontweight=fontweight, color=color, ha='left', va='center',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.6))
plt.legend(loc='lower right', fontsize=18, frameon=False)
plt.title('Flops vs. Inference time', fontsize=18)
plt.xlabel('Inference time (ms)', fontsize=18)
plt.ylabel('Flops (G)', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()