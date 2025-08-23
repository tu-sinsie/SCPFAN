import matplotlib.pyplot as plt
methods = ['CARN', 'SwinIR-light', 'SRFormer-light', 'BSRN', 'CAMixerSR', 'SCPFAN(Ours)']
params = [1592, 897, 873, 352, 765, 236]
inference_time = [904, 109, 159, 149, 133, 43]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(inference_time[:-1], params[:-1], color='pink', label="Other methods", edgecolors='black', alpha=0.8, s=60)
ax.scatter(inference_time[-1], params[-1], color='red', marker='*', s=200, label='SCPFAN(Ours)', edgecolors='black')
for i, method in enumerate(methods):
    offset_x, offset_y = 15, 5  #
    if method == 'SwinIR-light':
        offset_y = 70
    elif method == 'SRFormer-light':
        offset_y = -20
    elif method == 'CAMixerSR':
        offset_y = -50
    fontsize = 18 if i < len(methods) - 1 else 16
    fontweight = 'bold' if i == len(methods) - 1 else 'normal'
    color = 'red' if i == len(methods) - 1 else 'black'
    ax.text(inference_time[i] + offset_x, params[i] + offset_y, method,
            fontsize=fontsize, fontweight=fontweight, color=color, ha='left', va='center',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.6))  # 背景框防止重叠
plt.legend(loc='lower right', fontsize=18, frameon=False)
plt.title('Params vs. Inference time', fontsize=18)
plt.xlabel('Inference time (ms)', fontsize=18)
plt.ylabel('Params (K)', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
