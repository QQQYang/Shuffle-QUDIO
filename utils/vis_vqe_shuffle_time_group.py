from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import os
from ground_energy import GroundEnergy as GE

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189)]
marker = ["o", "s", '+', "v", "^", "<", ">"]
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">"]
fontsize = 30

# plt.style.use('seaborn-darkgrid')
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'bold', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

iter_gaps = [1, 2, 4, 8]
workers = [1, 2, 4, 8]

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6.4, 6.4))

# # time
for i in range(1):
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
log_dir_ideal = 'logs/vqe/ideal/LiH/baseline_group_average_speedup'
log_dir_base = 'logs/vqe/ideal/LiH/baseline_nogroup_average_speedup'

for i, iter_gap in enumerate(iter_gaps):
    for worker in workers[1:]:
        t_group_seed, t_no_seed = [], []
        for seed in range(5):
            t_groups, t_nos = [], []
            for w in range(worker):
                t_group = np.load(os.path.join(log_dir_ideal, str(seed), 'time_iter'+str(worker)+'_'+str(iter_gap)+'_0.0_1000{}.npy'.format(w)))
                t_no = np.load(os.path.join(log_dir_base, str(seed), 'time_iter'+str(worker)+'_'+str(iter_gap)+'_0.0_1000{}.npy'.format(w)))
                t_groups.append(t_group)
                t_nos.append(t_no)
            t_group_seed.append(np.mean(t_groups))
            t_no_seed.append(np.mean(t_nos))
        print('worker={}, iter_gap={}, group/nogroup={}'.format(worker, iter_gap, np.mean(t_group_seed)/np.mean(t_no_seed)))

## noiseless
print('Ideal:\n')
lines, line_labels = [], []

bases, base_energys = 0, 201
distance = 0.5
for seed in range(5):
    base = np.load(os.path.join(log_dir_base, str(seed), 'time_all'+str(1)+'_'+str(1)+'_0.0_1000.npy'))[0]
    if base_energys == 201:
        bases = base
    base_energy = np.load(os.path.join(log_dir_base, str(seed), 'loss_{}_{}_0.0_1000_0_{}.npy'.format(1, 1, distance)))
    find = False
    for k in range(9, 200, 8):
        if base_energy[k] <= -6.4:
            find = True
            base_energys = min(base_energys, k+1)
            bases = base
            break

for i, iter_gap in enumerate([1]):
    data, data_time = [], []
    for worker in workers:
        ts = 0
        energy_seed = 201
        for seed in range(5):
            t = np.load(os.path.join(log_dir_ideal, str(seed), 'time_all'+str(worker)+'_'+str(iter_gap)+'_0.0_1000.npy'))[0]
            if energy_seed == 201:
                ts = t
            energy_all = []
            for subworker in range(worker):
                if not os.path.exists(os.path.join(log_dir_ideal, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_1000_{}_{}.npy'.format(subworker, distance))):
                    print('Miss files')
                else:
                    energy = np.load(os.path.join(log_dir_ideal, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_1000_{}_{}.npy'.format(subworker, distance)))
                    energy_all.append(energy)
            energy = np.sum(energy_all, axis=0)
            find = False
            for k in range(8+1, 200, 8):
                if energy[k] <= -6.4:
                    find = True
                    energy_seed = min(energy_seed, k+1)
                    ts = t
                    break

        data.append(bases*base_energys/(ts*energy_seed))
        data_time.append(bases/(ts))
    ax.plot(workers, data, linestyle='--', color=np.array(color[i+1])/255, marker=marker[i+1], linewidth=2.5, markersize=12, label='group')

    # ax[0].plot(workers, data_time, linestyle='--', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, markersize=12)


for i, iter_gap in enumerate([1]):
    data, data_time = [], []
    for worker in workers:
        ts = 0
        energy_seed = 201
        for seed in range(5):
            t = np.load(os.path.join(log_dir_base, str(seed), 'time_all'+str(worker)+'_'+str(iter_gap)+'_0.0_1000.npy'))[0]
            if energy_seed == 201:
                ts = t
            energy_all = []
            for subworker in range(worker):
                if not os.path.exists(os.path.join(log_dir_base, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_1000_{}_{}.npy'.format(subworker, distance))):
                    print('Miss files')
                else:
                    energy = np.load(os.path.join(log_dir_base, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_1000_{}_{}.npy'.format(subworker, distance)))
                    energy_all.append(energy)
            energy = np.sum(energy_all, axis=0)
            find = False
            for k in range(8+1, 200, 8):
                if energy[k] <= -6.4:
                    find = True
                    energy_seed = min(energy_seed, k+1)
                    ts = t
                    break

        data.append(bases*base_energys/(ts*energy_seed))
        data_time.append(bases/(ts))
    ax.plot(workers, data, linestyle='--', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, markersize=12, label='base')

    # ax[0].plot(workers, data_time, linestyle='--', color=np.array(color[i+1])/255, marker=marker[i+1], linewidth=2.5, markersize=12)

# base_ts, base_energys = [], float('inf')
# distance = 0.5
# for seed in range(5):
#     base_t = np.load(os.path.join(log_dir_ideal, str(seed), 'time'+str(1)+'_'+str(1)+'_0.0_0.npy'))[0]
#     base_ts.append(base_t)
#     base_energy = np.load(os.path.join(log_dir_ideal, str(seed), 'loss_{}_{}_0.0_0_0_{}.npy'.format(1, 1, distance)))
#     find = False
#     base_energys = min((base_energy[33]-GE['LiH'][1])*base_t, base_energys)


# for i, iter_gap in enumerate(iter_gaps):
#     data, data_time = [], []
#     for worker in workers:
#         ts = []
#         energy_seed = float('inf')
#         for seed in range(5):
#             t = np.load(os.path.join(log_dir_ideal, str(seed), 'time'+str(worker)+'_'+str(iter_gap)+'_0.0_0.npy'))[0]
#             ts.append(t)
#             energy_all = []
#             for subworker in range(worker):
#                 if not os.path.exists(os.path.join(log_dir_ideal, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_0_{}_{}.npy'.format(subworker, distance))):
#                     print('Miss files')
#                 else:
#                     energy = np.load(os.path.join(log_dir_ideal, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_0_{}_{}.npy'.format(subworker, distance)))
#                     energy_all.append(energy)
#             energy = np.sum(energy_all, axis=0)
#             energy_seed = min((energy[33]-GE['LiH'][1])*t, energy_seed)

#         data.append(base_energys/(energy_seed))
#         data_time.append(min(base_ts)/min(ts))
#     ax[1].plot(workers, data, linestyle='--', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, markersize=12, label='W={}'.format(iter_gap))

#     ax[0].plot(workers, data_time, linestyle='--', color=np.array(color[i])/255, marker=marker[i], linewidth=2.5, markersize=12)

# ax[0].plot(workers, workers, color='k', linewidth=2.5)#
ax.plot(workers, workers, color='k', linewidth=2.5, label='linear speedup')#


for i in range(1):
    ax.tick_params(labelsize=fontsize)
    # ax.set_title('(a)', fontsize=20)
    ax.set_yticks(workers)
    ax.set_xticks(workers)
    font = {
        'family':'Times New Roman',
        # 'serif':'Times New Roman',
        'style':'normal',
        'weight':'bold', #or 'bold'
    }
    ax.set_xlabel('Number of local nodes (K)', font, fontsize=fontsize)
    ax.grid(True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
# ax[0].set_title('Speedup to time', font, fontsize=fontsize)
ax.set_title('Speedup to accuracy', font, fontsize=fontsize)
# ax.legend(loc='best', ncol=2, fontsize=20, frameon=True, columnspacing=0.5)
# leg = plt.legend(handles=lines, ncol=1, fontsize=20, frameon=True, columnspacing=0.5, bbox_to_anchor=(-1, -1))
# leg.get_frame().set_linewidth(2)
# leg.get_frame().set_edgecolor('k')

# def export_legend(legend, filename="figure/qnn_legend.png"):
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox)
# export_legend(leg)

handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc=(0.15, 0.65), fontsize=20, frameon=True, ncol=1)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

plt.tight_layout()
plt.savefig('figure/speedup_vqe_group.pdf')
plt.savefig('figure/speedup_vqe_group.png')
plt.show()