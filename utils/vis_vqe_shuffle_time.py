import matplotlib.pyplot as plt
import numpy as np
import os
from ground_energy import GroundEnergy as GE

color = ['#1f78b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']
color = [(0, 129, 204), (248, 182, 45), (0, 174, 187), (163, 31, 52), (44, 160, 44), (148, 103, 189)]
marker = ["o", "s", '+', "v", "^", "<", ">"]
color_grad = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
marker_grad = ["v", "^", "<", ">"]

# plt.style.use('seaborn-darkgrid')
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'bold', #or 'blod'
        }
from matplotlib import rcParams
rcParams.update(params)

iter_gaps = [1, 2, 4, 8, 16]
workers = [1, 2, 4, 8, 16]

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(13, 6.4))

log_dir_ideal = 'logs/vqe/ideal/h2/hpc'

aggre_strategy = ['baseline_nogroup_average', 'baseline_group_average', 'shuffle_nogroup_average', 'shuffle_group_average']
aggre_strategy_label = ['QUDIO', 'QUDIO+group', 'Shuffle-QUDIO', 'Shuffle-QUDIO+group']

## noiseless
print('Ideal:\n')
lines, line_labels = [], []

distance = 0.5
iter_gap = 32
worker = 4
shot = 100

data_aggre, data_time_aggre = [], []
for a, aggre in enumerate(aggre_strategy):
    data, data_time = [], []
    base_err = []
    for seed in range(5):
        base_energys = 201
        bases = np.load(os.path.join(log_dir_ideal, 'baseline_nogroup_average', str(seed), 'time'+str(1)+'_'+str(32)+'_0.0_{}.npy'.format(shot)))
        base_energy = np.load(os.path.join(log_dir_ideal, 'baseline_nogroup_average', str(seed), 'loss_{}_{}_0.0_{}_0_{}.npy'.format(1, 32, shot, distance)))
        base_estimate = np.load(os.path.join(log_dir_ideal, 'baseline_nogroup_average', str(seed), 'energy_{}_{}_0.0_{}_0_{}.npy'.format(1, 32, shot, distance)))
        base_err.append(np.abs(base_estimate - GE['H2'][1]))
        find = False
        for k in range(33, 200, 32):
            if base_energy[k] <= -6.5:
                find = True
                base_energys = min(base_energys, k+1)
                break
    
    err = []
    for seed in range(5):
        energy_seed = 201
        ts = np.load(os.path.join(log_dir_ideal, aggre, str(seed), 'time'+str(worker)+'_'+str(iter_gap)+'_0.0_{}.npy'.format(shot)))
        err.append(np.abs(np.load(os.path.join(log_dir_ideal, aggre, str(seed), 'energy_'+str(worker)+'_'+str(iter_gap)+'_0.0_{}_{}_{}.npy'.format(shot, 0, distance)))-GE['H2'][1]))
        energy_all = []
        for subworker in range(worker):
            if not os.path.exists(os.path.join(log_dir_ideal, aggre, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_{}_{}_{}.npy'.format(shot, subworker, distance))):
                print('Miss files')
            else:
                energy = np.load(os.path.join(log_dir_ideal, aggre, str(seed), 'loss_'+str(worker)+'_'+str(iter_gap)+'_0.0_{}_{}_{}.npy'.format(shot, subworker, distance)))
                energy_all.append(energy)
        energy = np.sum(energy_all, axis=0)
        find = False
        for k in range(33, 200, 32):
            if energy[k] <= -6.5:
                find = True
                energy_seed = min(energy_seed, k+1)
                break

        data.append(bases*base_energys/(ts*energy_seed))
        data_time.append(bases/(ts))
    print('Baseline={}, {}={}'.format(np.mean(base_err), aggre, np.mean(err)))
    data_aggre.append(np.max(data))
    data_time_aggre.append(np.max(data_time))
    ax.bar(np.arange(2)*4.5+a, [data_aggre[-1], data_time_aggre[-1]], width=1.0, fc=np.array(color[a])/255, label=aggre_strategy_label[a], edgecolor='k')


leg = ax.legend(fontsize=20, loc='upper left')
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('k')

plt.tight_layout()
# plt.savefig('figure/speedup_vqe_shuffle.pdf')
# plt.savefig('figure/speedup_vqe_shuffle.png')
plt.show()