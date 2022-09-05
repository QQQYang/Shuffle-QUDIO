# Shuffle-QUDIO: accelerate distributed VQE with enhanced trainability

This repository is the official implementation of [*Shuffle-QUDIO: accelerate distributed VQE with enhanced trainability*].

>The variational quantum eigensolver (VQE) is a leading strategy that exploits noisy intermediate-scale quantum (NISQ) machines to tackle chemical  problems outperforming classical computers. To gain such computational advantages on large-scale problems, a feasible solution is the \textbf{QU}antum \textbf{DI}stributed \textbf{O}ptimization (QUDIO) scheme, which partitions the original problem into $K$ subproblems and allocates them to $K$ quantum  machines followed by the parallel optimization. Despite the provable acceleration rate, the efficiency of QUDIO may heavily degrade  by  the synchronization operation. To conquer this issue, here we propose Shuffle-QUDIO by involving shuffle operations on local Hamiltonians during the quantum distributed optimization. Compared with QUDIO, Shuffle-QUDIO significantly reduces the communication frequency among quantum processors and simultaneously achieves better trainability. Particularly, we prove that Shuffle-QUDIO enables a faster convergence rate over QUDIO. Moreover, we conduct extensive numerical experiments to verify that Shuffle-QUDIO allows both a wall-clock time speedup and low  approximation error in the tasks of estimating the ground state energy of molecule. We further exhibit that our proposal can be seamlessly integrated with other acceleration techniques, such as operator grouping, to further improve the training efficacy of  VQE.

![scheme](./scheme.png)

## Requirements:

```setup
pip install -r requirements.txt
```

## Usage

### Run Shuffle-QUDIO

```bash
python train_pl_torch.py --K 1 --M 1000 --p 0 --W 2 --port 200 --seed 0 --mol LiH --random -1 --aggre average
```