These codes and sample datasets accompany subsection S5.2 of the paper arXiv:2507.17980

They are provided to reproduce the calculations of $p_2$, $q_6$, and entropy across multiple (or all) timesteps of a trajectory.

The resulting descriptors can then be combined with the C-index coefficients stored in the provided frozen JSON file.
These coefficients were trained on a single snapshot from a different replica (or from a system with comparable physics) and can be directly applied to compute the C-index for each atom based on its $p_2$, $q_6$, and entropy values.

You can see code/scripts/replica_train_apply.py to apply trained C-index from replica 1 to replica 2. 