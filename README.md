# reference state ensemble

### PURPOSE:

This code implements a least-squares optimization procedure to find the best reference state ensemble for the IM-SRG(2) solving the pairing model Hamiltonian. By ''best``, we mean the ensemble that maximizes the *reduction* of truncation error, due to correlation information lost in the truncation of operators above 2-body. The purpose of the ensemble is to add that correlation information *back into* the IM-SRG flow by incorporating correlations into the reference state.

### TO RUN:

Must have `gitlab.msu.edu/daviso53/tfimsrg` and `gitlab.msu.edu/daviso53/pyci/` installed. Clone the repositories and add their locations to your PYTHONPATH via:

`export PYTHONPATH="$PYTHONPATH:/path/to/code"`

Then, run the ensemble optimization via:

`python get_plot_data.py <NUM_STATES_IN_ENSEMBLE> <G_VAL> <PB_VAL> <GENERATOR>`

For generator options, refer to TFIMSRG docs.

### PLOT DATA:

`python plot_data.py <GENERATOR>/g<G_VAL>/pb<PB_VAL>`

