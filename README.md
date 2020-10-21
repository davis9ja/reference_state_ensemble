# reference state ensemble

#### PURPOSE:

This code implements a least-squares optimization procedure to find the best reference state ensemble for the IM-SRG(2). By ``best'', we mean the ensemble that
maximizes the *reduction* of truncation error, due to correlation information lost in the truncation of operators above 2-body. The purpose of the ensemble
is to add that correlation information *back into* the IM-SRG flow by incorporating correlations into the reference state.

