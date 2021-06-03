import pickle, glob, os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/mnt/home/daviso53/Research/')
#from pyci_pairing_plus_ph import *
import pyci.imsrg_ci.pyci_p3h as pyci

sns.set_context('paper',
                rc={'font.size':20,'axes.titlesize':15, 'axes.labelsize':15, 'lines.linewidth':2.5},
                font_scale=1.5)
#
#PALETTE = None
dataPath= sys.argv[1]

parsed = dataPath.split('/')
g = float(parsed[1].replace('g',''))
pb = float(parsed[2].replace('pb',''))

n_holes = 4
n_particles = 4

#states = pyci.init_states()
hme = pyci.matrix(n_holes,n_particles, 0.0, 1.0, g, pb)
ev_eigs1 = np.linalg.eigvalsh(hme)
true = ev_eigs1[0]
true_spec = ev_eigs1

fileList = glob.glob(dataPath+'plot_data_*_states.p')

#PALETTE = sns.color_palette('rocket', len(fileList))
PALETTE = None

fig,ax = plt.subplots(figsize=(16,8))
df_list = []
for filePath in fileList:
    print('Loading from file '+filePath)
    num_states = str(int(filePath.replace(dataPath,'').split('_')[2]))
    data_dic = pickle.load(open(filePath, 'rb'))
    data_df = pd.DataFrame(data_dic)

    for i in range(len(true_spec)):
        data_df['eig'+str(i)] = data_df.apply(lambda row: row.eig_list[i], axis=1)
        data_df['dag'+str(i)] = data_df.apply(lambda row: row.dag_list[i], axis=1)
    
    data_df['num_states'] = [num_states for x in range(data_df.shape[0])]
    df_list.append(data_df)

full_df = pd.concat(df_list).reset_index(drop=True)
full_df = full_df.sort_values(by='num_states').reset_index(drop=True)

# PLOT EIGENVALUE FLOW
sns.lineplot(x='s_vals', y=full_df['eig0']/true, data=full_df, hue='num_states', linewidth=3, ax=ax)#, palette=PALETTE)
ax.axhline(1,linewidth=3, color='red', linestyle='--')
#plt.axhline(true,linewidth=3)
ax.set_ylim(top=2)
ax.set_ylabel('eig0/true')
fig.suptitle('Ground state eigenvalue flow varying ensemble size')
#ax.legend([path.replace(dataPath,'').split('_')[2]+'-state ensemble' for path in fileList])
fig.savefig(dataPath+'eig0_flow.png')


fig,ax = plt.subplots(figsize=(16,8))
# full_df = []
# for filePath in fileList:
#     print('Plotting from file '+filePath)
#     data_dic = pickle.load(open(filePath, 'rb'))
#     data_df = pd.DataFrame(data_dic)

#     for i in range(len(true_spec)):
#         data_df['dag'+str(i)] = data_df.apply(lambda row: row.dag_list[i], axis=1)


# PLOT DIAGONAL FLOW
sns.lineplot(x='s_vals', y=full_df['dag0'], data=full_df, linewidth=3, hue='num_states', ax=ax)#, palette=PALETTE)
#plt.axhline(1,linewidth=3)
ax.axhline(true,linewidth=3, color='red', linestyle='--')
#plt.ylim(top=2)
ax.set_ylabel('0th diagonal element')
fig.suptitle('0th diagonal element flow varying ensemble size')
#ax.legend(fileList)
fig.savefig(dataPath+'dag0_flow.png')


for filePath in fileList:
    n_states = int((filePath.split('/')[-1].split('_'))[2])

    fig,ax = plt.subplots(figsize=(16,8))

    print('Plotting from file '+filePath)
    data_dic = pickle.load(open(filePath, 'rb'))
    data_df = pd.DataFrame(data_dic)

    for i in range(len(true_spec)):
        data_df['dag'+str(i)] = data_df.apply(lambda row: row.dag_list[i], axis=1)


        # PLOT DIAGONAL FLOW
        sns.lineplot(x='s_vals', y=data_df['dag'+str(i)], data=data_df, linewidth=3, color='black', ax=ax)
        #plt.axhline(1,linewidth=3)
    ax.axhline(true,linewidth=3, color='red', linestyle='--')
    ax.set_ylabel('diagonal ME')
    fig.suptitle('Diagonal spectrum flow {:n}-state ensemble'.format(n_states))
    #plt.xlim([0,2])
    #plt.legend(range(36))
    fig.savefig(dataPath+'dag_me_flow_{:n}.png'.format(n_states))

    fig,ax = plt.subplots(figsize=(16,8))
    for i in range(len(true_spec)):
        sns.lineplot(x='s_vals', y=data_df['dag'+str(i)]-true_spec[i], data=data_df, color='black', linewidth=3, ax=ax)

    last_spec = data_df.dag_list.iloc[-1]
    MSE = np.sqrt((true_spec-last_spec).T.dot(true_spec-last_spec))/len(true_spec)
    ax.set_ylabel('diagonal ME - true')
    ax.legend(['MSE: {: .4f}'.format(MSE)])
    fig.suptitle('Diagonal spectrum error flow {:n}-state ensemble'.format(n_states))
    fig.savefig(dataPath+'dag_me_error_flow_{:n}.png'.format(n_states))

    df = pd.DataFrame({'eigN':range(len(true_spec)), 'error':true_spec-last_spec, 'error^2':np.multiply(true_spec-last_spec,true_spec-last_spec)})
    pickle.dump(df, open(dataPath+'error_spec_{:n}.p'.format(n_states), 'wb'))
        
    
    fig,ax = plt.subplots(figsize=(16,8))
    entropy_df = data_df[['s_vals', 'entropy', 'entropy_ex1', 'entropy_ex2']]
    melt = entropy_df.melt('s_vals', var_name='CI_eigenstate', value_name='entanglement_S')

    sns.lineplot(x='s_vals', y='entanglement_S',hue='CI_eigenstate', data=melt, linewidth=3, ax=ax)#hue='num_states', ax=ax)
    ax.axhline(0.0, linewidth=3, color='red', linestyle='--')
    #ax.set_ylabel('entropy')
    fig.suptitle('IM-SRG(2) flow dependent single-particle entanglement entropy\nPairing model g={: .2f}, pb={: .2f}'.format(g, pb))
    #ax.legend(['single ground state reference', 'optimized 3-state ensemble'])
    fig.savefig(dataPath+'entropy_flow_{:n}.png'.format(n_states))
