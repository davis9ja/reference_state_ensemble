import pickle
import pandas as pd
import seaborn as sns

sns.set_context('paper')

data_dic = pickle.load(open('white/g2.0/pb0.0/plot_data_2_states.p', 'rb'))
data_df = pd.DataFrame(data_dic)



# PLOT EIGENVALUE FLOW
fig = plt.figure(figsize=[10,10])
annt_list = []
for i in range(eig_list.shape[1]):
    plt.plot(s_vals, eig_list[:, i], marker='o')
    
    count = 0
    for x,y in zip(s_vals, eig_list[:,i]):
        
        label = "{:.2f}".format(y)
        # annotation = plt.annotate(label, # this is the text
        #                           (x,y), # this is the point to label
        #                           textcoords="offset points", # how to position the text
        #                           xytext=(0,10), # distance from text to points (x,y)
        #                           ha='center') # horizontal alignment can be left, right or center
        if count % int(len(s_vals)/2) == 0:
            annt_list.append(plt.text(x,y,label, ha='center', va='center'))
        count += 1
plt.axhline(true[0])
plt.title('Pairing model eigenvalue flow')
plt.xlabel('flow param')
plt.ylabel('eigenvalue')
#plt.ylim([-6,11])
#plt.xlim([0,1.2])
plt.tight_layout()
plt.savefig('eig_flow.png')

# PLOT DIAGONAL MATRIX ELEMENT FLOW
fig = plt.figure(figsize=[10,10])
annt_list = []
for i in range(dag_list.shape[1]):
    plt.plot(s_vals, dag_list[:, i], marker='o')
    
    count = 0
    for x,y in zip(s_vals, dag_list[:,i]):
        
        label = "{:.2f}".format(y)
        # annotation = plt.annotate(label, # this is the text
        #                           (x,y), # this is the point to label
        #                           textcoords="offset points", # how to position the text
        #                           xytext=(0,10), # distance from text to points (x,y)
        #                           ha='center') # horizontal alignment can be left, right or center
#        if count % int(len(s_vals)/2) == 0:
#            annt_list.append(plt.text(x,y,label, ha='center', va='center'))
        count += 1
plt.axhline(true[0])
plt.title('Pairing model diagonal ME flow')
plt.xlabel('flow param')
plt.ylabel('diagonal ME')
#plt.ylim([-2,11])
#plt.xlim([0,1])
plt.tight_layout()
plt.savefig('dag_flow.png')


# PLOT TRACE FLOW
# true_trace = np.sum(np.diag(np.eye(dag_list.shape[1])))
# fig = plt.figure(figsize=[10,10])
# plt.plot(s_vals, trc_list/true_trace, marker='o')
# #plt.axhline(true_trace)
# plt.title('Pairing model trace flow')
# plt.xlabel('flow param')
# plt.ylabel('tr(H)')
# plt.tight_layout()
# plt.savefig('trc_flow.png')
