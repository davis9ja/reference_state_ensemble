import reference_ensemble as re
import sys, os, glob

NUM_STATES_IN_REF = int(sys.argv[1])
G_VAL = float(sys.argv[2])
PB_VAL = float(sys.argv[3])
GENERATOR = sys.argv[4]
OUT_FNAME = str(NUM_STATES_IN_REF)+'_states.p'

data_dic = {'s_val':[], 'dag_list':[], 'eig_list':[], 'trc_list':[], 'ham_list':[]}
DATA_DIR = './{}/g{:0.2f}/pb{:0.2f}/'.format(GENERATOR, G_VAL, PB_VAL)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

ensemble = re.ReferenceEnsemble(NUM_STATES_IN_REF, 2.0, 0.0, 'white', '', 'vac_coeffs')
opt_x = ensemble.optimize_reference(1)
ref1 = ensemble.refs.T.dot(opt_x)
print('Found optimal reference state: {d}'.format(d=ref1))

if not os.path.exists(DATA_DIR+'plot_data_{}'.format(OUT_FNAME)):

    states = init_states()

    main(4,4,g=G_VAL, ref=ref1, flow_data_log=1, generator=GENERATOR)

    # dag_list = []
    # eig_list = []
    # trc_list = []
    # ham_list = []
    # s_vals = []
    for i in range(0,10000):
        fname = 'vac_coeffs_flow_c{}.p'.format(i)
        if not os.path.exists(fname):
            continue
        else:
            print('Loading file {}'.format(fname))
            s, H0B, H1B, H2B, eta1B_vac, eta2B_vac = pickle.load(open(fname, 'rb'))
            hme = matrix(states, H0B, H1B, H2B, H2B)
            ev_eigs1 = np.linalg.eigvalsh(hme)
            
            data_dic['s_vals'].append(s)
            data_dic['eig_list'].append(ev_eigs1)
            data_dic['dag_list'].append(np.diag(hme))
            data_dic['trc_list'].append(np.trace(hme.conj().T.dot(hme)))
            data_dic['ham_list'].append(hme)
            
    # s_vals = np.asarray(s_vals)
    # eig_list = np.asarray(eig_list)
    # dag_list = np.asarray(dag_list)
    # trc_list = np.asarray(trc_list)
    # ham_list = np.asarray(ham_list)

    pickle.dump(data_dic, open(DATA_DIR+'plot_data_{}'.format(OPT_PARAM_FNAME), 'wb'))

fileList = glob.glob('./vac_coeffs_flow_c*.p')

for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)


