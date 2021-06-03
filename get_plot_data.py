import reference_ensemble as re
import sys, os, glob
import pickle
import numpy as np

#sys.path.append('/mnt/home/daviso53/Research/')
#from pyci_pairing_plus_ph import *
import pyci.imsrg_ci.pyci_p3h as pyci

#sys.path.append('/mnt/home/daviso53/Research/im-srg_tensorflow/')
from tfimsrg.main import main
from pyci.density_matrix.density_matrix import density_1b

NUM_STATES_IN_REF = int(sys.argv[1])
G_VAL = float(sys.argv[2])
PB_VAL = float(sys.argv[3])
GENERATOR = sys.argv[4]
OUT_FNAME = str(NUM_STATES_IN_REF)+'_states.p'

data_dic = {'s_vals':[], 'dag_list':[], 'eig_list':[], 'trc_list':[], 'ham_list':[], 'entropy':[], 'entropy_ex1':[], 'entropy_ex2':[]}
DATA_DIR = './{}/g{:0.2f}/pb{:0.2f}/'.format(GENERATOR, G_VAL, PB_VAL)
OUTPUT_ROOT = 'temp_{:n}{:0.2f}{:0.2f}{:s}/'.format(NUM_STATES_IN_REF, G_VAL, PB_VAL, GENERATOR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(OUTPUT_ROOT):
    os.mkdir(OUTPUT_ROOT)

n_holes=4
n_particles=4

if NUM_STATES_IN_REF == 1:
    # ensemble = re.ReferenceEnsemble(36,n_holes,n_particles, G_VAL, PB_VAL, GENERATOR, '', 'vac_coeffs')
    # hme = pyci.matrix(n_holes, n_particles, 0.0, 1.0, G_VAL, PB_VAL)
    # w,v = np.linalg.eigh(hme)
    # ref1 = ensemble.refs.T.dot(v[:,0]*v[:,0])
    ref1 = None
    opt_x = 1
else:
    ensemble = re.ReferenceEnsemble(NUM_STATES_IN_REF,n_holes,n_particles, G_VAL, PB_VAL, GENERATOR, '', 'vac_coeffs')
    opt_x,lsq = ensemble.optimize_reference(int(np.ceil(NUM_STATES_IN_REF/2)))
    ref1 = ensemble.refs.T.dot(opt_x)
#ref1 = [0,0,0,0,1,1,1,1]
#opt_x = 1
print("""Optimization complete.
         Ensemble states in ref   = {:n}
         G_VAL                    = {:0.2f}
         PB_VAL                   = {:0.2f}
         GENERATOR                = {:s}
         Optimal weights          = {g}
         optimal reference state  = {d}
""".format(NUM_STATES_IN_REF, G_VAL, PB_VAL, GENERATOR, g=opt_x, d=ref1))



if not os.path.exists(DATA_DIR+'plot_data_{}'.format(OUT_FNAME)):


    main(n_holes,n_particles,g=G_VAL, pb=PB_VAL, ref=ref1, flow_data_log=1, generator=GENERATOR, output_root=OUTPUT_ROOT)

    # dag_list = []
    # eig_list = []
    # trc_list = []
    # ham_list = []
    # s_vals = []
    for i in range(0,10000):
        fname = OUTPUT_ROOT+'vac_coeffs_flow_c{}.p'.format(i)
        if not os.path.exists(fname):
            continue
        else:
            print('Loading file {}'.format(fname))
            s, H0B, H1B, H2B, eta1B_vac, eta2B_vac = pickle.load(open(fname, 'rb'))
            hme = pyci.matrix(n_holes,n_particles, H0B, H1B, H2B, H2B, imsrg=True)
            w, v = np.linalg.eigh(hme)
            
            rho1b = density_1b(n_holes,n_particles, weights=v[:,0])
            wdens, vdens = np.linalg.eigh(rho1b)
            S = -1*sum([n*np.log(n) for n in wdens])

            rho1b = density_1b(n_holes,n_particles, weights=v[:,3])
            wdens, vdens = np.linalg.eigh(rho1b)
            S1 = 0
            for n in wdens:
                if abs(n) < 10**-10:
                    S1 += 0
                else:
                    S1 += -n*np.log(n)

            rho1b = density_1b(n_holes,n_particles, weights=v[:,14])
            wdens, vdens = np.linalg.eigh(rho1b)
            S2 = 0
            for n in wdens:
                if abs(n) < 10**-10:
                    S2 += 0
                else:
                    S2 += -n*np.log(n)

            data_dic['s_vals'].append(s)
            data_dic['eig_list'].append(w)
            data_dic['dag_list'].append(np.diag(hme))
            data_dic['trc_list'].append(np.trace(hme.conj().T.dot(hme)))
            data_dic['ham_list'].append(hme)
            data_dic['entropy'].append(S)
            data_dic['entropy_ex1'].append(S1)
            data_dic['entropy_ex2'].append(S2)
            
    # s_vals = np.asarray(s_vals)
    # eig_list = np.asarray(eig_list)
    # dag_list = np.asarray(dag_list)
    # trc_list = np.asarray(trc_list)
    # ham_list = np.asarray(ham_list)

    pickle.dump(data_dic, open(DATA_DIR+'plot_data_{}'.format(OUT_FNAME), 'wb'))

fileList = glob.glob(OUTPUT_ROOT+'vac_coeffs_*.p')

for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

try:
    os.rmdir(OUTPUT_ROOT)
except:
    print("Error while deleting file : ", OUTPUT_ROOT)
