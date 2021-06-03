import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import sys
import itertools
import pickle

#sys.path.append('/mnt/home/daviso53/Research/')
#from pyci_pairing_plus_ph import *
import pyci.imsrg_ci.pyci_p3h as pyci

#sys.path.append('/mnt/home/daviso53/Research/im-srg_tensorflow/')
from tfimsrg.main import main

    
class ReferenceEnsemble(object):
    """Create a reference state ensemble for the pairing plus particle-hole model 
    Hamiltonian. Ensemble is a matrix-vector product that generates a 
    reference state from a linear combination of reference state configurations
    determined by the pairing model."""

    def __init__(self, num_states_in_ref, n_holes, n_particles, g_val, pb_val, generator, outfile_name, coeffs_root):
        """Class constructor.
        
        Arguments:
        
        num_states_in_ref -- number of configurations to include in the reference
        g_val -- pairing strength
        pb_val -- pair-breaking strength
        generator -- IM-SRG(2) generator (kwarg passed to IM-SRG-TN
        outfile_name -- (not implemented currently)
        coeffs_root -- directory to store IM-SRG-TN output"""

        self._number_states_in_ref = num_states_in_ref
        self._g_val = g_val
        self._pb_val = pb_val
        self._generator = generator
        self._outfile_name = outfile_name
        self._coeffs_root = coeffs_root

        self._n_holes= n_holes
        self._n_particles = n_particles

        ref_matrix = self.generate_ref_matrix(n_holes,n_particles)

        assert num_states_in_ref <= ref_matrix.shape[0]

        self._refs = ref_matrix[0:num_states_in_ref, :]

        #states = pyci.init_states()
        hme = pyci.matrix(n_holes,n_particles, 0.0, 1.0, g_val, pb_val)
        self._true = np.linalg.eigvalsh(hme)


    @property
    def refs(self):
        """Returns:
        
        refs -- matrix of S=0 reference state configs"""
        return self._refs

    def generate_ref_matrix(self, n_holes, n_particles):
        """Generate a matrix of basis configs from
        which the reference state ensemble will be
        built. We only use the basis config where total
        spin S = 0.

        Arguments:

        n_holes -- number of hole states in the model
        n_particles -- number of particle states in the model

        Returns:
        
        ref_matrix -- each row is a basis configuration"""

        gs_config = np.append(np.ones(n_holes),
                              np.zeros(n_particles))
        gs_config_str = ''.join([str(int(state)) for state in gs_config])

        refs = list(map("".join, itertools.permutations(gs_config_str)))
        refs = list(dict.fromkeys(refs)) # remove duplicates
        refs = [list(map(int, list(ref))) for ref in refs]

        # # COLLECT TOTAL SPIN = 0 REFS
        refss0 = []
        for ref in refs:
            S = 0
            for i in range(len(ref)):
                if ref[i] == 1:
                    if i%2 == 0:
                        S += 1
                    else:
                        S += -1
                else: continue
            if S == 0:
                refss0.append(ref)

            
        refs = refss0

        refs.sort(key=self.number_of_pairs, reverse=True)        
        #refs = refs[0:num_states_in_ref]
        
        #indices = np.r_[0,1,5,27,28]

        #refs = refs[0,1,5,27,28]
        #refs = np.asarray(refs)
        #refs = refs[np.r_[0,1,5,27,28], :]

        ref_matrix = np.asarray(refs)

        return ref_matrix

    def number_of_pairs(self, state):
        """Key for sorting permutations.

        Returns:
        
        pairs -- number of pairs in ref state config"""

        pairs = 0
        for i in range(0,len(state),2):
            if state[i] == 1 and state[i+1] == 1:
                pairs += 1
        return pairs

    def residual(self, x, y, num_targets):
        """Residual function to pass into least-squares optimizer.
        Computes the eigenvalues obtained from a full IM-SRG-TN flow. Then,
        returns the absolute error in the target number of eigenvalues. True
        eigenvalue targets are computed from FCI.
        
        Arguments:
        
        x -- probability weights
        y -- true eigenvalues
        num_targets -- number of eigenvalues to target in optimization

        Returns:
        
        absolute error in target eigenvalues"""
        
        x = x/sum(x)  # normalize weights

        # RUN IM-SRG(2)
        ref = self._refs.T.dot(x)
        main(self._n_holes,self._n_particles, 
             g=self._g_val, 
             pb=self._pb_val, 
             ref=ref, 
             verbose=0, 
             generator=self._generator,
             output_root = self._coeffs_root)

        # LOAD EVOLVED COEFFICIENTS
        H0B, H1B, H2B, eta1B_vac, eta2B_vac = pickle.load(open(self._coeffs_root+'/vac_coeffs_evolved.p', 'rb'))

        # PERFORM FULL CI AND GET EIGENVALUES
        hme = pyci.matrix(self._n_holes,self._n_particles, H0B, H1B, H2B, H2B, imsrg=True)
        ev_eigs = np.linalg.eigvalsh(hme)

        #return np.sqrt(np.mean((ev_eigs-y)**2))
        #return abs(ev_eigs[0:num_targets] - y[0:num_targets])
        #return abs(ev_eigs[1] - y[1])
        #return abs(ev_eigs[0] - y[0])
        return np.sqrt(0.80*(ev_eigs[0]-y[0])**2 + 0.20/35*((ev_eigs[1::]-y[1::]).T.dot(ev_eigs[1::]-y[1::])))

    def optimize_reference(self, num_targets):
        """Run the least-squares optimization, minimizing the residual
        function from self.
        
        Arguments:
        
        num_targets -- number of targets to pass into residual function
        
        Returns:

        normalized optimized weights"""
        
        target_weights = np.ones(num_targets)
        target_weights = target_weights/sum(target_weights)
        
        #target_weights = [0.8,0.2]

        x0 = np.asarray(np.insert(np.zeros(self._refs.shape[0]-len(target_weights)),0,target_weights))
        print(x0)
        
        # num_weights = self._refs.shape[0]
        # x0 = np.ones(num_weights)/sum(np.ones(num_weights))

        #x0 = np.asarray([0.8,0.2,0,0,0,0,0,0])

        assert num_targets >= 1,      'num_targets  >= 1'
        assert num_targets <= len(x0), 'num_targets < len(x0)'


        lsq = sciopt.least_squares(self.residual, 
                                   x0, 
                                   #args=(np.reshape(self._true, (1,36)), num_targets), 
                                   args=(self._true, num_targets), 
                                   bounds=(0,1), 
                                   method='trf', 
                                   tr_solver='lsmr', 
                                   diff_step=0.1, 
                                   loss='soft_l1',
                                   tr_options={'damp':1.0},
                                   ftol=1e-6,
                                   verbose=2) #diff_step=0.2, verbose=2)#, loss='linear')

        return (lsq.x/sum(lsq.x), lsq)

    
if __name__ == '__main__':
    ReferenceEnsemble()
