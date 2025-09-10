import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import quicklens as ql
import pickle
from . import tools as tls
import sys
# TODO: install BasicILC
sys.path.insert(0, '/home/ce425/rds/rds-dirac-dp002/ce425/ksz_biases/BasicILC/')
import cmb_ilc
import concurrent
from scipy.special import roots_legendre

import jax.numpy as jnp
from jax import device_put, jit
from functools import partial

try:
    from pyccl.pyutils import _fftlog_transform
    ccl_available = True
except ImportError:
    ccl_available = False

class Exp_minimal: # CEV: this might actually never be used?
    """ A helper class to encapsulate some essential attributes of experiment() objects to be passed to parallelized
        workers, saving as much memory as possible
        - Inputs:
            * exp = an instance of the experiment() class
    """
    def __init__(self, exp):
        dict = {"cltt_tot": exp.cltt_tot, "qe_norm_at_lbins_sec_bispec": exp.qe_norm_at_lbins_sec_bispec,
                "lmax": exp.lmax, "nx": exp.nx, "dx": exp.dx, "pix": exp.pix, "tsz_filter": exp.tsz_filter,
                "massCut": exp.massCut, "ls":exp.ls, "cl_len":exp.cl_len, "cl_unl":exp.cl_unl, "qest_lib":exp.qest_lib,
                "ivf_lib":exp.ivf_lib, "qe_norm":exp.qe_norm_compressed, "nx_secbispec":exp.nx_secbispec,
                "dx_secbispec":exp.dx_secbispec, "weights_mat_total":exp.weights_mat_total, "nodes":exp.nodes}
        self.__dict__ = dict

class experiment:
    def __init__(self, nlev_t=np.array([5.]), beam_size=np.array([1.]), lmax=3500, massCut_Mvir = np.inf, nx=1024,
                 dx_arcmin=1., nx_secbispec=128, dx_arcmin_secbispec=0.1, fname_scalar=None, fname_lensed=None,
                 freq_GHz=np.array([150.]), fg=True, atm_fg=False, MV_ILC_bool=False, deproject_tSZ=False,
                 deproject_CIB=False, bare_bones=False, nlee=None,
                 gauss_order=1000,
                 estimator="lensing",                                   # CEV add flag to swap lensing and kSZ
                 ls_gal_cls = None, cl_gg = None, cl_taug = None):  # CEV: spectra and corresponding ells for kSZ QE filters and norm.
        """ Initialise a cosmology and experimental charactierstics
            - Inputs:
                * nlev_t = np array. Temperature noise level, in uK.arcmin. Either single value or one for each freq
                * beam_size = np array. beam fwhm (symmetric). In arcmin. Either single value or one for each freq
                * lmax = reconstruction lmax.
                * (optional) massCut_Mvir = Maximum halo virial masss, in solar masses. Default is no cut (infinite)
                * (optional) fname_scalar = CAMB files for unlensed CMB
                * (optional) fname_lensed = CAMB files for lensed CMB
                * (otional) nx = int. Width in number of pixels of grid used in quicklens computations
                * (optional) dx = float. Pixel width in arcmin for quicklens computations
                * (otional) nx_secbispec = int. Same as nx, but for secondary bispectrum bias calculation
                * (optional) dx_arcmin_secbispec = float. Same as dx, but for secondary bispectrum bias calculation
                * (optional) freq_GHz =np array of one or many floats. Frequency of observqtion (in GHZ). If array,
                                        frequencies that get combined as ILC using ILC_weights as weights
                * (optional) fg = Whether or not to include non-atmospheric fg power in inverse-variance filter
                * (optional) atm_fg = Whether or not to include atmospheric fg power in inverse-variance filter. Default
                                    is False, as BasicILC does not implement atm fg properly correlated across
                                    frequencies. If True, include atm fgs, but note that this is only correct for the
                                     hardcoded case of SO.
                * (optional) MV_ILC_bool = Bool. If true, form a MV ILC of freqs
                * (optional) deproject_tSZ = Bool. If true, form ILC deprojecting tSZ and retaining unit response to CMB
                * (optional) deproject_CIB = Bool. If true, form ILC deprojecting CIB and retaining unit response to CMB
                * (optional) bare_bones= Bool. If True, don't run any of the costly operations at initialisation
                * (optional) nlee = np array of size lmax+1 containing E-mode noise (and fg) power for delensing template
                * (optional) gauss_order= int. Order of the Gaussian quadrature used to compute analytic QE
                # CEV modifs:
                * (optional) estimator = str. Estimator to compute biases for. Options: "lensing", "ksz_vel". Default is "lensing" so CosmoBLENDER can run as usual.
                * (optional) ls_gal_cls = 1D numpy array. Multipoles at which cl_gg and cl_taug are defined. Needed if estimator="ksz_vel"
                * (optional) cl_gg = 1D numpy array. Galaxy auto spectrum at ls_gal_cls including shot noise. Needed if estimator="ksz_vel"
                * (optional) cl_taug = 1D numpy array. Cross-spectrum of electron optical depth and galaxy overdensity at ls_gal_cls. Needed if estimator="ksz_vel"
        """
        if fname_scalar is None:
            fname_scalar = None#'~/Software/Quicklens-with-fixes/quicklens/data\/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
        if fname_lensed is None:
            fname_lensed = None#'~/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat'

        #Initialise CAMB spectra for filtering
        self.cl_unl = ql.spec.get_camb_scalcl(fname_scalar, lmax=lmax)
        self.cl_len = ql.spec.get_camb_lensedcl(fname_lensed, lmax=lmax)
        self.nlee = nlee
        self.ls = self.cl_len.ls # CEV: these ls that come from cl_len will be used across for cltt_tot but therefor also for ksz filters.
        self.lmax = lmax
        self.lmin = 1
        self.freq_GHz = freq_GHz

        #CEV: choice of estimator
        self.estimator = estimator

        # CEV: Requiere and interpolate cl_gg and cl_taug if using ksz_vel estimator to ls to match cltt_tot.
        if self.estimator == "ksz_vel":
            # Flags for ksz_vel case
            assert ls_gal_cls is not None, "Please provide ls_gal_cls when using ksz_vel estimator"
            assert cl_gg is not None, "Please provide cl_gg when using ksz_vel estimator"
            assert cl_taug is not None, "Please provide cl_taug when using ksz_vel estimator"

            # Ensure ls_gal_cls goes up to at least lmax
            if ls_gal_cls[-1] < self.lmax:
                raise ValueError(f"ls_gal_cls must go up to at least lmax of reconstruction ({self.lmax}), but got ls_gal_cls[-1]={ls_gal_cls[-1]}")

            # Directly interpolate cl_gg and cl_taug to self.ls
            self.cl_gg = np.interp(self.ls, ls_gal_cls, cl_gg, left=0., right=0.)       # CEV: I think there are enough flags for these left and right to never be used. If they are it will probably break the filters (~1/cl_gg).
            self.cl_taug = np.interp(self.ls, ls_gal_cls, cl_taug, left=0., right=0.)

        # Hyperparams for analytic QE calculation
        self.gauss_order = gauss_order
        self.nodes, self.weights = self.get_quad_nodes_weights(gauss_order, self.lmin, self.lmax) # Nodes and weights for Gaussian quadrature
        # CEV: we have a double integral that can be computed applying Gaus quad twice. That then translates into matrix multiplication (C.12) or (C.14).
        # Create 2D meshgrid of nodes for evaluating the ell-dependence of the QE integrand.
        self.lnodes_grid, self.lpnodes_grid = np.meshgrid(self.nodes, self.nodes) 

        self.massCut = massCut_Mvir #Convert from M_vir (which is what Alex uses) to M_200 (which is what the
                                    # Tinker mass function in hmvec uses) using the relation from White 01.

        # Experiment info
        self.nlev_t = nlev_t # noise level for temperature
        self.nlev_p = np.sqrt(2) * nlev_t # noise level for polarization?
        self.beam_size = beam_size

        # Set up grid for Quicklens calculations
        self.nx = nx
        self.dx = dx_arcmin/60./180.*np.pi # pixel width in radians.
        self.nx_secbispec = nx_secbispec
        self.dx_secbispec = dx_arcmin_secbispec/60./180.*np.pi # pixel width in radians.
        self.pix = ql.maps.cfft(self.nx, self.dx)
        self.ivf_lib = None
        self.qest_lib = None

        self.tsz_filter = None

        self.MV_ILC_bool = MV_ILC_bool
        self.deproject_tSZ = deproject_tSZ
        self.deproject_CIB = deproject_CIB

        # Initialise an empty dictionary to store the biases
        empty_arr = {}
        self.biases = tls.CustomBiasesDict({ 'ells': empty_arr,
                        'second_bispec_bias_ells': empty_arr,
                        'tsz' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'cross_w_gals' : {'1h' : empty_arr, '2h' : empty_arr}},
                        'cib' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'cross_w_gals' : {'1h' : empty_arr, '2h' : empty_arr}},
                        'mixed': {'trispec': {'1h': empty_arr, '2h': empty_arr},
                                'prim_bispec': {'1h': empty_arr, '2h': empty_arr},
                                'second_bispec': {'1h': empty_arr, '2h': empty_arr},
                                 'cross_w_gals' : {'1h' : empty_arr, '2h' : empty_arr}} })

        if not bare_bones:
            #Initialise sky model
            self.sky = cmb_ilc.CMBILC(freq_GHz*1e9, beam_size, nlev_t, fg=fg, atm=atm_fg, lMaxT=self.lmax)
            if len(self.freq_GHz)>1:
                # In cases where there are several, compute ILC weights for combining different channels
                assert MV_ILC_bool or deproject_tSZ or deproject_CIB, 'Please indicate how to combine different channels'
                assert not (MV_ILC_bool and (deproject_tSZ or deproject_CIB)), 'Only one ILC type at a time!'
                self.get_ilc_weights()
            # Compute total TT power (incl. noise, fgs, cmb) for use in inverse-variance filtering
            self.get_total_TT_power() # CEV: this gives you self.cltt_tot

            # Calculate inverse-variance filters
            self.inverse_variance_filters()
            # Calculate QE norm
            if self.estimator == "lensing":
                self.get_qe_norm()
            # CEV: in the kSZ case, this calculation depends on weights_mat_total, which is initialized in biases.py.
            # Therefore, one needs to initialize the normalization also in biases.py right after get_weights_mat_total is called.
            # CEV: TODO: potentially when code is working, one could make it so that weights_mat_total and qe_norm is only computed once. But not a priority.



    def get_quad_nodes_weights(self, gauss_order, a, b): # CEV: gets Gaussian quadrature nodes and weights 1D for a given order. 
        """
        Computes the nodes and weights for Gaussian quadrature given the order. And translates them to the desired integration domain [a, b].
        - Inputs:
            - gauss_order = int. Hyperparameter of the integration. What Gaussian order to use in the quadrature.
            - a = int. The lower bound of the integration domain. In practice lmin.
            - b = int. The upper bound of the integration domain. In practice lmax.
        - Returns:
            - nodes = 1D array, len = gauss_order. Quad nodes for the given gauss order.
            - weights = 1D array, len = gauss_order. Quad weights for the given gauss order.
        """
        # Get the nodes and weights for Gaussian quadrature of chosen order
        nodes_on_minus1to1, weights = roots_legendre(gauss_order) # scipy function gives nodes and weights from -1 to 1
        # We must now convert the nodes to our actual integration domain
        nodes = (b - a) / 2. * nodes_on_minus1to1 + (a + b) / 2.
        return nodes, (b - a) / 2. * weights

    def W_phi(self, lmax_clkk): # CEV: only used for delensing
        # TODO: might want to specify cosmo here and elsewhere for ql
        clpp = ql.spec.get_camb_scalcl(None, lmax=lmax_clkk).clpp
        nlpp = self.get_nlpp(lmin=30, lmax=lmax_clkk, bin_width=30)
        return clpp / (clpp + nlpp)

    def W_E(self, lmax_clee): # CEV: only used for delensing
        ells = np.arange(lmax_clee+1)
        if self.nlee is not None:
            self.clee_tot = self.sky.cmb[0, 0].flensedEE(ells) + self.nlee
        else:
            self.get_total_EE_power(lmax_clee)
        return np.nan_to_num(self.sky.cmb[0, 0].flensedEE(ells) / self.clee_tot)

    def inverse_variance_filters(self):
        """
        Calculate the inverse-variance filters to be applied to the fields prior to lensing reconstruction
        """
        lmin = 2 #TODO: define this as a method of the class
        # Initialize some dummy object that are required by quicklens
        # Initialise a dummy set of maps for the computation
        tmap = qmap = umap = np.random.randn(self.nx, self.nx)
        tqumap = ql.maps.tqumap(self.nx, self.dx, maps=[tmap, qmap, umap])
        transf = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : self.lmax, 'cltt' : np.ones(self.lmax+1),
                                                     'clee' : np.ones(self.lmax+1),
                                                     'clbb' : np.ones(self.lmax+1)} ), self.pix)
        cl_tot_theory  = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : self.lmax, 'cltt' : self.cltt_tot,
                                                          'clee' : np.zeros(self.lmax+1),
                                                              'clbb' :np.zeros(self.lmax+1)} ) )
        # TODO: find a neater way of doing this using ivf.library_diag()
        self.ivf_lib = ql.sims.ivf.library_l_mask(ql.sims.ivf.library_diag_emp(tqumap, cl_tot_theory, transf=transf,
                                                                               nlev_t=0, nlev_p=0),
                                                  lmin=lmin, lmax=self.lmax)

    def get_ilc_weights(self):
        """
        Get the harmonic ILC weights. CEV: used to construct tsz filter (qest.py) and CIB filter (biases.py)
        """
        lmin_cutoff = 14
        num_of_ells = 50 # Sum of weights is still 1 to 1 part in 10^14 even with ells spaced 100 apart
        # Evaluate only at discrete ells, and interpolate later.
        W_sILC_Ls = np.linspace(lmin_cutoff, self.lmax, num_of_ells)

        if self.MV_ILC_bool:
            W_sILC = np.array(
                list(map(self.sky.weightsIlcCmb, W_sILC_Ls)))
        elif self.deproject_tSZ and self.deproject_CIB:
            W_sILC = np.array(
                list(map(self.sky.weightsDeprojTszCIB, W_sILC_Ls)))
        elif self.deproject_tSZ:
            W_sILC = np.array(
                list(map(self.sky.weightsDeprojTsz, W_sILC_Ls)))
        elif self.deproject_CIB:
            W_sILC = np.array(
                list(map(self.sky.weightsDeprojCIB, W_sILC_Ls)))

        self.ILC_weights, self.ILC_weights_ells = tls.spline_interpolate_weights(W_sILC, W_sILC_Ls, self.lmax)
        return

    def get_tsz_filter(self):
        """
        Calculate the ell-dependent filter to be applied to the y-profile harmonics. In the single-frequency scenario,
        this just applies the tSZ frequency-dependence. If doing ILC cleaning, it includes both the frequency
        dependence and the effect of the frequency-and-ell-dependent weights
        """
        if len(self.freq_GHz)>1:
            # Multiply ILC weights at each freq by tSZ scaling at that freq, then sum them together at every multipole
            tsz_filter = np.sum(tls.scale_sz(self.freq_GHz) * self.ILC_weights, axis=1)
            # Return the filter interpolated at every ell where we will perform lensing recs, i.e. [0, self.lmax]
            #TODO: I don't think this interpolation step is needed anymore
            return np.interp(np.arange(self.lmax+1), self.ILC_weights_ells, tsz_filter, left=0, right=0)
        else:
            # Single-frequency scenario. Return a single number.
            return tls.scale_sz(self.freq_GHz)

    def get_total_TT_power(self):
        """
        Get total TT power from CMB, noise and fgs.
        Note that if both self.deproject_tSZ=1 and self.deproject_CIB=1, both are deprojected
        """
        #TODO: Why can't we get ells below 10 in cltt_tot?
        if len(self.freq_GHz)==1:
            self.cltt_tot = self.sky.cmb[0, 0].ftotalTT(self.cl_unl.ls)
        else:
            nL = 201
            L = np.logspace(np.log10(self.lmin), np.log10(self.lmax), nL)
            # ToDo: sample better in L
            if self.MV_ILC_bool:
                f = lambda l: self.sky.powerIlc(self.sky.weightsIlcCmb(l), l)
            elif self.deproject_tSZ and self.deproject_CIB:
                f = lambda l: self.sky.powerIlc(self.sky.weightsDeprojTszCIB(l), l)
            elif self.deproject_tSZ:
                f = lambda l: self.sky.powerIlc(self.sky.weightsDeprojTsz(l), l)
            elif self.deproject_CIB:
                f = lambda l: self.sky.powerIlc(self.sky.weightsDeprojCIB(l), l)
            #TODO: turn zeros into infinities to avoid issues when dividing by this
            self.cltt_tot = np.interp(self.cl_unl.ls, L, np.array(list(map(f, L))))
        # Avoid infinities when dividing by inverse variance
        self.cltt_tot[np.where(np.isnan(self.cltt_tot))] = np.inf

    def get_total_EE_power(self, lmax): # CEV: only used for delensing
        """
        Get total EE power from CMB, noise and fgs.
        At present, this assumes the E-modes are obtained from exactly the same channels as the temperature
        Note that if both self.deproject_tSZ=1 and self.deproject_CIB=1, both are deprojected

        # TODO: Allow E-modes to come from a different set of observations
        # TODO: Allow ClEE to change with background cosmology
        # TODO: Allow for the possibility of deprojecting various components from the E-modes
        """
        ells = np.arange(lmax+1)
        if len(self.freq_GHz)==1:
            self.clee_tot = self.sky.cmb[0, 0].ftotalEE(ells)
        else:
            nL = 201
            L = np.logspace(np.log10(self.lmin), np.log10(lmax), nL)
            # ToDo: sample better in L
            f = lambda l: self.sky.powerIlcEE(self.sky.weightsIlcCmbEE(l), l)

            #TODO: turn zeros into infinities to avoid issues when dividing by this
            self.clee_tot = np.interp(ells, L, np.array(list(map(f, L))))
        # Avoid infinities when dividing by inverse variance
        #self.clee_tot[np.where(np.isnan(self.clee_tot))] = np.inf
        self.clee_tot = np.nan_to_num(self.clee_tot)

    def __getstate__(self):
        # this method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()

        # don't pickle the parameter fun. otherwise will raise
        # AttributeError: Can't pickle local object 'Process.__init__.<locals>.<lambda>'
        del state['sky']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_weights_mat_total(self, ells_out): # CEV: takes W(L) and just computes the full weights matrix. keeps in self. Nothing to edit
        # CEV: This function is never called in qest.py. It is called everytime you call to compute a bias. Given that this only depends on ell stuff I wonder why not call once here when you initialize qest and then use whenever. Because of being able to call different ells_out every time presumably. Not really because it's defined at initialization of biases anyway.
        ''' Get the matrices needed for Gaussian quadrature of QE integral '''
        self.weights_mat_total = device_put(jnp.array([self.weights_mat_at_L(L) for L in ells_out]))
        self.get_qe_ksz_norm(self.nodes, self.cl_gg, self.cl_taug, self.cltt_tot, self.ls) if self.estimator == "ksz_vel" else None

    def weights_mat_at_L(self, L): 
        # CEV: This computes the weights for equation C.12 as a function of L. In particular: w_i * w_j * W(L,l_i,l_j) given you already have w and W
        # CEV: w are just weights from the Gaussian quadrature that will be the same for kSZ
        '''
        Calculate the matrix to be used in the QE integration, i.e.,
        H(L). This is derived from
        W(L, l, l') \equiv -\Delta(l,l',L) l l' \left(\frac{L^2 + l'^2 - l^2}{2Ll'} \right)\left[1 - \left( \frac{L^2 +l'^2 -l^2}{2Ll'}\right)\right]^{-\frac{1}{2}}

        by sampling at the quadrature nodes in l and l' and  multiplying rows and columns by the quadrature weights w_i.

        We cache the outputs of this function as they are used recurrently at every mass and redshift step

        Inputs:
            - L = int. The L at which we are evaluating the QE reconstruction
        Returns:
            - W(L, l_i, l_j)
        '''
        return (self.weights * self.weights[:, np.newaxis] * self.ell_dependence(L, self.lnodes_grid, self.lpnodes_grid)).astype(np.float32)

    def ell_dependence(self, L, l, lp): # CEV: Here is where you actually code the weights
        '''
        Sample the kernel of the chosen reconstruction (self.estimator).
        For lensing:
        W(L, l, l') \equiv -\Delta(l,l',L) l l' \left(\frac{L^2 + l'^2 - l^2}{2Ll'} \right)\left[1 - \left( \frac{L^2 +l'^2 -l^2}{2Ll'}\right)\right]^{-\frac{1}{2}}

        Inputs:
            - L = int. The L at which we are evaluating the QE reconstruction
        Returns:
            - W(L, l, lp)
        '''

        # CEV: triangle condition is the same in both cases, keep as is.
        L = np.asarray(L, dtype=int)  # Ensure L is an integer
        condition = (L + l >= lp) & (L + lp >= l) & (l + lp >= L) # CEV: First two encapsulate the two options of the |l-lp|<=L , third is same.
        singular_condition = (L + l == lp) | (L + lp == l) | (l + lp == L) #CEV: exclude cases where Delta=0 to avoid infinities.

        result = np.zeros_like(l, dtype=float)  # Initialize result array with appropriate dtype

        valid_indices = np.where(condition & ~singular_condition)
        #TODO: I've removed minus sign to get expected -ve sign at low L in prim bispec. What's up?

        if self.estimator == "lensing":
            result[valid_indices] = 2 * lp[valid_indices] * l[valid_indices] * (
                    (L ** 2 + lp[valid_indices] ** 2 - l[valid_indices] ** 2) / (2 * L * lp[valid_indices])) * (1 - (
                    (L ** 2 + lp[valid_indices] ** 2 - l[valid_indices] ** 2) / (2 * L * lp[valid_indices])) ** 2) ** (
                                    -0.5)
            
        elif self.estimator == "ksz_vel":
            triangle = np.zeros_like(l, dtype=float)
            triangle[valid_indices] = 1./2 * ((L+l[valid_indices]+lp[valid_indices]) * 
                                            (-L+l[valid_indices]+lp[valid_indices]) * 
                                            (L-l[valid_indices]+lp[valid_indices]) * 
                                            (L+l[valid_indices]-lp[valid_indices]))**(0.5)
            result[valid_indices] = l[valid_indices] * lp[valid_indices] / triangle[valid_indices]

        else:
            raise ValueError(f"Unknown estimator: {self.estimator}. Only 'lensing' and 'ksz_vel' are supported.")

        return result

    def get_qe_norm(self, key='ptt'):
        """
        Calculate the QE normalisation as the reciprocal of the N^{(0)} bias
        Inputs:
            * (optional) key = String. The quadratic estimator key. Default is 'ptt' for TT
        # TODO: replace this with a faster analytic calculation that does away with Quicklens dependence
        """
        self.qest_lib = ql.sims.qest.library(self.cl_unl, self.cl_len, self.ivf_lib)
        self.qe_norm = self.qest_lib.get_qr(key)

    def get_qe_ksz_norm(self, nodes, cl_gg, cl_taug, cltt_tot, ls):
        """
        Calculate the kSZ velocity quadratic estimator normalisation
        Inputs:
            * weights_mat_total = np array with dimensions (len(ells_out), len(nodes), len(nodes)).
            * nodes = 1D np array. The Gaussian-quadrature-determined ells at which to evaluate integrals. Needed
                                            if use_gauss=True
            * cl_gg = 1D numpy array. Galaxy auto spectrum at ls including shot noise.
            * cl_taug = 1D numpy array. Cross-spectrum of electron optical depth and galaxy overdensity at ls.
            * cltt_tot = 1d numpy array. Total power in observed TT fields at ls.
            * ls = 1d numpy array. Multipoles at which cltt_tot is defined.
        Returns:
            * qe_norm = 1D numpy array. Normalisation at ells_out multipoles.
        """
        # Get the filters F_1 and F_2 from new function
        al_F_1, al_F_2 = get_filters_kSZ_norm(cltt_tot=cltt_tot, ls=ls, cl_gg=cl_gg, cl_taug=cl_taug)
        F_1_array = jnp.array(al_F_1(nodes).astype(np.float32)) # CEV: Evaluate Fs at nodes and convert to jax arrays.
        F_2_array = jnp.array(al_F_2(nodes).astype(np.float32))
        # print("F_1_array ",F_1_array) THESE WORK
        # print("F_2_array ",F_2_array)
        norm = self.QE_via_quad(F_1_array, F_2_array)
        self.qe_ksz_norm = norm

    def get_nlpp(self, lmin=30, lmax=3000, bin_width=30):
        # TODO: adapt  the lmax of these bins to the lmax_out of hm_object
        # TODO: this N0 is not smooth. Find a better way to calculate
        ells = np.arange(lmax+1)
        lbins = np.arange(lmin, lmax, bin_width)
        norm = self.qe_norm.get_ml(lbins)
        # Extrapolate in clkk, for which we expect flatness at low L
        nlkk = np.interp(ells, norm.ls, np.nan_to_num(norm.ls**4/norm.specs['cl']))
        return np.nan_to_num(nlkk/ells**4)

    def __getattr__(self, spec):
        try:
            return self.biases[spec]
        except KeyError:
            raise AttributeError(spec)

    def __str__(self):
        """ Print out halo model calculator properties """
        massCut = '{:.2e}'.format(self.massCut)
        return 'Mass Cut: ' + str(massCut) + '  lmax: ' + str(self.lmax) + '  Beam FWHM: '+ str(self.beam_size) + \
               ' Noise (uK arcmin): ' + str(self.nlev_t) + '  Freq (GHz): ' + str(self.freq_GHz)

    def save_biases(self, output_filename='./dict_with_biases'):
        """
        Save the dictionary of biases to file
        Inputs:
            * output_filename = str. Output filename
        """
        with open(output_filename+'.pkl', 'wb') as output:
            pickle.dump(self.biases, output, pickle.HIGHEST_PROTOCOL)

    def get_TT_qe(self, fftlog_way, ell_out, profile_leg1, qe_norm, pix, lmax, cltt_tot=None, ls=None, cltt_len=None,
                  qest_lib=None, ivf_lib=None, profile_leg2=None, N_l=2 * 4096, lmin=0.000135, alpha=-1.3499,
                  norm_bin_width=40, key='ptt', use_gauss=True, weights_mat_total=None, nodes=None):
        """
        Helper function to get the TT QE reconstruction for spherically-symmetric profiles using FFTlog
        Inputs:
            * fftlog_way = Bool. If true, use fftlog of Gaussian quad reconstruction. Otherwise use quicklens.
            * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
            * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * qe_norm = if fftlog_way=True, an experiment.qe_norm() instance.
                        otherwise, a 1D array containg the normalization of the TT QE at ell_out
            * pix = ql.maps.cfft() object. Contains numerical hyperparameters nx and dx
            * lmax = int. Maximum multipole used in the reconstruction
            * (optional) cltt_tot = 1d numpy array. Total power in observed TT fields. Needed if fftlog_way=1
            * (optional) ls = 1d numpy array. Multipoles at which cltt_tot is defined. Needed if fftlog_way=1
            * (optional) cltt_len = 1d numpy array. Lensed TT power spectrum at ls. Needed if fftlog_way=1
            * (optional) qest_lib = experiment.qest_lib() instance for quicklens lensing rec. Needed if fftlog_way=0
            * (optional) ivf_lib = experiment.ivf_lib() instance for quicklens lensing rec. Needed if fftlog_way=0
            * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
            * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
                               Needed if fftlog_way=1
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values
                                (e.g., lmin=1e-4) to avoid ringing. Needed if fftlog_way=1
            * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
                                 Needed if fftlog_way=1
            * (optional) norm_bin_width = int. Bin width to use when taking spectra of the semi-analytic QE
                                          normalisation. Needed if fftlog_way=1
        Returns:
            * If fftlog_way=True, a 1D array with the unnormalised reconstruction at the multipoles specified in ell_out
            * (optional) key = String. The quadratic estimator key for quicklens. Default is 'ptt' for TT
            * (optional) use_gauss = Bool. If True, use Gaussian quad. Otherwise pyccl FFTlog
            * (optional) weights_mat_total = np array with dimensions (len(ells_out), len(nodes), len(nodes)). Required if
                                            use_gauss=True
            * (optional) nodes = 1D np array. The Gaussian-quadrature-determined ells at which to evaluate integrals. Needed
                                            if use_gauss=True
        """
        if profile_leg2 is None:
            profile_leg2 = profile_leg1
        if fftlog_way:
            assert (cltt_tot is not None and ls is not None and cltt_len is not None)
            al_F_1, al_F_2 = get_filtered_profiles_fftlog(profile_leg1, cltt_tot, ls, cltt_len, profile_leg2)
            # Calculate unnormalised QE
            if use_gauss == True:
                assert (weights_mat_total is not None and nodes is not None)
                F_1_array = jnp.array(al_F_1(nodes).astype(np.float32)) # CEV: Evaluate Fs at nodes and convert to jax arrays.
                F_2_array = jnp.array(al_F_2(nodes).astype(np.float32))
                unnorm_TT_qe = self.QE_via_quad(F_1_array, F_2_array) # CEV: already given at ells_out through weights_mat_total.
            else:
                assert (ccl_available), 'pyccl not available. Please install pyccl to use FFTlog'
                unnorm_TT_qe = unnorm_TT_qe_fftlog(al_F_1, al_F_2, N_l, lmin, alpha, lmax)(ell_out)
            # Apply a convention correction to match Quicklens
            #TODO: do we need a factor of 2pi here?
            conv_corr = 1  / (2 * np.pi) /2
            return conv_corr * np.nan_to_num(unnorm_TT_qe / qe_norm)
        else:
            assert (ivf_lib is not None and qest_lib is not None)
            tft1 = ql.spec.cl2cfft(profile_leg1, pix)
            # Apply filters and do lensing reconstruction
            t_filter = ivf_lib.get_fl().get_cffts()[0]
            tft1.fft *= t_filter.fft
            if profile_leg2 is None:
                tft2 = tft1.copy()
            else:
                tft2 = ql.spec.cl2cfft(profile_leg2, pix)
                tft2.fft *= t_filter.fft
            unnormalized_phi = qest_lib.get_qft(key, tft1, 0 * tft1.copy(), 0 * tft1.copy(),
                                                tft2, 0 * tft1.copy(), 0 * tft1.copy())
            # In QL, the unnormalised reconstruction (obtained via eval_flatsky()) comes with a factor of sqrt(skyarea)
            A_sky = (pix.dx * pix.nx) ** 2
            # Normalize the reconstruction
            return np.nan_to_num(unnormalized_phi.fft[:, :] / qe_norm.fft[:, :]) / np.sqrt(A_sky)
        
    def get_kSZ_qe(self, profile_leg_T, profile_leg_g, cltt_tot, ls, cl_gg, cl_taug,
                   weights_mat_total, nodes=None):
        """
        Helper function to get the kSZ QE reconstruction for spherically-symmetric profiles using Gaussian quadratures. No fftlog or other options available in this case.
        Inputs:
            * profile_leg_T = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * profile_leg_g = 1D numpy array. Galaxy density profile for the other QE leg.
            * qe_norm = a 1D array containg the normalization of the kSZ QE at ell_out
            * cltt_tot = 1d numpy array. Total power in observed TT fields. Needed.
            * ls = 1d numpy array. Multipoles at which cltt_tot is defined. Needed.
            * cl_gg = 1d numpy array. Galaxy autospectrum spectrum at ls. Needed.
            * cl_taug = 1d numpy array. Galaxy-electron power spectrum at ls. Needed.
            * weights_mat_total = np array with dimensions (len(ells_out), len(nodes), len(nodes)).
            * nodes = 1D np array. The Gaussian-quadrature-determined ells at which to evaluate integrals.
        Returns:
            * 1D array with the normalised reconstruction at the multipoles specified in ell_out.
        """

        if profile_leg_g is None:
            raise ValueError(f"Both profiles need to be specified when estimator='ksz_vel'.")

        assert (cltt_tot is not None and ls is not None and cl_gg is not None and cl_taug is not None)
        al_F_T, al_F_g = get_filtered_profiles_kSZ(profile_leg_T, cltt_tot, ls, cl_gg, cl_taug, profile_leg_g)
    
        # Calculate unnormalised QE
        assert (weights_mat_total is not None and nodes is not None)
        F_T_array = jnp.array(al_F_T(nodes).astype(np.float32)) # CEV: Evaluate Fs at nodes and convert to jax arrays.
        F_g_array = jnp.array(al_F_g(nodes).astype(np.float32))
        unnorm_ksz_qe = self.QE_via_quad(F_T_array, F_g_array) # CEV: already gives result at ells_out through weights_mat_total

        # CEV: in principle, no need for convention correction if normalization has been computed consistently.
        # CEV: not calling this here anymore, as get_weights_mat_total calls it when initializing biases.
        # self.get_qe_ksz_norm(nodes, cl_gg, cl_taug, cltt_tot, ls)
        qe_ksz_norm_jx = jnp.array(self.qe_ksz_norm.astype(np.float32))
        return np.nan_to_num(unnorm_ksz_qe / qe_ksz_norm_jx)

    def QE_via_quad(self, F_1_array, F_2_array):
        '''
        Unnormalized TT quadratic estimator

        \hat{\phi}(\vL) = 2 \int \frac{dl\,dl'}{2\pi} F_1(l) F_2(l') W(L, l, l')

        where

         W(L, l, l') \equiv - \Delta(l,l',L) l l' \left(\frac{L^2 + l'^2 - l^2}{2Ll'} \right)\left[1 - \left( \frac{L^2 +l'^2 -l^2}{2Ll'}\right)\right]^{-\frac{1}{2}}

        and \Delta(l,l',L) is the triangle condition. The double integral is calculated using Gaussian quadratures
        implemented in the form of matrix multiplication to harness the speed of numpy's BLAS library:

        F_1(l_i) H(L) F_2(l_i)

        where l_i are the Gaussian quadrature nodes (and similarly for F_2), and H(L) is a matrix derived from W(L, l_i, l_j)
        after it has absorbed the quadrature weights [as documented in weights_mat_total()]. It appears that only a few dozen nodes
        are needed for reasonable accuracy, but if more were required, one could explore using GPUs.

        Furthermore, we evaluate F_1(l_i) H(L) F_2(l_i) at the required L's via matrix-multiplication

        Inputs:
            - F_1_array = 1D np array. The filtered inputs of the QE evaluated at the Gaussian quadrature nodes
            - F_2_array = 1D np array. Same as F_1_array, but for F_2
            - weights_mat = 3D np array (len(L), len(ell), len(ellprime)) featuring the L, ell and ellprime dependence
        Returns:
            - The unnormalized lensing reconstruction at L
        '''
        # CEV: TODO: ojo with this 2pi. Not sure it's consistent with my kSZ normalization. Check.
        return jnp.dot(self.inner_mult(F_2_array), F_1_array) / (2 * np.pi)

    @partial(jit, static_argnums=(0,))
    def inner_mult(self, arr1):
        return jnp.matmul(arr1, self.weights_mat_total)

def load_dict_of_biases(filename='./dict_with_biases.pkl', verbose=False):
    """
    Load a dictionary of biases that was previously saved using experiment.save_biases()
    Inputs:
        * filename = str. Filename for the pickle object to be loaded
    Returns:
        * Dict of biases with indexing as in experiment.biases
    """
    with open(filename, 'rb') as input:
        experiment_object = pickle.load(input)
    if verbose:
        print('Successfully loaded experiment object with properties:\n')
        print(experiment_object)
    return experiment_object

def get_brute_force_unnorm_TT_qe(ell_out, profile_leg1, cltt_tot, ls, cltt_len, lmax,
                                 profile_leg2=None, max_workers=None):
    """
    Slow but sure method to calculate the 1D TT QE reconstruction.
    Scales as O(N^3), but useful as a cross-check of get_unnorm_TT_qe(fftlog_way=True)
    Inputs:
        * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
        * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
        * cltt_tot = 1d numpy array. Total power in observed TT fields.
        * ls = 1d numpy array. Multipoles at which cltt_tot is defined
        * cltt_len = 1d numpy array. Lensed TT power spectrum at ls.
        * lmax = int. Maximum multipole used in the reconstruction
        * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
        * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
    """
    al_F_1, al_F_2 = get_filtered_profiles_fftlog(profile_leg1, cltt_tot, ls, cltt_len, profile_leg2=profile_leg2)
    output_unnormalised_phi = np.zeros(ell_out.shape)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        n = len(ell_out)
        outputs = executor.map(int_func, n * [lmax], n * [ell_out], np.arange(n), n * [al_F_1], n * [al_F_2])

    for idx, outs in enumerate(outputs):
        output_unnormalised_phi[idx] = outs

    return output_unnormalised_phi

def int_func(lmax, ell_out, n, al_F_1, al_F_2):
    """
    Helper function to parallelize get_brute_force_unnorm_TT_qe()
    """
    def ell_dependence(L, l, lp):
        '''L is outter multipole'''
        if (L+l>=lp) and (L+lp>=l) and (l+lp>=L):
            #check triangle inequality
            if (L+l==lp) or (L+lp==l) or (l+lp==L):
                # integrand is singular at the triangle equality
                print('dealing with integrable singularity by setting to 0')
                return 0
            return 2 * ( (L**2 + lp**2 - l**2) / (2*L*lp) )* ( 1 - ((L**2 + lp**2 - l**2) / (2*L*lp) )**2  )**(-0.5)
        else:
            return 0
    def inner_integrand(lp, L, l):
        return lp * al_F_2(lp) * ell_dependence(L, l, lp)
    def outer_integrand(l, L):
        return l * al_F_1(l) * quad(inner_integrand, 1, lmax, args=(L, l))[0]
    L = ell_out[n]
    return quad(outer_integrand, 1, lmax, args=L)[0]/(2*np.pi)

def unnorm_TT_qe_fftlog(al_F_1, al_F_2, N_l, lmin, alpha, lmax):
    """
    Compute the unnormalised TT QE reconstruction for spherically symmetric profiles using FFTlog.
    Inputs:
        * al_F_1 = Interpolatable object from which to get F_1 (e.g., in eq. (7.9) of Lewis & Challinor 06)
                   at every multipole.
        * al_F_2 = Interpolatable object from which to get F_2 (e.g., in eq. (7.9) of Lewis & Challinor 06)
                   at every multipole.
        * N_l = Int (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
        * lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values
                            (e.g., lmin=1e-4) to avoid ringing
        * alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
        * lmax = int. Maximum multipole used in the reconstruction
    Returns:
        * An interp1d object into which you can plug in an array of ells to get the QE at those ells.
    """
    ell = np.logspace(np.log10(lmin), np.log10(lmax), N_l)

    # The underscore notation _xyz refers to x=hankel order, y=F_y, z=powers of ell
    r_arr_0, f_010 = _fftlog_transform(ell, al_F_1(ell), 2, 0, alpha)
    r_arr_1, f_121 = _fftlog_transform(ell, ell * al_F_2(ell), 2, 1, alpha)
    r_arr_2, f_111 = _fftlog_transform(ell, ell * al_F_1(ell), 2, 1, alpha)
    r_arr_3, f_022 = _fftlog_transform(ell, ell**2 * al_F_2(ell), 2, 0, alpha)
    r_arr_4, f_222 = _fftlog_transform(ell, ell**2 * al_F_2(ell), 2, 2, alpha)

    ell_out_arr, fl_total = _fftlog_transform(r_arr_4, f_121 * (-f_010/r_arr_0 + f_111)
                                              + 0.5 * f_010*(-f_022 + f_222) , 2, 0, alpha)
    # Interpolate and correct factors of 2pi from fftlog conventions
    unnormalised_phi = interp1d(ell_out_arr, - (2*np.pi)**3 * fl_total, bounds_error=False, fill_value=0.0)
    return unnormalised_phi

def get_filtered_profiles_fftlog(profile_leg1, cltt_tot, ls, cltt_len, profile_leg2=None):
    """
    Filter the profiles in the way of, e.g., eq. (7.9) of Lewis & Challinor 06.
    Inputs:
        * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
        * cltt_tot = 1d numpy array. Total power in observed TT fields.
        * ls = 1d numpy array. Multipoles at which cltt_tot is defined
        * cltt_len = 1d numpy array. Lensed TT power spectrum at ls.
        * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
    Returns:
        * Interpolatable objects from which to get F_1 and F_2 at every multipole.
    """

    def smooth_low_monopoles(array):
        new = array[2:]
        return np.interp(np.arange(len(array)), np.arange(len(array))[2:], new)

    if profile_leg2 is None:
        profile_leg2 = profile_leg1
    F_1_of_l = smooth_low_monopoles(np.nan_to_num(profile_leg1 / cltt_tot)) # CEV: Here is where filters are constructed
    F_2_of_l = smooth_low_monopoles(np.nan_to_num(cltt_len * profile_leg2/ cltt_tot)) # CEV: find out how profile_legs are defined
    al_F_1 = interp1d(ls, F_1_of_l, bounds_error=False,  fill_value='extrapolate')
    al_F_2 = interp1d(ls, F_2_of_l, bounds_error=False,  fill_value='extrapolate')
    return al_F_1, al_F_2

def get_filtered_profiles_kSZ(profile_leg_T, cltt_tot, ls, cl_gg, cl_taug, profile_leg_g):
    """
    Filter the profiles in the way of, e.g., eq. 13 of Kvasiuk & Munchmeyer (24). Or CEV.
    Inputs:
        * profile_leg_T = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax. T(ell) # CEV: this can be the same as antons
        * profile_leg_g = 1D numpy array. Projected galaxy field g^alpha(ell). # CEV: this will be coming from the HOD that I will be defined by user 
        * cltt_tot = 1d numpy array. Total power in observed TT fields.
        * ls = 1d numpy array. Multipoles at which cltt_tot is defined
        * cl_gg = 1d numpy array. Galaxy auto spectrum at ls including shot noise.
        * cl_taug = 1d numpy array. Galaxy-electron power spectrum at ls.
    Returns:
        * Interpolatable objects from which to get F_T and F_g at every multipole.
        CEV: al_F_T and al_F_g are now functions of l.
    """
    
    def smooth_low_monopoles(array): 
        ''' CEV: deals with 2 first multipoles by ignoring whatever info is in array and linearly extrapolating from l=3 backwards.'''
        new = array[2:]
        return np.interp(np.arange(len(array)), np.arange(len(array))[2:], new)
    
    # CEV: if ls[-1]<self.lmax-1 not necessary, it is so by construction.

    F_T_of_l = smooth_low_monopoles(np.nan_to_num(profile_leg_T / cltt_tot)) 
    F_g_of_l = smooth_low_monopoles(np.nan_to_num(cl_taug * profile_leg_g/ cl_gg)) # CEV: find out how to define delta field
    al_F_T = interp1d(ls, F_T_of_l, bounds_error=False,  fill_value='extrapolate') # CEV: function to get the value of F at any l
    al_F_g = interp1d(ls, F_g_of_l, bounds_error=False,  fill_value='extrapolate')
    return al_F_T, al_F_g

def get_filters_kSZ_norm(cltt_tot, ls, cl_gg, cl_taug):
    """
    Same function as above but without profiles, for the kSZ normalization.
    CEV: TODO: see if this can be merged with the above function.
    """
    
    def smooth_low_monopoles(array): 
        ''' CEV: deals with 2 first multipoles by ignoring whatever info is in array and linearly extrapolating from l=3 backwards.'''
        new = array[2:]
        return np.interp(np.arange(len(array)), np.arange(len(array))[2:], new)
    
    # if ls[-1]<self.lmax-1:
    #     raise ValueError(f"ls and Cl spectra should be provided up to lmax of reconstruction to avoid extrapolation. ls[-1]={ls[-1]}, self.lmax={self.lmax}")

    F_1_of_l = smooth_low_monopoles(np.nan_to_num(1.0 / cltt_tot)) 
    F_2_of_l = smooth_low_monopoles(np.nan_to_num(cl_taug * cl_taug / cl_gg)) 
    al_F_1 = interp1d(ls, F_1_of_l, bounds_error=False,  fill_value='extrapolate') # CEV: function to get the value of F at any l
    al_F_2 = interp1d(ls, F_2_of_l, bounds_error=False,  fill_value='extrapolate')
    return al_F_1, al_F_2