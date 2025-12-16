"""
Terminology:
itgnd = integrand
oneH = one halo
twoH = two halo
"""

import numpy as np
import hmvec as hm
from . import tools as tls
from . import qest
from . import second_bispec_bias_stuff as sbbs
import quicklens as ql
import concurrent

class Hm_minimal:
    """ A helper class to encapsulate some essential attributes of hm_framework() objects to be passed to parallelized
        workers, saving as much memory as possible
        - Inputs:
            * hm_full = an instance of the hm_framework() class
    """
    def __init__(self, hm_full):
        dict = {"m_consistency": hm_full.m_consistency, "ms": hm_full.hcos.ms, "nzm": hm_full.hcos.nzm,
                "ms_rescaled": hm_full.ms_rescaled, "zs": hm_full.hcos.zs, "ks": hm_full.hcos.ks,
                "comoving_radial_distance": hm_full.hcos.comoving_radial_distance(hm_full.hcos.zs),
                "uk_profiles": hm_full.hcos.uk_profiles, "pk_profiles": hm_full.hcos.pk_profiles,
                "p": hm_full.hcos.p, "bh": hm_full.hcos.bh, "lmax_out": hm_full.lmax_out,
                "y_consistency": hm_full.y_consistency, "g_consistency": hm_full.g_consistency,
                "I_consistency": hm_full.I_consistency, "Pzk": hm_full.hcos.Pzk, "nMasses": hm_full.nMasses,
                "nZs": hm_full.nZs, "hods":hm_full.hcos.hods, "CIB_satellite_filter":hm_full.CIB_satellite_filter,
                "CIB_central_filter":hm_full.CIB_central_filter}
        self.__dict__ = dict

class hm_framework:
    """ Set the halo model parameters """
    def __init__(self, lmax_out=500, m_min=1e10, m_max=5e15, nMasses=30, z_min=0.07, z_max=3, nZs=30, k_min = 1e-4,
                 k_max=10, nks=1001, mass_function='sheth-torman', mdef='vir', cib_model='planck13', cosmoParams=None
                 , xmax=5, nxs=40000, tsz_param_override={}):
        """ Inputs:
                * lmax_out = int. Maximum multipole (L) at which to return the lensing reconstruction
                * m_min = Minimum virial mass for the halo model calculation
                * m_max = Maximum virial mass for the halo model calculation (note that massCut_Mvir will overide this)
                * nMasses = Integer. Number of steps in mass for the integrals
                * z_min = Minimum redshift for the halo model calc
                * z_max = Maximum redshift for the halo model calc
                * nZs = Integer. Number of steps in redshift for the integrals
                * k_min = Minimum k for the halo model calc
                * k_max = Maximum k for the halo model calc
                * nks = Integer. Number of steps in k for the integrals
                * mass_function = String. Halo mass function to use. Must be coded into hmvec
                * mdef = String. Mass definition. Must be defined in hmvec for the chosen mass_function
                * cib_model = CIB halo model and fit params. Either 'planck13' or 'viero' (after Viero et al 13.)
                * cosmoParams = Dictionary of cosmological parameters to initialised HaloModel hmvec object
                * xmax = Float. Electron pressure profile integral xmax (see further docs at hmvec.add_nfw_profile() )
                * nxs = Integer. Electron pressure profile integral number of x's
                * tsz_param_override = Dictionary. Override the default parameters for the tSZ profile
        """
        self.lmax_out = lmax_out
        self.nMasses = nMasses
        self.m_min = m_min
        self.m_max = m_max
        self.z_min = z_min
        self.z_max = z_max
        self.nZs = nZs
        self.mass_function = mass_function
        self.mdef = mdef
        zs = np.linspace(z_min,z_max,nZs) # redshifts
        ms = np.geomspace(m_min,m_max,nMasses) # masses
        ks = np.geomspace(k_min,k_max,nks) # wavenumbers
        self.T_CMB = 2.7255e6 # 1. #
        self.nZs = nZs
        self.nxs = nxs
        self.xmax = xmax
        self.cosmoParams = cosmoParams
        self.tsz_param_override = tsz_param_override

        self.hcos = hm.HaloModel(zs,ks,ms=ms,mass_function=mass_function,params=cosmoParams,mdef=mdef)
        self.hcos.add_battaglia_pres_profile("y",family="pres",xmax=xmax,nxs=nxs, param_override=self.tsz_param_override)
        self.hcos.set_cibParams(cib_model)

        self.ms_rescaled = self.hcos.ms[...]/self.hcos.rho_matter_z(0)

        self.m_consistency = np.zeros(len(self.hcos.zs))
        self.y_consistency = np.zeros(len(self.hcos.zs))
        self.g_consistency = np.zeros(len(self.hcos.zs))
        self.I_consistency = np.zeros(len(self.hcos.zs))

        self.CIB_central_filter = None
        self.CIB_satellite_filter = None

    def __str__(self):
        """ Print out halo model calculator properties """
        m_min = '{:.2e}'.format(self.m_min)
        m_max = '{:.2e}'.format(self.m_max)
        z_min = '{:.2f}'.format(self.z_min)
        z_max = '{:.2f}'.format(self.z_max)

        return 'M_min: ' + m_min + '  M_max: ' + m_max + '  n_Masses: '+ str(self.nMasses) + '\n'\
               + '  z_min: ' + z_min + '  z_max: ' + z_max + '  n_zs: ' + str(self.nZs) +  '\n'\
               +'  Mass function: ' + self.mass_function + '  Mass definition: ' + self.mdef

### AUXILIARY FUNCTIONS
    def get_matter_consistency(self, exp):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over dark matter
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0
        I = np.trapz(self.hcos.nzm*self.hcos.bh*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)
        self.m_consistency = 1 - I # A function of z

    def get_galaxy_consistency(self, exp, survey_name, lmax_proj=None):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over galaxy number density
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        if lmax_proj is None:
            lmax_proj = self.lmax_out
        ugal_proj_of_Mlow = np.zeros((len(self.hcos.zs), lmax_proj+1))
        for i, z in enumerate(self.hcos.zs):
            ugal_proj_of_Mlow[i, :] = tls.pkToPell(self.hcos.comoving_radial_distance(self.hcos.zs[i]),
                                        self.hcos.ks, self.hcos.uk_profiles['nfw'][i, 0],
                                        ellmax=lmax_proj) # A function of z and k
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0
        I = np.trapz(self.hcos.nzm*self.hcos.bh*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)
        W_of_Mlow = (self.hcos.hods[survey_name]['Nc'][:, 0] + self.hcos.hods[survey_name]['Ns'][:, 0])[:,None]\
                    / self.hcos.hods[survey_name]['ngal'][:,None] * ugal_proj_of_Mlow # A function of z and k
        self.g_consistency = ((1 - I)/(self.hcos.ms[0]/self.hcos.rho_matter_z(0)))[:,None]*W_of_Mlow #Function of z & k

    def get_tsz_consistency(self, exp, lmax_proj=None):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over tsz emission.
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        if lmax_proj is None:
            lmax_proj = self.lmax_out

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        tsz_filter = exp.get_tsz_filter()

        W_of_Mlow = np.zeros((len(self.hcos.zs), lmax_proj + 1))
        for i, z in enumerate(self.hcos.zs):
            W_of_Mlow[i, :] = tsz_filter * tls.pkToPell(self.hcos.comoving_radial_distance(self.hcos.zs[i]),
                                                   self.hcos.ks, self.hcos.pk_profiles['y'][i, 0],
                                                   ellmax=lmax_proj)  # A function of z and k
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut < self.hcos.ms] = 0
        I = np.trapz(self.hcos.nzm * self.hcos.bh * self.hcos.ms / self.hcos.rho_matter_z(0) * mMask, self.hcos.ms,
                     axis=-1)
        self.y_consistency = ((1 - I) / (self.hcos.ms[0] / self.hcos.rho_matter_z(0)))[:,None] * W_of_Mlow

    def get_cib_consistency(self, exp, lmax_proj=None):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over CIB emission.
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        if lmax_proj is None:
            lmax_proj = self.lmax_out
        self.get_CIB_filters(exp)
        ucen_plus_usat_of_Mlow = np.zeros((len(self.hcos.zs), lmax_proj+1))
        for i, z in enumerate(self.hcos.zs):
            u_of_Mlow_proj = tls.pkToPell(self.hcos.comoving_radial_distance(self.hcos.zs[i]),
                                        self.hcos.ks, self.hcos.uk_profiles['nfw'][i, 0, :], ellmax=lmax_proj)
            u_cen = self.CIB_central_filter[:, i, 0]
            u_sat = self.CIB_satellite_filter[:, i, 0] * u_of_Mlow_proj
            ucen_plus_usat_of_Mlow[i, :] = u_cen + u_sat # A function of z and k
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0
        I = np.trapz(self.hcos.nzm*self.hcos.bh*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)
        W_of_Mlow =  ucen_plus_usat_of_Mlow # A function of z and k
        self.I_consistency = ((1 - I)/(self.hcos.ms[0]/self.hcos.rho_matter_z(0)))[:,None]*W_of_Mlow # Function of z & k

    def get_CIB_filters(self, exp):
        """
        Get f_cen and f_sat factors for CIB halo model scaled by foreground cleaning weights. That is,
        compute \Sum_{\nu} f^{\nu}(z,M) w^{\nu, ILC}_l. While you're at it, convert to CMB units.
        Input:
            * exp = a qest.experiment object
        """
        if len(exp.freq_GHz)>1:
            f_cen_array = np.zeros((len(exp.freq_GHz), len(self.hcos.zs), len(self.hcos.ms)))
            f_sat_array = f_cen_array.copy()
            for i, freq in enumerate(np.array(exp.freq_GHz*1e9)):
                freq = np.array([freq])
                f_cen_array[i, :, :] = tls.from_Jypersr_to_uK(freq[0]*1e-9) * self.hcos._get_fcen(freq)[:,:,0]
                f_sat_array[i, :, :] = tls.from_Jypersr_to_uK(freq[0]*1e-9)\
                                       * self.hcos._get_fsat(freq, cibinteg='trap', satmf='Tinker')[:,:,0]
            # Compute \Sum_{\nu} f^{\nu}(z,M) w^{\nu, ILC}_l
            self.CIB_central_filter = np.sum(exp.ILC_weights[:,:,None,None] * f_cen_array, axis=1)
            self.CIB_satellite_filter = np.sum(exp.ILC_weights[:,:,None,None] * f_sat_array, axis=1)
        else:
            # Single-frequency scenario. Return two (nZs, nMs) array containing f_cen(M,z) and f_sat(M,z)
            # Compute \Sum_{\nu} f^{\nu}(z,M) w^{\nu, ILC}_l
            # CEV: since: tls.from_Jypersr_to_uK returns filter in T_CMB[muK]
            self.CIB_central_filter = tls.from_Jypersr_to_uK(exp.freq_GHz)\
                                      * self.hcos._get_fcen(exp.freq_GHz*1e9)[:,:,0][np.newaxis,:,:]
            self.CIB_satellite_filter = tls.from_Jypersr_to_uK(exp.freq_GHz) \
                                        * self.hcos._get_fsat(exp.freq_GHz*1e9, cibinteg='trap',
                                                             satmf='Tinker')[:,:,0][np.newaxis,:,:]

    def get_g_cross_kappa(self, exp, survey_name, gzs, gdndz, damp_1h_prof=True, fftlog_way=True, gal_consistency=False):
        """
        Calculate galaxy cross CMB lensing spectrum. This is a for a test.
        Input:
            * exp = a qest.experiment object
            * survey_name = str. Name of the HOD
            * gzs = array of floats. Redshifts at which the dndz is defined
            * gdndz = array of floats. dndz of the galaxy sample, at the zs given by gzs. Need not be normalized
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
        """
        hcos = self.hcos
        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)
        self.get_matter_consistency(exp)

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)

        #nx = self.lmax_out+1
        nx = len(ells_out) if fftlog_way else exp.pix.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_ps = np.zeros([nx,self.nZs])+0j; twoH_ps = oneH_ps.copy()
        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_ps = np.zeros([nx,self.nMasses])+0j
            itgnd_2h_1g = itgnd_1h_ps.copy(); itgnd_2h_1m = itgnd_1h_ps.copy()

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks, hcos.uk_profiles['nfw'][i,j])(ells_out)
                gal = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks, hcos.uk_profiles['nfw'][i, j])(ells_out)
                # TODO: should ngal in denominator depend on z? ms_rescaled doesn't
                galfft = gal / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal,exp.pix).fft / \
                                                                                         hcos.hods[survey_name]['ngal'][
                                                                                             i]
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]

                # CEV: TODO: this actually has a big impact at low L. Check how accurate it is, it looks good but i need to have it more clear.
                if damp_1h_prof:
                    gal_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                       hcos.ks, hcos.uk_profiles['nfw'][i, j]
                                       *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))))(ells_out)
                    kap_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                            hcos.uk_profiles['nfw'][i, j] *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))))(ells_out)
                    galfft_damp = gal_damp / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal,
                                                                                                        exp.pix).fft / \
                                                                                        hcos.hods[survey_name]['ngal'][
                                                                                            i]
                    kfft_damp = kap_damp * self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp,
                                                                                                  exp.pix).fft * \
                                                                                  self.ms_rescaled[j]
                else:
                    kap_damp = kap; gal_damp = gal; kfft_damp = kfft

                mean_Ngal = hcos.hods[survey_name]['Nc'][i, j] + hcos.hods[survey_name]['Ns'][i, j]
                # Accumulate the itgnds
                itgnd_1h_ps[:,j] = mean_Ngal * galfft_damp * np.conjugate(kfft_damp)*hcos.nzm[i,j]
                # TODO: Implement 2h including consistency
                itgnd_2h_1g[:, j] = mean_Ngal * np.conjugate(galfft)*hcos.nzm[i,j]*hcos.bh[i,j]
                itgnd_2h_1m[:, j] = np.conjugate(kfft)*hcos.nzm[i,j]*hcos.bh[i,j]

            # Perform the m integrals
            oneH_ps[:,i]=np.trapz(itgnd_1h_ps,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i])(ells_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            twoH_ps[:, i] = (np.trapz(itgnd_2h_1g, hcos.ms, axis=-1) + self.g_consistency[i])\
                            * (np.trapz(itgnd_2h_1m, hcos.ms, axis=-1) + self.m_consistency[i]) * pk
        # Integrate over z
        gk_intgrnd = tls.limber_itgrnd_kernel(hcos, 2) * tls.gal_window(hcos, hcos.zs, gzs, gdndz)\
                     * tls.my_lensing_window(hcos, 1100.)
        ps_oneH = np.trapz( oneH_ps * gk_intgrnd, hcos.zs, axis=-1)
        ps_twoH = np.trapz( twoH_ps * gk_intgrnd, hcos.zs, axis=-1)
        return ps_oneH, ps_twoH

### BIASES CALCULATIONS

# CEV: TODO: potentially this can be generalized to g s g where you pick which s you want.

# tSZ
    def get_tsz_cross_biases(self, exp, gzs, gdndz, gzs2=None, gdndz2=None, bin_width_out=30, survey_name='LSST',
                             damp_1h_prof=True, gal_consistency=False, tsz_consistency=False):
        """
        Calculate the tsz biases to the cross-correlation of CMB lensing with a galaxy survey, (C^{g\phi}_L)
        given an "experiment" object (defined in qest.py)
        Uses serialization/pickling to parallelize the calculation of each z-point in the integrands
        Input:
            * exp = a qest.experiment object
            * gzs = array. Redsfhits at which the dndz is defined. Assumed to be zero otherwise.
            * gdndz = array of same size as gzs. The dndz of the galaxy sample (does not need to be normalized)
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) survey_name = str. Name labelling the HOD characterizing the survey we are x-ing lensing with
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15.
            * (optional) tsz_consistency = Bool. Whether to impose consistency condition on tSZ to correct for missing
                              low mass halos in integrals a la Schmidt 15.
        """
        # CEV: hcos created at init: 
        # self.hcos = hm.HaloModel(zs,ks,ms=ms,mass_function=mass_function,params=cosmoParams,mdef=mdef)
        # self.hcos.add_battaglia_pres_profile("y",family="pres",xmax=xmax,nxs=nxs, param_override=self.tsz_param_override)
        # self.hcos.set_cibParams(cib_model)
        # CEV: you add HODs to correlate with before running cross biases:
        # hm_calc = biases.hm_framework(cosmoParams=cosmoParams, m_min=Mmin, nZs=nZs, nMasses=nMasses, cib_model=cib_model, z_max=z_max)
        # hm_calc.hcos.add_hod(name=survey_name, mthresh=10**11.5+hm_calc.hcos.zs*0.) 
        # hm_calc.get_tsz_cross_biases(experiment, z_mean_gal, surface_ngal_of_z_gal, survey_name=survey_name)
        hcos = self.hcos
        # CEV: Low mass corrections from Schmidt-style. Deals with problems from not integrating to low enough mass. 
        # In principle it should be 0 to inf.
        # CEV: TODO: test importance of these once code is working.
        if tsz_consistency:
            self.get_tsz_consistency(exp, lmax_proj=exp.lmax)
        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)

        # CEV: incorporate new redshift for galaxy tracer. Useful flags:
        if gzs2 is None and gdndz2 is not None:
            print("gzs2 is None but gdndz2 is not None. Assuming zs for tracer 2 are the same as tracer 1")
            gzs2 = np.copy(gzs)
        elif gzs2 is None and gdndz2 is None:
            print("gdndz2 and gzs2 not provided. Assuming same as gzs and gdndz")
            gdndz2 = np.copy(gdndz)
            gzs2 = np.copy(gzs)
        elif gzs2 is not None and gdndz2 is None:
            raise ValueError("gzs2 is provided but gdndz2 is None. Did you forget about it?")
        elif gzs2 is not None and gdndz2 is not None:
            print("Using provided gzs2 and gdndz2.")

        # Output ells
        ells_out = np.linspace(1, self.lmax_out)
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)

        nx = len(ells_out)

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        # CEV: you can think of this as a multiplicative transfer function
        # CEV: TODO: undestand
        exp.tsz_filter = exp.get_tsz_filter()

        # The one and two halo bias terms -- these store the itgnd to be integrated over z.
        # Dimensions depend on method
        oneH_cross = np.zeros([nx,self.nZs])+0j
        twoH_cross = np.zeros([nx,self.nZs])+0j

        # Run in parallel
        hm_minimal = Hm_minimal(self)
        exp_minimal = exp

        n = len(hcos.zs) # CEV: so zs = np.linspace(z_min,z_max,nZs), but then we are gonna integrate over np.arange(len(zs))? Yes becuase n is used as the index of the redshift, not the redshift itself.

        # CEV: map the function to each redshift slice. Each of the variables from 2 to end are the inputs to tsZ_cross_itgrnds_each_z.
        # CEV: 'map' applies the function to each element of the first argument (here np.arange(n)) and the other arguments are just repeated n times.
        # CEV: TODO: will need to edit this according to tsZ_cross_itgrnds_each_z new inputs.
        outputs = map(self.tsZ_cross_itgrnds_each_z, np.arange(n), n * [ells_out],
                               n * [damp_1h_prof], n * [exp_minimal], n * [hm_minimal], n * [survey_name])

        # CEV: three dots mean "as many colons as needed to make the shape work out" in this case, tsz_cross_itgrnds_each_z returns two numbers? for 1h and 2h terms, so those in the end oneH_cross and twoH_cross are just 1d arrays of len(ells_out)?
        for idx, itgnds_at_i in enumerate(outputs):
            oneH_cross[...,idx], twoH_cross[...,idx] = itgnds_at_i

        # Integrate over z
        # CEV: TODO: eventually allow G and g to be different. For now this will be first approximation.
        gyg_intgrnd = tls.limber_itgrnd_kernel(hcos, 3) \
                        * tls.gal_window(hcos, hcos.zs, gzs, gdndz) \
                        * tls.y_window(hcos) \
                        * tls.gal_window(hcos, hcos.zs, gzs2, gdndz2)
        
        exp.biases['tsz']['cross_w_gals']['1h'] = np.trapz(oneH_cross * gyg_intgrnd, hcos.zs, axis=-1)
        exp.biases['tsz']['cross_w_gals']['2h'] = np.trapz(twoH_cross * gyg_intgrnd, hcos.zs, axis=-1)

        exp.biases['ells'] = ells_out
        return
     

    def tsZ_cross_itgrnds_each_z(self, i, ells_out, damp_1h_prof, exp_minimal, hm_minimal, survey_name):
        """
        Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
        Input:
            * i = int. Index of the ith redshift in the halo model calculation
            * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * exp_minimal = instance of qest.exp_minimal(exp)
            * hm_minimal = instance of biases.hm_minimal(hm_framework)
        """
        
        nx = len(ells_out)
        ells_in = np.arange(0, exp_minimal.lmax + 1) # CEV: always used in pkToPell calls. I think it's small ell?

        # Temporary storage. CEV: nMasses is the # of steps for the mass integral, given by user at init.
        itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        # For term one where QE acts simply on profiles
        itgnd_2h_1_2g = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        itgnd_2h_1_1g = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        # For terms where first mass int has to be done before QE
        itgnd_2h_y_Gg = itgnd_1h_cross.copy(); # to store the already integrated over M profile after gone through QE * all prefactors and such
        # itgnd_2h_y_Gg = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        itgnd_2h_g_yG = itgnd_1h_cross.copy();
        # The integrands for the first M int
        itgnd_y_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) 
        itgnd_g_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) 

        # To keep QE calls tidy
        QE = lambda prof_T, prof_g: exp_minimal.get_kSZ_qe(prof_T, prof_g, 
                                                           exp_minimal.cltt_tot, exp_minimal.ls, 
                                                           exp_minimal.cl_gg, exp_minimal.cl_taug,
                                                           exp_minimal.weights_mat_total, exp_minimal.nodes)
        
        # Project the matter power spectrum for two-halo terms
        # CEV: hm_minimal.Pzk is actually hm_full.hcos.Pzk, so Pzk is already defined at hcos.zs
        pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i])(ells_in)
        pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i])(ells_out)

        # Integral over M for 2halo bispectrum. This will later go into a QE
        for j, m in enumerate(hm_minimal.ms):
            if m > exp_minimal.massCut: continue

            # Mean number of galaxies in a halo of mass m at redshift i to be applied to nfw profiles
            mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]
    
            y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                                      hm_minimal.pk_profiles['y'][i, j])(ells_in) # = y_{3D}(l/chi, M=j, z=i)
            itgnd_y_for_2hbispec[..., j] = y * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]

            g = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                             hm_minimal.uk_profiles['nfw'][i, j])(ells_in) # = u_{3D}(l/chi, M=j, z=i)

            gfft = g / hm_minimal.hods[survey_name]['ngal'][i]

            itgnd_g_for_2hbispec[..., j] = mean_Ngal * np.conjugate(gfft) * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]

        int_over_M_of_y = pk_of_l * (
                    np.trapz(itgnd_y_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.y_consistency[i])
        
        int_over_M_of_g = pk_of_l * (
                    np.trapz(itgnd_g_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.g_consistency[i]) # CEV: g_consistency = 0 if gal_consistency = False.

        # M integral.
        # CEV: ms is array of masses from mmin to mmax in nMasses steps, for integration, given by user at init.
        for j, m in enumerate(hm_minimal.ms):
            if m > exp_minimal.massCut: continue # massCut given to experiment by user.
            # CEV: TODO: i don't find anywhere exp.tsz_filter being defined? I think exp_minimal.tsz_filter = None...
            # CEV: hm_minimal.pk_profiles['y'] comes directly from hmvec hcos.pk_profiles['y'].
            y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                                      hm_minimal.pk_profiles['y'][i, j])(ells_in)
            
            # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
            # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 #TODO: why do you say that?
            Gal = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                               hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j])(ells_out)
            
            g = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                               hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j])(ells_in)
            
            # TODO: should ngal in denominator depend on z? ms_rescaled doesn't
            # CEV: TODO: understand why it needs to be conjugated.
            Galfft = Gal / hm_minimal.hods[survey_name]['ngal'][i]
            gfft = g / hm_minimal.hods[survey_name]['ngal'][i]

            phicfft_1 = QE(y, gfft) # CEV: TODO: does ngal depend on k? why is it going into QE?
            phicfft_2_int = QE(int_over_M_of_y, gfft) 
            phicfft_3_int = QE(y, int_over_M_of_g)

            # Consider damping the profiles at low k in 1h terms to avoid it exceeding many-halo amplitude 
            if damp_1h_prof:
                y_damp = exp_minimal.tsz_filter \
                         * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                        hm_minimal.pk_profiles['y'][i, j]
                                        * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_in)
                Gal_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                        hm_minimal.uk_profiles['nfw'][i, j]
                                        * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_out)
                g_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                        hm_minimal.uk_profiles['nfw'][i, j]
                                        * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_in)
                
                Galfft_damp = Gal_damp / hm_minimal.hods[survey_name]['ngal'][i] 
                gfft_damp = g_damp / hm_minimal.hods[survey_name]['ngal'][i] 

                phicfft_1_damp = QE(y_damp, gfft_damp) # CEV: only this one is needed for correcting 1h.
            else:
                y_damp = y; 
                g_damp = g; 
                Gal_damp = Gal

                Galfft_damp = Galfft
                phicfft_1_damp = phicfft_1

            # Accumulate the itgnds
            mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]
            # 1h
            itgnd_1h_cross[..., j] = mean_Ngal * np.conjugate(Galfft_damp) * phicfft_1_damp * mean_Ngal * hm_minimal.nzm[i, j]
            # 2h
            # 1
            itgnd_2h_1_1g[..., j] = mean_Ngal * np.conjugate(Galfft) * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j] # Missing P(k) here. no, see below.
            itgnd_2h_1_2g[..., j] = mean_Ngal * phicfft_1 * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]
            # 2
            itgnd_2h_y_Gg[..., j] = mean_Ngal * np.conjugate(Galfft) * mean_Ngal * phicfft_2_int \
                                    * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]
            # 3
            itgnd_2h_g_yG[..., j] = mean_Ngal * np.conjugate(Galfft) * phicfft_3_int \
                                    * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]

        # Perform the m integrals
        # 1h
        oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)
        # 2h CEV: strange way of doing it but ok.
        thoH_cross_1_1 = np.trapz(itgnd_2h_1_1g, hm_minimal.ms, axis=-1)
        # CEV: TODO: understand how this consistency is applied in general. Make sure it's applied consistently.
        twoH_cross_at_i = np.trapz(itgnd_2h_1_2g, hm_minimal.ms, axis=-1) * (thoH_cross_1_1 + hm_minimal.g_consistency[i]) * pk_of_L \
                          + np.trapz(itgnd_2h_y_Gg, hm_minimal.ms, axis=-1) + np.trapz(itgnd_2h_g_yG, hm_minimal.ms, axis=-1)
        return oneH_cross_at_i, twoH_cross_at_i

# CIB
    def get_cib_cross_biases(self, exp, gzs, gdndz, gzs2=None, gdndz2=None, bin_width_out=30, survey_name='LSST',
                             damp_1h_prof=True, gal_consistency=False, cib_consistency=False, max_workers=None):
        """
        Calculate the CIB biases to the cross-correlation of CMB lensing with a galaxy survey, (C^{g\phi}_L)
        given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * gzs = array. Redsfhits at which the dndz is defined. Assumed to be zero otherwise.
            * gdndz = array of same size as gzs. The dndz of the galaxy sample (does not need to be normalized)
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) survey_name = str. Name labelling the HOD characterizing the survey we are x-ing lensing with
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos

        # CEV: Low mass corrections. CEV: TODO: test importance of these once code is working.
        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)
        if cib_consistency:
            self.get_cib_consistency(exp, lmax_proj=exp.lmax)

        # CEV: incorporate new redshift for galaxy tracer. Useful flags:
        if gzs2 is None and gdndz2 is not None:
            print("gzs2 is None but gdndz2 is not None. Assuming zs for tracer 2 are the same as tracer 1")
            gzs2 = np.copy(gzs)
        elif gzs2 is None and gdndz2 is None:
            print("gdndz2 and gzs2 not provided. Assuming same as gzs and gdndz")
            gdndz2 = np.copy(gdndz)
            gzs2 = np.copy(gzs)
        elif gzs2 is not None and gdndz2 is None:
            raise ValueError("gzs2 is provided but gdndz2 is None. Did you forget about it?")
        elif gzs2 is not None and gdndz2 is not None:
            print("Using provided gzs2 and gdndz2.")

        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        # CEV: Gets self.CIB_central_filter and self.CIB_satellite_filter , to be applied to the nfw profiles.
        # It can also take into account multifreq foreground cleaning.
        self.get_CIB_filters(exp) # [T_CMB muK]

        # Output ells
        ells_out = np.linspace(1, self.lmax_out)
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)

        #nx = self.lmax_out+1
        nx = len(ells_out)

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_cross = np.zeros([nx,self.nZs])+0j
        twoH_cross = oneH_cross.copy()

        # Run in parallel
        hm_minimal = Hm_minimal(self)
        exp_minimal = exp

        n = len(hcos.zs)
        outputs = map(self.cib_cross_itgrnds_each_z, np.arange(n), n * [ells_out],
                               n * [damp_1h_prof], n * [exp_minimal], n * [hm_minimal], n * [survey_name])

        for idx, itgnds_at_i in enumerate(outputs):
            oneH_cross[...,idx], twoH_cross[...,idx] = itgnds_at_i

        # itgnd factors from Limber projection (adapted to hmvec conventions)
        # CEV: need to convert CIB from T_CMB to dimensionless.
        # CEV: TODO: check that you've done this correctly, both central and sat have T_CMB units.
        gIg_itgnd = tls.limber_itgrnd_kernel(hcos, 3) \
                    * tls.gal_window(hcos, hcos.zs, gzs, gdndz) \
                    * tls.CIB_window(hcos) / self.T_CMB \
                    * tls.gal_window(hcos, hcos.zs, gzs2, gdndz2)

        # Integrate over z
        exp.biases['cib']['cross_w_gals']['1h'] = np.trapz( oneH_cross*gIg_itgnd, hcos.zs, axis=-1)
        exp.biases['cib']['cross_w_gals']['2h'] = np.trapz( twoH_cross*gIg_itgnd, hcos.zs, axis=-1)

        exp.biases['ells'] = ells_out
        return

    def cib_cross_itgrnds_each_z(self, i, ells_out, damp_1h_prof, exp_minimal, hm_minimal, survey_name):
        """
        Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
        Input:
            * i = int. Index of the ith redshift in the halo model calculation
            * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * exp_minimal = instance of qest.exp_minimal(exp)
            * hm_minimal = instance of biases.hm_minimal(hm_framework)
        """
        
        nx = len(ells_out)
        ells_in = np.arange(0, exp_minimal.lmax + 1)

        # Temporary storage. CEV: nMasses is the # of steps for the mass integral, given by user at init.
        itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        # For term one where QE acts simply on profiles
        itgnd_2h_1_2g = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        itgnd_2h_1_1g = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        # For terms where first mass int has to be done before QE
        itgnd_2h_y_Gg = itgnd_1h_cross.copy(); # to store the already integrated over M profile after gone through QE * all prefactors and such
        # itgnd_2h_y_Gg = np.zeros([nx, hm_minimal.nMasses]) + 0j 
        itgnd_2h_g_yG = itgnd_1h_cross.copy();
        # The integrands for the first M int
        itgnd_I_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) 
        itgnd_g_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) 

        # To keep QE calls tidy, define
        QE = lambda prof_T, prof_g: exp_minimal.get_kSZ_qe(prof_T, prof_g, 
                                                           exp_minimal.cltt_tot, exp_minimal.ls, 
                                                           exp_minimal.cl_gg, exp_minimal.cl_taug,
                                                           exp_minimal.weights_mat_total, exp_minimal.nodes)

        # Project the matter power spectrum for two-halo terms
        pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i])(ells_in)
        pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i])(ells_out)

        # Integral over M for 2halo trispectrum. This will later go into a QE
        # CEV: compute integrals that then go into QE
        for j, m in enumerate(hm_minimal.ms):
            if m > exp_minimal.massCut: continue

            # CEV: prepare for mass integral over CIB profile.
            u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                             hm_minimal.uk_profiles['nfw'][i, j])(ells_in)
            # CEV: TODO: understand this u factors.
            u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
            u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

            itgnd_I_for_2hbispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh[i, j] * (u_cen + u_sat)

            # CEV: prepare for mass integral over galaxy profile.
            mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]

            g = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                             hm_minimal.uk_profiles['nfw'][i, j])(ells_in) # = u_{3D}(l/chi, M=j, z=i)
            gfft = g / hm_minimal.hods[survey_name]['ngal'][i]

            itgnd_g_for_2hbispec[..., j] = mean_Ngal * np.conjugate(gfft) * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]

        # Mass integral over CIB profile that will go into QE.
        int_over_M_of_I = pk_of_l * (
                    np.trapz(itgnd_I_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.I_consistency[i])
        # Mass integral over galaxy profile that will go into QE.
        int_over_M_of_g = pk_of_l * (
                    np.trapz(itgnd_g_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.g_consistency[i]) # CEV: g_consistency = 0 if gal_consistency = False.


        # M integral.
        for j, m in enumerate(hm_minimal.ms):
            if m > exp_minimal.massCut: continue
            # project the I profiles
            u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                             hm_minimal.uk_profiles['nfw'][i, j])(ells_in)
            u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
            u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

            # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
            # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 # TODO:Why?
            Gal = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                               hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j])(ells_out)
            
            g = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                               hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j])(ells_in)

            # CEV: TODO: understand why it needs to be conjugated.
            Galfft = Gal / hm_minimal.hods[survey_name]['ngal'][i]
            gfft = g / hm_minimal.hods[survey_name]['ngal'][i]

            # CEV: TODO: for now I'll ignore damping.
            phicfft_1 = QE(u_cen + u_sat, gfft)
            phicfft_2_int = QE(int_over_M_of_I, gfft)
            phicfft_3_int = QE(u_cen + u_sat, int_over_M_of_g)

            if damp_1h_prof:

                u_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                      hm_minimal.uk_profiles['nfw'][i, j]
                                      * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_in)
                u_sat_damp = hm_minimal.CIB_satellite_filter[:, i, j] * u_damp

                # CEV: me
                Gal_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                        hm_minimal.uk_profiles['nfw'][i, j]
                                        * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_out)

                g_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                        hm_minimal.uk_profiles['nfw'][i, j]
                                        * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_in)

                Galfft_damp = Gal_damp / hm_minimal.hods[survey_name]['ngal'][i] 
                gfft_damp = g_damp / hm_minimal.hods[survey_name]['ngal'][i] 

                # CEV: ASK: Anton why he doesn't damp u_cen.
                phicfft_ucen_g_damp = QE(u_cen, gfft_damp)
                phicfft_usat_g_damp = QE(u_sat_damp, gfft_damp)
                phicfft_1_damp = (phicfft_ucen_g_damp + phicfft_usat_g_damp)
            else:
                # galfft_damp = galfft;
                # phicfft_ucen_usat_damp = phicfft_ucen_usat;
                # phicfft_usat_usat_damp = phicfft_usat_usat

                Galfft_damp = Galfft
                phicfft_1_damp = phicfft_1

            # Accumulate the itgnds
            mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]
            # 1h
            itgnd_1h_cross[..., j] = mean_Ngal * np.conjugate(Galfft_damp) * phicfft_1_damp * mean_Ngal * hm_minimal.nzm[i, j]
            # 2h
            # 1
            itgnd_2h_1_1g[..., j] = mean_Ngal * np.conjugate(Galfft) * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]
            itgnd_2h_1_2g[..., j] = mean_Ngal * phicfft_1 * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]
            # 2
            itgnd_2h_y_Gg[..., j] = mean_Ngal * np.conjugate(Galfft) * mean_Ngal * phicfft_2_int \
                                    * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]
            # 3
            itgnd_2h_g_yG[..., j] = mean_Ngal * np.conjugate(Galfft) * phicfft_3_int \
                                    * hm_minimal.nzm[i, j] * hm_minimal.bh[i, j]

        # Perform the m integrals
        # 1h
        oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)
        # 2h CEV: strange way of doing it but ok.
        thoH_cross_1_1 = np.trapz(itgnd_2h_1_1g, hm_minimal.ms, axis=-1)
        # CEV: TODO: understand how this consistency is applied in general. Make sure it's applied consistently.
        twoH_cross_at_i = np.trapz(itgnd_2h_1_2g, hm_minimal.ms, axis=-1) * (thoH_cross_1_1 + hm_minimal.g_consistency[i]) * pk_of_L \
                          + np.trapz(itgnd_2h_y_Gg, hm_minimal.ms, axis=-1) + np.trapz(itgnd_2h_g_yG, hm_minimal.ms, axis=-1)
        return oneH_cross_at_i, twoH_cross_at_i
    