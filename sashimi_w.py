import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy import special
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.special import cbrt, gammainc, erf, erfc, hyp2f1
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from numpy.polynomial.hermite import hermgauss
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning, append = 1)




###############################
#  Constants
############################### 
cm       = 1.
km       = 1.e5*cm
s        = 1.
gram     = 1.
c        = 2.99792e+10*cm/s
G        = 6.6742e-8*cm**3/gram/s**2
Mpc      = 3.086e+24*cm
kpc      = Mpc/1000.
pc       = kpc/1000.
Msolar   = 1.988435e+33*gram
GeV      = 1.7827e-24*gram
keV      = 1.0e-6*GeV




###############################
#  Matter Power Spectrum
############################### 
""" WMAP7 """ 
filename_PS      = "WMAP7_camb_matterpower_z0_extrapolated.dat"
PowerSpectrum    = np.genfromtxt(filename_PS, skip_header = 5)
Pk_file, k_file  = PowerSpectrum[:,0], PowerSpectrum[:,1]
k_min            = k_file.min() * 1.15
k_max            = k_file.max() * 0.86
Pk_interp        = interp1d(k_file, Pk_file)


""" Cosmology from WMAP7 """ 
PS_cosmology  = np.genfromtxt(filename_PS,max_rows=5)
Omegar        = 0.0
Omega0        = 1.0
OmegaB        = PS_cosmology[0]
OmegaM        = PS_cosmology[1]
OmegaC        = OmegaM - OmegaB
OmegaL        = Omega0 - OmegaC - Omegar
pOmega        = [OmegaC+OmegaB,Omegar,OmegaL]
h             = PS_cosmology[2]
H0            = h*100*km/s/Mpc 
rhocrit0      = 3*pow(H0,2)*pow(8.0*np.pi*G,-1)
sigma_8       = PS_cosmology[4]




class subhalos:

    def __init__(self, mass_wdm=1.5):
        self.mass_wdm = mass_wdm

        ########################################################
        #  Truncated Power Spectrum -- Variance -- Concentration  
        #  Functions adapted from A. Ludlow, arXiv: 1601.0262 
        ########################################################

        self.G_units       = G * (Msolar/Mpc)*(s**2/km**2) 
        self.Rhocrit_z     = 3.0/(8.0 * np.pi * self.G_units ) * 1e4 # M_solar/Mpc^3/h^2
        self.Omz           = OmegaM
        self.Rhomean_z     = self.Rhocrit_z * self.Omz # M_solar/Mpc/h^2

        """ Filter mass [Msolar/h] and filter radius [Mpc/h] """ 
        self.MassMin       = 1e-12
        self.MassMax       = 1e18
        self.dlogm         = (np.log10(self.MassMax) - np.log10(self.MassMin)) / (100-1)
        self.logM0         = np.log10(self.MassMin) + np.arange(100)*self.dlogm + 0.5*self.dlogm
        self.filter_Mass   = 10**self.logM0
        self.R             = cbrt(self.filter_Mass / (4/3 * np.pi * self.Rhomean_z)) / 2.5 # Mpc/h Sharp-k filter, Schneider (2014) 

        """ Integrate Pk and obtain Sigma(M) with sharp-k filter """ 
        self.log_k_min     = np.log10(k_min)
        self.log_k_max     = np.log10((9*np.pi/2)**(1./3)/self.R)
        self.dlogk         = (self.log_k_max - self.log_k_min)/(500 - 1)
        self.tot_sum       = 0.
        """ Transfer function WDM Power Spectrum, Viel et al. (2011) """ 
        self.a             = 0.049*(1/self.mass_wdm)**1.11 * (OmegaC/0.25)**0.11 * (h/0.7)**1.22 # Mpc/h
        for ii in range(500):
            self.logk      = self.log_k_min + self.dlogk*ii
            self.sum_rect  = Pk_interp(10**self.logk)*(10**self.logk)**2*10**self.logk*np.log(10) * \
                                (1+(self.a*(10**self.logk))**(2*1.12))**(-5./1.12)
            self.tot_sum   = self.tot_sum + self.sum_rect         
        self.log_k_min     = self.log_k_min - self.dlogk
        self.sum_rect_min  = Pk_interp(10**self.log_k_min)*(10**self.log_k_min)**2 * 10**self.log_k_min * np.log(10) * \
                                (1+(self.a*(10**self.log_k_min))**(2*1.12))**(-5./1.12)
        self.log_k_max     = self.log_k_max + self.dlogk
        self.sum_rect_max  = Pk_interp(10**self.log_k_max)*(10**self.log_k_max)**2 * 10**self.log_k_max * np.log(10) * \
                                (1+(self.a*(10**self.log_k_max))**(2*1.12))**(-5./1.12)
        self.sigma_sq      = (self.tot_sum + 0.5*self.sum_rect_min + 0.5*self.sum_rect_max) * self.dlogk
        self.sigma_sq     /= (2*np.pi**2)     
        
        ##### Sigma (M)
        self.Sigma         = np.sqrt(self.sigma_sq)
        self.sig_interp    = interp1d(self.filter_Mass, self.Sigma)
        self.MassIn8Mpc    = 4/3 * np.pi * 8**3 * self.Rhomean_z
        self.sig_8         = self.sig_interp(self.MassIn8Mpc)
        self.normalise     = self.sig_8 / sigma_8
        self.Sigma         = self.Sigma/self.normalise
        self.Sigma_Sq      = self.Sigma**2 
        self.Sigma_interp  = interp1d(self.filter_Mass,self.Sigma)  


    def dlnSigmadlnM_interp(self, M):
        lnSigma_Sq  = np.log(self.Sigma)
        lnMass      = np.log(self.filter_Mass)
        derivSigma   = np.diff(lnSigma_Sq) / np.diff(lnMass)
        return np.interp(M,lnMass[:-1],derivSigma)    



        ##################################################
        #  Calculate c(M,z) using Top-Hat filter
        #  Functions adapted from A. Ludlow, arXiv: 1601.0262   
        ##################################################

        """ Tophat """ 
    def TopHat(self,k, r):
        return 3.0/(k*r)**2 * (np.sin(k*r)/(k*r) - np.cos(k*r))  

    def SigmaIntegrand(self,k, r):
        om_wdm = OmegaM-OmegaB
        a = 0.049*(1/self.mass_wdm)**1.11 * (om_wdm/0.25)**0.11 * (h/0.7)**1.22 # Mpc/h
        nu = 1.12        
        return k**2 * Pk_interp(k) * self.TopHat(k,r)**2     * (1+(a*k)**(2*nu))**(-5./nu)

    """ Integration function """ 
    def integratePk_th(self,kmin, kmax, r):

        log_k_min_th   = np.log10(kmin) 
        log_k_max_th   = np.log10(kmax)
        dlogk_th       = (log_k_max_th - log_k_min_th)/(500 - 1)
        tot_sum_th     = 0. 

        logk_th = log_k_min_th+np.arange(500)*dlogk_th
        logk_th = np.reshape(logk_th,(500,1))
        sum_rect_th = self.SigmaIntegrand(10**logk_th, r) * 10**logk_th * np.log(10)
        tot_sum_th  = np.sum(sum_rect_th,axis=0)

        log_k_min_th    = log_k_min_th - dlogk_th
        sum_rect_min_th = self.SigmaIntegrand(10**log_k_min_th, r) * 10**log_k_min_th * np.log(10)
        log_k_max_th    = log_k_max_th + dlogk_th
        sum_rect_max_th = self.SigmaIntegrand(10**log_k_max_th, r) * 10**log_k_max_th * np.log(10)
        
        sigma_sq_th     = (tot_sum_th + 0.5*sum_rect_min_th + 0.5*sum_rect_max_th) * dlogk_th
        sigma_sq_th    /= (2*np.pi**2)
        return sigma_sq_th            

    def linear_growth_factor(self, Omega_m0, Omega_l0, z):
        if len(np.atleast_1d(z)) == 2:
            z1    = z[0]
            z2    = z[1] # z2 > z1                                                                                            

        if (len(np.atleast_1d(z)) == 1) or (len(np.atleast_1d(z)) > 2):
            z1    = 0.
            z2    = z    # z2 > z1                                                                                           
                 
        Omega_lz1 = Omega_l0 / (Omega_l0 + Omega_m0 * (1.+z1)**3)
        Omega_mz1 = 1. - Omega_lz1
        gz1       = (5./2.) * Omega_mz1 / (Omega_mz1**(4./7.) - Omega_lz1 + (1. + Omega_mz1/2.) * (1. + Omega_lz1/70.))
        Omega_lz2 = Omega_l0 / (Omega_l0 + Omega_m0 * (1.+z2)**3)
        Omega_mz2 = 1. - Omega_lz2
        gz2       = (5./2.) * Omega_mz2 / (Omega_mz2**(4./7.) - Omega_lz2 + (1. + Omega_mz2/2.) * (1. + Omega_lz2/70.))
        return (gz2 / (1.+z2)) / (gz1 / (1+z1))



    def conc200(self,M,z): 
        M = M/Msolar/h

        redshiftvect  = np.linspace(0,7,8) 

        R_th         = cbrt(self.filter_Mass / (4/3 * np.pi * self.Rhomean_z)) 
        Sigma_Sq_th     = np.zeros(len(self.filter_Mass))
        Sigma_Sq_th     = self.integratePk_th(k_min, k_max, R_th)
        Sigma_th        = np.sqrt(Sigma_Sq_th)
        sig_interp_th   = interp1d(self.filter_Mass, Sigma_th)
        MassIn8Mpc   = 4/3 * np.pi * 8**3 * self.Rhomean_z
        sig_8_th        = sig_interp_th(MassIn8Mpc)
        normalise_th    = sig_8_th / sigma_8
        Sigma_th       /= normalise_th
        Sigma_Sq_th     = Sigma_th**2 
              
        """ free model parameters, Ludlow et al. (2016) """ 
        A           = 650. / 200
        f           = 0.02
        delta_sc    = 1.686
        delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.-OmegaM, redshiftvect)
        OmegaL      = 1.-OmegaM
        sig2_interp_th = splrep(self.logM0-10., Sigma_Sq_th, k=1)



        if np.shape(M) == ():
            index = (np.abs(10**self.logM0 - M)).argmin()
            c_array     = 10**(np.arange(100) * 4./99.)
            M2          = (np.log(2.)-0.5) / (np.log(1.+c_array)-c_array/(1.+c_array))
            rho_2       = 200. * c_array**3 * M2
            rhoc        = rho_2 / (200. * A)
            z2          = (1. / OmegaM *(rhoc* (OmegaM*(1+z)**3 + OmegaL) - OmegaL))**0.3333 - 1.
            delta_sc_z2 = delta_sc / self.linear_growth_factor(OmegaM, OmegaL, z2)
            delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.-OmegaM, z)
            
            sig2fM_th      = splev(self.logM0[index] -10. + np.log10(f), sig2_interp_th)
            sig2M_th       = Sigma_Sq_th[index]
            sig2Min_th     = splev(np.log10(M2), sig2_interp_th)

            arg         = A*rhoc/c_array**3 - (1.-erf( (delta_sc_z2-delta_sc_0_vect) / np.sqrt(2.*(sig2fM_th-sig2M_th)) ))        
            mask        = np.isinf(arg) | np.isnan(arg)
            arg         = arg[mask == False]     

            c_array     = c_array[mask==False]
            conc_interp = interp1d(arg, c_array)
            c_nfw   = np.interp(0,arg,c_array)

        elif M.ndim == 1:
            M_reshaped = M.flatten()
            c_nfw = np.zeros(len(M_reshaped))
            for i in range(len(M_reshaped)):
                index = (np.abs(10**self.logM0 - M_reshaped[i])).argmin()
                c_array     = 10**(np.arange(100) * 4./99.)
                M2          = (np.log(2.)-0.5) / (np.log(1.+c_array)-c_array/(1.+c_array))
                rho_2       = 200. * c_array**3 * M2
                rhoc        = rho_2 / (200. * A)
                z2          = (1. / OmegaM *(rhoc* (OmegaM*(1+z)**3 + OmegaL) - OmegaL))**0.3333 - 1.
                delta_sc_z2 = delta_sc / self.linear_growth_factor(OmegaM, OmegaL, z2)
                delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.-OmegaM, z)
                
                sig2fM_th      = splev(self.logM0[index] -10. + np.log10(f), sig2_interp_th)
                sig2M_th       = Sigma_Sq_th[index]
                sig2Min_th     = splev(np.log10(M2), sig2_interp_th)

                arg         = A*rhoc/c_array**3 - (1.-erf( (delta_sc_z2-delta_sc_0_vect) / np.sqrt(2.*(sig2fM_th-sig2M_th)) ))        
                mask        = np.isinf(arg) | np.isnan(arg)
                arg         = arg[mask == False]      

                c_array     = c_array[mask==False]
                conc_interp = interp1d(arg, c_array)
                c_nfw[i]   = np.interp(0,arg,c_array)
            c_nfw = np.reshape(c_nfw,np.shape(M))                        

        elif M.ndim == 2:
            M_reshaped = M.flatten()
            z_reshaped = z.flatten()
            c_nfw = np.zeros(len(M_reshaped))
            for i in range(len(M_reshaped)):
                index = (np.abs(10**self.logM0 - M_reshaped[i])).argmin()
                c_array     = 10**(np.arange(100) * 4./99.)
                M2          = (np.log(2.)-0.5) / (np.log(1.+c_array)-c_array/(1.+c_array))
                rho_2       = 200. * c_array**3 * M2
                rhoc        = rho_2 / (200. * A)
                z2          = (1. / OmegaM *(rhoc* (OmegaM*(1+z_reshaped[i])**3 + OmegaL) - OmegaL))**0.3333 - 1.
                delta_sc_z2 = delta_sc / self.linear_growth_factor(OmegaM, OmegaL, z2)
                delta_sc_0_vect = delta_sc / self.linear_growth_factor(OmegaM, 1.-OmegaM, z_reshaped[i])
                
                sig2fM_th      = splev(self.logM0[index] -10. + np.log10(f), sig2_interp_th)
                sig2M_th       = Sigma_Sq_th[index]
                sig2Min_th     = splev(np.log10(M2), sig2_interp_th)

                arg         = A*rhoc/c_array**3 - (1.-erf( (delta_sc_z2-delta_sc_0_vect) / np.sqrt(2.*(sig2fM_th-sig2M_th)) ))        
                mask        = np.isinf(arg) | np.isnan(arg)
                arg         = arg[mask == False]      

                c_array     = c_array[mask==False]
                conc_interp = interp1d(arg, c_array)
                c_nfw[i]   = np.interp(0,arg,c_array)
            c_nfw = np.reshape(c_nfw,np.shape(M))         


        return c_nfw         
       


    ###############################
    #  Functions for subhalo model
    ###############################    

    def fc(self, x):
        return np.log(1+x)-x*pow(1+x,-1)

    def g(self, z):
        return (OmegaB+OmegaC)*(1.+z)**3+Omegar*(1+z)**4+OmegaL

    def rhocrit(self, z):
        return 3.0*pow(self.Hz(z),2)*pow(np.pi*8.0*G,-1)

    def Mvir_from_M200(self, M200, z):
        gz = self.g(z)
        c200 = self.conc200(M200,z)
        r200 = (3.0*M200/(4*np.pi*200*rhocrit0*gz))**(1./3.)
        rs = r200/c200
        fc200 = self.fc(c200)
        rhos = M200/(4*np.pi*rs**3*fc200)
        Dc = self.Delc(self.Omegaz(pOmega,z)-1.)
        rvir = optimize.fsolve(lambda r: 3.*(rs/r)**3*self.fc(r/rs)*rhos-Dc*rhocrit0*gz,r200)
        Mvir = 4*np.pi*rs**3*rhos*self.fc(rvir/rs)
        return Mvir

    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.Omegaz(pOmega,z)
        def ffunc(x):
            return np.power(x,3.0)*(np.log(1.0+1.0/x)-1.0/(1.0+x))
        def xfunc(f):
            p = a2 + a3*np.log(f) + a4*np.power(np.log(f),2.0)
            return np.power(a1*np.power(f,2.0*p)+(3.0/4.0)**2,-0.5)+2.0*f
        return self.Delc(Oz-1)/200.0*M200 \
            *np.power(self.conc200(M200,z) \
            *xfunc(self.Delc(Oz-1)/200.0*ffunc(1.0/self.conc200(M200,z))),-3.0)

    def growthD(self, z):
        Omega_Lz = OmegaL*pow(OmegaL+OmegaM*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz = pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
        phi0 = pow(OmegaM,4.0/7.0)-OmegaL+(1+OmegaM/2.0)*(1+OmegaL/70.0)
        return (Omega_Mz/OmegaM)*(phi0/phiz)*pow(1+z,-1)

    def xi(self, M):
        return pow(M*pow((1e+10)*pow(h,-1),-1),-1)

    def sigmaMz(self, M, z):
        sigmaM0 = self.Sigma_interp(M)
        sigmaMz = sigmaM0*self.growthD(z)
        return sigmaMz

    def dOdz(self, z):
        return -OmegaL*3*OmegaM*pow(1+z,2)*pow(OmegaL+OmegaM*pow(1+z,3),-2)

    def dDdz(self, z):
        Omega_Lz = OmegaL*pow(OmegaL+OmegaM*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz = pow(Omega_Mz,4.0/7.0)-Omega_Lz+(1+Omega_Mz/2.0)*(1+Omega_Lz/70.0)
        phi0 = pow(OmegaM,4.0/7.0)-OmegaL+(1+OmegaM/2.0)*(1+OmegaL/70.0)
        dphidz = self.dOdz(z)*((-4.0/7.0)*pow(Omega_Mz,-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.0+(1.0/70.0)-(3.0/2.0))
        return (phi0/OmegaM)*(-self.dOdz(z)*pow(phiz*(1+z),-1)-Omega_Mz*(dphidz*(1+z)+phiz)*pow(phiz,-2)*pow(1+z,-2))

    def Mzi(self, M0, z):
        a = 1.686*np.sqrt(2.0/np.pi)*self.dDdz(0)+1.0
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fM0 = pow(pow(self.sigmaMz(M0/q,0),2)-pow(self.sigmaMz(M0,0),2),-0.5)
        return M0*pow(1+z,a*fM0)*np.exp(-fM0*z)

    def Mzzi(self, M0, z, zi):
        Mzi0 = self.Mzi(M0,zi)
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0/q,zi),2)-pow(self.sigmaMz(Mzi0,zi),2),-0.5)
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(self.growthD(zi),-2)*self.dDdz(zi)+1)
        beta = -fMzi
        return Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))

    def Hz(self, z):
        return H0*np.sqrt(OmegaL+OmegaM*pow(1+z,3))

    def Omegaz(self, p, x):
        E=p[0]*pow(1+x,3)+p[1]*pow(1+x,2)+p[2]
        return p[0]*pow(1+x,3)*pow(E,-1)

    def Delc(self, x):
        return 18*pow(np.pi,2)+(82.*x)-39*pow(x,2)

    def dMdz(self, M0, z, zi, sigmafac=0):
        Mzi0 = self.Mzi(M0,zi)
        zf = -0.0064*pow(np.log10(M0),2)+0.0237*np.log10(M0)+1.8837
        q = 4.137*pow(zf,-0.9476)
        fMzi = pow(pow(self.sigmaMz(Mzi0/q,zi),2)-pow(self.sigmaMz(Mzi0,zi),2),-0.5)
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)*pow(self.growthD(zi),-2)*self.dDdz(zi)+1)
        beta = -fMzi
        Mzzidef = Mzi0*pow(1+z-zi,alpha)*np.exp(beta*(z-zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef*Msolar,z)
        return (beta+alpha*pow(1+z-zi,-1))*Mzzivir/Msolar

    def dsdm(self, M, z):
        sigma0 = self.Sigma_interp(M)
        dsdM0 = self.dlnSigmadlnM_interp(np.log(M))*2.*sigma0**2/M
        dsdMz = dsdM0*self.growthD(z)
        return dsdMz

    def delc_Y11(self,M, z):
        """ Critical overdensity for collapse for WDM, Benson et al. (2012): Eq.7) """ 
        gx = 1.5
        z_eq = 3600*(OmegaM*pow(h,2)/0.15)-1
        Mj = 3.06*10**8* pow((1+z_eq)/3000,1.5) * pow((OmegaM*pow(h,2)/0.15),0.5)* pow(gx/1.5,-1) * pow(self.mass_wdm,-4) 
        x = np.log(M/Mj)
        hh = 1./(1+np.exp((x+2.4)/0.1))        
        return 1.686*pow(self.growthD(z),-1) * (hh*(0.04/np.exp(2.3*x))+(1-hh)*np.exp(0.31687/np.exp(0.809*x)))

    def s_Y11(self, M):
        return pow(self.sigmaMz(M,0),2)

    def Ffunc_Yang(self, delc1, delc2, sig1, sig2):
        """ Returns Eq. (14) of Yang et al. (2011) """
        return pow(2*np.pi,-0.5)*(delc2-delc1)*pow(sig2-sig1,-1.5) \
            *np.exp(-pow(delc2-delc1,2)*pow(2*(sig2-sig1),-1))

    def Na_calc(self, ma, zacc, Mhost, z0=0, N_herm=200, Nrand=1000, sigmafac=0):
        """ Returns Na, Eq. (3) of Yang et al. (2011) """ 
        zacc_2d = zacc.reshape(np.alen(zacc),1)
        M200_0 = self.Mzzi(Mhost,zacc_2d,z0)
        logM200_0 = np.log10(M200_0)
        if N_herm==1:
            sigmalogM200_0 = 0.12+0.15*np.log10(Mhost/M200_0)
            sigmalogM200_1 = sigmalogM200_0[zacc_2d>1.][0] \
                /np.log10(M200_0[zacc_2d>1.][0]/Mhost) \
                *np.log10(M200_0/Mhost)
            sigmalogM200 = np.where(zacc_2d>1.,sigmalogM200_0,sigmalogM200_1)
            logM200=logM200_0+sigmafac*sigmalogM200
            M200=10**logM200
            if(sigmafac>0.):
                M200 = np.where(M200<Mhost,M200,Mhost)
        else:
            xxi,wwi = hermgauss(N_herm)
            xxi = xxi.reshape(np.alen(xxi),1,1)
            wwi = wwi.reshape(np.alen(wwi),1,1)
            """ eq. (21) in Yang et al. (2011) """ 
            sigmalogM200 = 0.12-0.15*np.log10(M200_0/Mhost)
            logM200 = np.sqrt(2)*sigmalogM200*xxi+logM200_0
            M200 = 10**logM200
        mmax=np.minimum(M200,Mhost/2.0)
        Mmax=np.minimum(M200_0+mmax,Mhost)
        zlist = zacc_2d*np.linspace(1,0,Nrand)
        iMmax = np.argmin(np.abs(self.Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)
        z_Max = zlist[np.arange(np.alen(zlist)),iMmax]
        z_Max_3d = z_Max.reshape(N_herm,np.alen(zlist),1)
        delcM = self.delc_Y11(Mmax,z_Max_3d)
        delca = self.delc_Y11(ma,zacc_2d)
        sM = self.s_Y11(Mmax)
        sa = self.s_Y11(ma)
        xmax = pow((delca-delcM),2)*pow((2*(self.s_Y11(mmax)-sM)),-1)
        normB = special.gamma(0.5)*special.gammainc(0.5,xmax)/np.sqrt(np.pi)
        """ those reside in the exponential part of eq.14 """ 
        Phi = self.Ffunc_Yang(delcM,delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        if N_herm==1:
            F2t = np.nan_to_num(Phi)
            F2=F2t.reshape((len(zacc_2d),len(ma)))
        else:
            F2 = np.sum(np.nan_to_num(Phi)*wwi/np.sqrt(np.pi),axis=0)
        Na = F2*self.dsdm(ma,0)*self.dMdz(Mhost,zacc_2d,z0,sigmafac)*(1+zacc_2d)
        return Na

    ###############################
    # Calculate subhalo properties at accretion and after tidal stripping
    ###############################         

    def rs_rhos_calc(self, M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=100, sigmalogc=0.128,
                     N_herm=5, logmamin=1, logmamax=None, sigmafac=0,
                     N_hermNa=200, profile_change=True):

        zdist = np.arange(redshift+dz,zmax+dz,dz)
        if logmamax==None:
            logmamax = np.log10(0.1*M0)
        ma200 = np.logspace(logmamin,logmamax,N_ma)
        rs_acc = np.zeros((len(zdist),N_herm,len(ma200)))
        rhos_acc = np.zeros((len(zdist),N_herm,len(ma200)))
        rs_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
        rhos_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
        ct_z0 = np.zeros((len(zdist),N_herm,len(ma200)))
        survive=np.zeros((len(zdist),N_herm,len(ma200)))
        m0_matrix = np.zeros((len(zdist),N_herm,len(ma200)))
        Oz_0 = self.Omegaz(pOmega,redshift)

        def Mzvir(z):
            Mz200 = self.Mzzi(M0,z,0)
            if N_hermNa==1:
                logM200_0 = np.log10(Mz200)
                sigmalogM200_0 = 0.12-0.15*np.log10(Mz200/M0)
                Mz1 = self.Mzzi(M0,1.,0.)
                sigma1 = 0.12-0.15*np.log10(Mz1/M0)
                sigmalogM200_1 = sigma1/np.log10(Mz1/M0)*np.log10(Mz200/M0)
                sigmalogM200 = np.where(z>1.,sigmalogM200_0,sigmalogM200_1)
                logM200 = logM200_0+sigmafac*sigmalogM200
                M200 = 10**logM200
                if(sigmafac>0.):
                    M200 = np.where(M200<M0,M200,M0)
                Mz200solar = M200*Msolar
                Mvirsolar = self.Mvir_from_M200(Mz200solar,z)
            else:
                Mz200solar = Mz200*Msolar
                Mvirsolar = self.Mvir_from_M200(Mz200solar,z)
            return Mvirsolar/Msolar

        """ Fitting functions for A, zeta """
        def AMz(z):
            log10a=(-0.0019*np.log10(Mzvir(z))+0.045)*z+(0.0097*np.log10(Mzvir(z))-0.313)
            return pow(10,log10a)
        def zetaMz(z):
            return (-5.55e-5*np.log10(Mzvir(z))+1.43e-03)*z+(3.34e-04*np.log10(Mzvir(z))-8.11e-03)        

        def tdynz(z):
            Oz_z = self.Omegaz(pOmega,z)
            return 1.628*pow(h,-1)*pow(self.Delc(Oz_z-1)/178.0,-0.5)*pow(self.Hz(z)/H0,-1)*(86400*365*(1e+9))

        def msolve(m, z):
            return AMz(z)*(m/tdynz(z))*pow(m/Mzvir(z),zetaMz(z))*pow(self.Hz(z)*(1+z),-1)

        for iz in range(len(zdist)):
            ma = self.Mvir_from_M200(ma200*Msolar,zdist[iz])/Msolar
            Oz = self.Omegaz(pOmega,zdist[iz])
            zcalc = np.linspace(zdist[iz],redshift,100)
            sol = odeint(msolve,ma,zcalc)
            m0 = sol[-1]
            c200sub = self.conc200(ma200*Msolar,zdist[iz])
            rvirsub = pow(3*ma*Msolar*pow(rhocrit0*self.g(zdist[iz]) \
                *self.Delc(Oz-1)*4*np.pi,-1),1.0/3.0)
            r200sub = pow(3*ma200*Msolar*pow(rhocrit0*self.g(zdist[iz]) \
                *200*4*np.pi,-1),1.0/3.0)
            c_mz = c200sub*rvirsub/r200sub
            x1,w1 = hermgauss(N_herm)
            x1 = x1.reshape(np.alen(x1),1)
            w1 = w1.reshape(np.alen(w1),1)
            log10c_sub = np.sqrt(2)*sigmalogc*x1+np.log10(c_mz)
            c_sub = pow(10.0,log10c_sub)
            rs_acc[iz] = rvirsub/c_sub
            rhos_acc[iz] = ma*Msolar/(4*np.pi*rs_acc[iz]**3*self.fc(c_sub))
            if(profile_change==True):
                rmax_acc = rs_acc[iz]*2.163
                Vmax_acc = np.sqrt(rhos_acc[iz]*4*np.pi*G/4.625)*rs_acc[iz]
                Vmax_z0 = Vmax_acc*(pow(2,0.4)*pow(m0/ma,0.3)*pow(1+m0/ma,-0.4))
                rmax_z0 = rmax_acc*(pow(2,-0.3)*pow(m0/ma,0.4)*pow(1+m0/ma,0.3))
                rs_z0[iz] = rmax_z0/2.163
                rhos_z0[iz] = (4.625/(4*np.pi*G))*pow(Vmax_z0/rs_z0[iz],2)
            else:
                rs_z0[iz] = rs_acc[iz]
                rhos_z0[iz] = rhos_acc[iz]
            ctemp = np.linspace(0,100,1000)
            ftemp = interp1d(self.fc(ctemp),ctemp,fill_value='extrapolate')
            ct_z0[iz] = ftemp(m0*Msolar/(4*np.pi*rhos_z0[iz]*rs_z0[iz]**3))
            survive[iz] = np.where(ct_z0[iz]>0.77,1,0)
            m0_matrix[iz] = m0*np.ones((N_herm,1))

        Na = self.Na_calc(ma,zdist,M0,z0=0,N_herm=N_hermNa,Nrand=1000,
                          sigmafac=sigmafac)
        Na_total = integrate.simps(integrate.simps(Na,x=np.log(ma)),x=np.log(1+zdist))
        weight = Na/(1.0+zdist.reshape(np.alen(zdist),1))
        weight = weight/np.sum(weight)*Na_total
        weight = (weight.reshape((len(zdist),1,len(ma))))*w1/np.sqrt(np.pi)
        z_acc = (zdist.reshape(len(zdist),1,1))*np.ones((1,N_herm,N_ma))
        z_acc = z_acc.reshape(len(zdist)*N_herm*N_ma)
        ma200_matrix = ma200*np.ones((len(zdist),N_herm,1))
        ma200_matrix = ma200_matrix.reshape(len(zdist)*N_herm*len(ma200))
        m0_matrix = m0_matrix.reshape(len(zdist)*N_herm*len(ma200))
        rs_acc = rs_acc.reshape(len(zdist)*N_herm*len(ma200))
        rhos_acc = rhos_acc.reshape(len(zdist)*N_herm*len(ma200))
        rs_z0 = rs_z0.reshape(len(zdist)*N_herm*len(ma200))
        rhos_z0 = rhos_z0.reshape(len(zdist)*N_herm*len(ma200))
        ct_z0 = ct_z0.reshape(len(zdist)*N_herm*len(ma200))
        weight = weight.reshape(len(zdist)*N_herm*len(ma200))
        survive = (survive==1).reshape(len(zdist)*N_herm*len(ma200))

        return ma200_matrix, z_acc, rs_acc/kpc, rhos_acc/(Msolar/pc**3), \
            m0_matrix, rs_z0/kpc, rhos_z0/(Msolar/pc**3), ct_z0, \
            weight, survive
            
            




    #################################################################################
    # Subhalo distribution with host halo M0
    # Input: M0 in units [M_solar]. Optional: Distribution at accretion (accretion=True) instead of at present (redshift = 0)
    # Output: subhalo masses in units [M_solar], subhalo distribution dN/dm
    #################################################################################
    def subhalo_distr(self,M0, accretion=False, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5, logmamin=1, logmamax=None, \
        sigmafac=0, N_hermNa=200, profile_change=True):
            
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = self.rs_rhos_calc(M0, redshift, dz, zmax, N_ma, sigmalogc,N_herm, \
            logmamin, logmamax, sigmafac,N_hermNa, profile_change=True)

        if accretion == False:
            N,lnm_edges = np.histogram(np.log(m0),weights=weight,bins=100)
        else:
            N,lnm_edges = np.histogram(np.log(ma200),weights=weight,bins=100)

        lnm = (lnm_edges[1:]+lnm_edges[:-1])/2.
        dlnm = lnm_edges[1:]-lnm_edges[:-1]

        m = np.exp(lnm)
        dNdlnm = N/dlnm
        dNdm   = dNdlnm/m

        return m, dNdm



    #################################################################################
    # Calculate expected number of satellites for given host halo.
    # Input: M0, Mpeak in units [M_solar]
    # Optional: Satellite forming condition with threshold on subhalo peak mass, Mpeak, in units [M_solar] (Mpeak_thres=True)
    # Output: Total number of satellites, subhalo masses [M_solar], cumulative distribution subhalo mass
    #################################################################################
    def N_sat(self,M0, Mpeak=None,Mpeak_thres=False,redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5, logmamin=1, logmamax=None, \
        sigmafac=0, N_hermNa=200, profile_change=True):
            
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = self.rs_rhos_calc(M0, redshift, dz, zmax, N_ma, sigmalogc,N_herm, logmamin, \
            logmamax, sigmafac,N_hermNa, profile_change=True)

        if Mpeak_thres == True:
            N,x_edges = np.histogram(ma200[ma200>Mpeak],weights=weight[ma200>Mpeak],bins=10000)
        else:
            N,x_edges = np.histogram(ma200,weights=weight,bins=10000)

        x = (x_edges[1:]+x_edges[:-1])/2.
        Ncum = np.cumsum(N)
        Ncum = Ncum[-1]-Ncum

        return Ncum[0], x, Ncum


    #################################################################################
    # Calculate expected number of satellites for given host halo with threshold on Vpeak (Vpeak_thres=True) or Vmax (Vpeak_thres=False)
    # Input: M0 in units [M_solar], Vpeak/Vmax in units [km/s]
    # Output: Total number of satellites, subhalo Vmax or Vpeak [km/s], cumulative distribution subhalo Vmax or Vpeak
    #################################################################################
    def N_sat_Vthres(self,M0, Vpeak_max,Vpeak_thres=True,redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128, N_herm=5, logmamin=1, logmamax=None, \
        sigmafac=0, N_hermNa=200, profile_change=True):
            
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive = self.rs_rhos_calc(M0, redshift, dz, zmax, N_ma, sigmalogc,N_herm, logmamin, \
            logmamax, sigmafac,N_hermNa, profile_change=True)

        ma200  *= Msolar
        m0     *= Msolar
        rs_a   *= kpc
        rs0    *= kpc
        rhos_a *= Msolar/pc**3
        rhos0  *= Msolar/pc**3
        rpeak = 2.163*rs_a
        rmax  = 2.163*rs0
        Vpeak = np.sqrt(4.*np.pi*G*rhos_a/4.625)*rs_a
        Vmax  = np.sqrt(4.*np.pi*G*rhos0/4.625)*rs0
        
        if Vpeak_thres == True:
            N,x_edges = np.histogram(Vmax[Vpeak>Vpeak_max*km/s]/(km/s),weights=weight[Vpeak>Vpeak_max*km/s],bins=10000)
        else:
            N,x_edges = np.histogram(Vmax[Vmax>Vpeak_max*km/s]/(km/s),weights=weight[Vmax>Vpeak_max*km/s],bins=10000)

        x = (x_edges[1:]+x_edges[:-1])/2.
        Ncum = np.cumsum(N)
        Ncum = Ncum[-1]-Ncum

        return Ncum[0], x, Ncum


