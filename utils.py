import numpy as np

def getList(data,var):
    result = []
    for i in range(len(data)):
        result.append(data[i][var])
    return np.array(result) 



def cv_surface(data,wf,method='midpoint'):
    '''
        Calculate quantities at the CV surfaces (front, rear, south, north, and top).
        Needed for the advection, PGF and stress terms of the momentum availability.

        Input:
            - 3D flow field of the CV region.
            - Windfarm layout
            - Surface average method
        
        Two methods to calculate surface averages:
            - Trapzoidal rule
            - Midpoint rule


    '''

    # Define parameters
    HF = wf['HF']  # height of CV
    W = wf['W']   # width  of CV
    L = wf['L']   # length of CV
    
    # Dictionary for storing output
    out = {}

    #Front surface
    dataf = data.isel(x=0)
    if method == 'midpoint':
        datafm = dataf.mean()
    elif method == 'trapz':
        datafm = dataf.integrate(['y','z'])/(HF*W)
    else:
        ValueError("Invalid method. Choose 'midpoint' or 'trapz'.")
    out['front'] = datafm['u'].values*datafm['u'].values*HF*W   # advection integral
    tau11f = -datafm['uu_tot'].values
    out['fronts'] = -tau11f*HF*W  # stress integral; tau11 and dA1 point in opposite direction, so multiply by -1.
    out['frontp'] = datafm['p'].values  # surface-averaged pertubation pressure

    # Rear surface
    datar = data.isel(x=-1)
    if method == 'midpoint':
        datarm = datar.mean()
    elif method == 'trapz':
        datarm = datar.integrate(['y','z'])/(HF*W)
    else:
        ValueError("Invalid method. Choose 'midpoint' or 'trapz'.")
    out['rear'] = -datarm['u'].values*datarm['u'].values*HF*W
    tau11r = -datarm['uu_tot'].values
    out['rears'] = tau11r*HF*W # tau11 and dA1 point in same direction, so multiply by 1.
    out['rearp'] = datarm['p'].values
    
    # South surface
    datas = data.isel(y=0)
    if method == 'midpoint':
        datasm = datas.mean()
    elif method == 'trapz':
        datasm = datas.integrate(["x","z"])/(HF*L)
    else:
        ValueError("Invalid method. Choose 'midpoint' or 'trapz'.")
    out['south'] = datasm['u'].values*datasm['v'].values*HF*L
    tau12s = -datasm['uv_tot'].values
    out['souths'] = -tau12s*HF*L # tau12 and dA2 point in opposite direction, so multiply by -1.
    
    # North surface
    datan = data.isel(y=-1)
    if method == 'midpoint':
        datanm = datan.mean()
    elif method == 'trapz':
        datanm = datan.integrate(["x","z"])/(HF*L)
    else:
        ValueError("Invalid method. Choose 'midpoint' or 'trapz'.")
    out['north'] = -datanm['u'].values*datanm['v'].values*HF*L
    tau12n = -datanm['uv_tot'].values
    out['norths'] = tau12n*HF*L # tau12 and dA2 point in same direction, so multiply by 1.
    
    # Wall shear stress (linear extrapolate from second and third cell values)
    widx1 = 1
    widx2 = 2
    z1 = float(data['z'][widx1]); z2 = float(data['z'][widx2])
    datab1 = data.isel(z=widx1)
    uw1 = float(datab1['uw_tot'].mean())
    datab2 = data.isel(z=widx2)
    uw2 = float(datab2['uw_tot'].mean())
    uw_w = uw1 + (uw2-uw1)/(z2-z1) * (-z1)     
    out['tauw'] = -uw_w
    
    # Top surface
    datat = data.isel(z=-1)
    if method == 'midpoint':
        datatm = datat.mean()
    elif method == 'trapz':
        datatm = datat.integrate(["x","y"])/(L*W)
    else:
        ValueError("Invalid method. Choose 'midpoint' or 'trapz'.")
    out['top'] = -datatm['u'].values*datatm['w'].values*L*W
    tau13t = -datatm['uw_tot'].values
    out['tops'] = tau13t*L*W # tau13 and dA3 point in same direction, so multiply by 1.

    # Add sums to output
    out['total'] = out['front'] + out['rear'] + out['south'] + out['north'] + out['top'] # total advection
    out['totals'] = out['fronts'] + out['rears'] + out['souths'] + out['norths'] + out['tops'] # total stress
    return out


def m_models(data,wf,method='midpoint'):
    '''
        Calculate model values of the momentum availability sub-models:

        Input:

    '''

    # Define parameters
    HF = wf['HF']  # height of CV
    W = wf['W']   # width  of CV
    L = wf['L']   # length of CV
    UF0 = data['UF0']
    beta0 = data['beta0']
    betaL = data['betaL']

    # Output dict
    out = {}

    # Advection terms
    out['adv']['front'] = UF0**2*beta0**2*HF*W
    out['adv']['rear']  = -UF0**2*betaL**2*HF*W
    out['adv']['south'] = 0
    out['adv']['north'] = 0
    out['adv']['top']   = W*HF*UF0**2*0.5*(betaL**2 - beta0**2)
    out['adv']['total'] = 0.5*UF0**2*HF*W*(beta0**2 - betaL**2)

    # PGF
    out['pgf']['deltaPfront'] = 0.5*UF0**2*(1 - beta0**2)
    out['pgf']['total'] = HF*W*2*out['pgf']['deltaPfront']

    return out


def bl_height_5p(z, uw, uw0='None',widx=2):
    '''
        Calculate boundary-layer height as the height where stress profile
        falls to 5% of its surface value.
        
        Use the widx wall-adjacent cell to represent the wall value.
        
    '''
    
    # Take absolute value of uw, so it is always positive
    uwp = np.abs(uw)
    
    # Absolute wall value
    if uw0 == 'None':
        uw_w = uwp[widx]
    else:
        uw_w = uw0
    
    # Target value
    uw_t = 0.05*uw_w
    
    
    # Search from top and down
    idx = len(z)-1
    
    uw_s = np.abs(uw[idx])
    #print('start search')
    while uw_s < uw_t:
        idx = idx - 1
        uw_s = uwp[idx]
    
    
    # Version 1: Just take directly the value at the stopped cell
    #h = z[idx]
    
    # Version 2: Interpolate in the 10 cells around the stopped cell
    uwi = uwp[idx-5:idx+5]
    zi = z[idx-5:idx+5]
    h = np.interp(-uw_t, -uwi, zi) # interp only works on monotonically increasing arrays, hence the minus
    
    
    
    return h#, uw_t, uwi, zi


def power_law_fit_iterative(z, tau, tauw0=1.0):
    '''
        Fit tau/tauw0 = (1 - z/h)^alpha in two-step iterative procedure.
        
        :param z: Height [m]
        :param tau: Shear stress profile [m2/s2]
        :param tauw0: Wall shear stress [m2/s2]
        :return: h, alpha
    '''
    # Import curve_fit
    from scipy.optimize import curve_fit

    # Normalize tau
    tau_norm = tau / tauw0

    # 5% height extrapolated with alpha=1.5 is the first guess of h
    h_05 = bl_height_5p(z, tau, uw0=tauw0)
    h_p = h_05 / (1 - 0.05**(2/3))

    # Define model function with h as a parameter
    def model(z, alpha, h):
        result = (1 - z/h)**(alpha)
        result = np.nan_to_num(result)  # Convert NaNs to 0.0.
        return result
    
    # First iteration: fit alpha with h = h_p
    alpha_init = 1.5
    p0 = [alpha_init]
    # Fit using curve_fit with h_p fixed via lambda
    popt, pcov = curve_fit(lambda z, alpha: model(z, alpha, h_p), z, tau_norm, p0=p0, bounds=([1.0], [3.5]))
    alpha = popt[0]

    # Second iteration: estimate new h_p with fitted alpha and make a new fit for alpha with this
    h_p2 = h_05 / (1 - 0.05**(1/alpha))
    popt, pcov = curve_fit(lambda z, alpha: model(z, alpha, h_p2), z, tau_norm, p0=[alpha], bounds=([1.0], [3.5]))
    alpha2 = popt[0]

    return h_p2, alpha2
