import math
import numpy as np
from scipy import stats
import scipy.integrate as integrate

def c4(kappa):
    integrande0 = lambda t: (1-np.cos(t))*np.exp(kappa*(.5+np.cos(t)))*(np.sin(t/2)**4)*(np.cos(t/2)**2)
    integrandeZ = lambda t: np.exp(kappa*(.5+np.cos(t)))*(np.sin(t/2)**4)*(np.cos(t/2)**2)
    Z = integrate.quad(integrandeZ,0,math.pi)
    I0 = integrate.quad(integrande0,0,math.pi)
    return (1/5)*I0[0]/Z[0]

def test_sisyphe():
    """
    This function creates a system of body-oriented particles in a ``milling configuration''. 
    The test is considered as successful if the computed milling speed is within 
    a 5% relative error range around the theoretical value.     
    """
    print("Welcome! This test function will create a system of body-oriented particles in a ``milling configuration'' (cf. the example gallery). The test will be considered as successful if the computed milling speed is within a 5% relative error range around the theoretical value.")
    
    print("\n Running test, this may take a few minutes...")
    N = 1500000
    L = 1
    R = .025
    nu = 40
    c = 1
    kappa = 10
    
    print("\n Check configuration... ")
    try:
        import pykeops
        pykeops.test_torch_bindings()
    except ImportError:
        print("[SiSyPHE]: pyKeOps not found.")
        return
    except:
        print("[SiSyPHE]: unexpected error.")
        return        
    try:
        import torch
        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        print("Done.")
        if not use_cuda:
            print("[SiSyPHE]: Warning! No GPU detected, the script may be very slow...")
    except ImportError:
        print("[SiSyPHE]: torch not found.")
        return
    except:
        print("[SiSyPHE]: unexpected error.")
        return
    
    print("\n Sample an initial condition... ")
    from sisyphe.initial import cyclotron_twist_z
    pos, bo = cyclotron_twist_z(N,L,1,kappa,dtype)
    print("Done.")        
    
    print("\n Create a model... ")
    import sisyphe.models as models
    simu = models.BOAsynchronousVicsek(pos=pos,bo=bo,
                     v=c,
                     jump_rate=nu,kappa=kappa,
                     interaction_radius=R,
                     box_size=L,
                     boundary_conditions='periodic',
                     variant = {"name" : "normalised", "parameters" : {}},
                     options = {},
                     sampling_method='vonmises',
                     block_sparse_reduction=True,
                     number_of_cells=15**3)
    print("Done.")
    
    print("\n Run the simulation... ")
    from sisyphe.display import save
    frames = [.5]
    data = save(simu,frames,[],["phi"],save_file=False)
    print("Done.")
    
    print("\n Check the result... ")
    try:
        res = stats.linregress(data["time"][1000:], data["phi"][1000:])
        theoretical_value = 2*np.pi*c4(kappa)
        if np.allclose(theoretical_value, -res.slope, rtol=.05):
            print("Done.")
            print("\n SiSyPHE is working!")
        else:
            raise ValueError()
    except ValueError:
            print("\n [SiSyPHE]: wrong result...")
            print("Theoretical result: "+str(theoretical_value))
            print("Computed result: "+str(-res.slope))
            return
    except:             
        print("[SiSyPHE]: unexpected error.")
        return  
        
        
        