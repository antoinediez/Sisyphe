import math
import torch
from .sampling import vonmises_quat_rand, quat_mult
# from display import bo_display


def cyclotron_twist_z(N,L,k,kappa,dtype):
    pos = torch.tensor(L).type(dtype)*torch.rand(N,3).type(dtype)
    
    if type(L)==float or type(L)==int:
        qtarget = torch.cat((
            torch.cos((2*math.pi*k/L)*pos[:,2]/2).reshape(N,1),
            torch.sin((2*math.pi*k/L)*pos[:,2]/2).reshape(N,1),
            torch.zeros(N,1).type(dtype),
            torch.zeros(N,1).type(dtype)),dim=1).type(dtype)
    else:
        qtarget = torch.cat((
            torch.cos((2*math.pi*k/L[2])*pos[:,2]/2).reshape(N,1),
            torch.sin((2*math.pi*k/L[2])*pos[:,2]/2).reshape(N,1),
            torch.zeros(N,1).type(dtype),
            torch.zeros(N,1).type(dtype)),dim=1).type(dtype)    
    bo = vonmises_quat_rand(qtarget, torch.tensor([kappa]).type(dtype))
    
    return pos, bo


def inverted_cyclotron_twist_z(N,L,k,kappa,dtype):
    pos = torch.tensor(L).type(dtype)*torch.rand(N,3).type(dtype)
    bo = torch.zeros(N,4).type(dtype)
   
    if type(L)==float or type(L)==int:
        N1 = torch.sum(pos[:,2]<L/2.).int()
        N2 = torch.sum(pos[:,2]>=L/2.).int()
        qtarget1 = torch.cat((
            torch.cos((2*math.pi*k/L)*pos[pos[:,2]<L/2.,2]/2).reshape(N1,1),
            torch.sin((2*math.pi*k/L)*pos[pos[:,2]<L/2.,2]/2).reshape(N1,1),
            torch.zeros(N1,1).type(dtype),
            torch.zeros(N1,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,2]<L/2.,:] = vonmises_quat_rand(qtarget1, torch.tensor([kappa]).type(dtype))
       
        qtarget2 = torch.cat((
            torch.cos((2*math.pi*k/L)*pos[pos[:,2]>=L/2.,2]/2).reshape(N2,1),
            -torch.sin((2*math.pi*k/L)*pos[pos[:,2]>=L/2.,2]/2).reshape(N2,1),
            torch.zeros(N2,1).type(dtype),
            torch.zeros(N2,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,2]>=L/2.,:] = vonmises_quat_rand(qtarget2, torch.tensor([kappa]).type(dtype))
    else:
        N1 = torch.sum(pos[:,2]<L[2]/2.).int()
        N2 = torch.sum(pos[:,2]>=L[2]/2.).int()
        qtarget1 = torch.cat((
            torch.cos((2*math.pi*k/L[2])*pos[pos[:,2]<L[2]/2.,2]/2).reshape(N1,1),
            torch.sin((2*math.pi*k/L[2])*pos[pos[:,2]<L[2]/2.,2]/2).reshape(N1,1),
            torch.zeros(N1,1).type(dtype),
            torch.zeros(N1,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,2]<L[2]/2.,:] = vonmises_quat_rand(qtarget1, torch.tensor([kappa]).type(dtype))
        
        qtarget2 = torch.cat((
            torch.cos((2*math.pi*k/L[2])*pos[pos[:,2]>=L[2]/2.,2]/2).reshape(N2,1),
            -torch.sin((2*math.pi*k/L[2])*pos[pos[:,2]>=L[2]/2.,2]/2).reshape(N2,1),
            torch.zeros(N2,1).type(dtype),
            torch.zeros(N2,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,2]>=L[2]/2.,:] = vonmises_quat_rand(qtarget2, torch.tensor([kappa]).type(dtype))
           
    return pos, bo

def helical_twist_x(N,L,k,kappa,dtype):
    pos = torch.tensor(L).type(dtype)*torch.rand(N,3).type(dtype)
    
    if type(L)==float or type(L)==int:
        qtarget = torch.cat((
            torch.cos((2*math.pi*k/L)*pos[:,0]/2).reshape(N,1),
            torch.sin((2*math.pi*k/L)*pos[:,0]/2).reshape(N,1),
            torch.zeros(N,1).type(dtype),
            torch.zeros(N,1).type(dtype)),dim=1).type(dtype)
    else:
        qtarget = torch.cat((
            torch.cos((2*math.pi*k/L[0])*pos[:,0]/2).reshape(N,1),
            torch.sin((2*math.pi*k/L[0])*pos[:,0]/2).reshape(N,1),
            torch.zeros(N,1).type(dtype),
            torch.zeros(N,1).type(dtype)),dim=1).type(dtype)    
    bo = vonmises_quat_rand(qtarget, torch.tensor([kappa]).type(dtype))
    
    return pos, bo


def inverted_helical_twist_x(N,L,k,kappa,dtype):
    pos = torch.tensor(L).type(dtype)*torch.rand(N,3).type(dtype)
    bo = torch.zeros(N,4).type(dtype)
    
    if type(L)==float or type(L)==int:
        N1 = torch.sum(pos[:,0]<L/2.).int()
        N2 = torch.sum(pos[:,0]>=L/2.).int()
        qtarget1 = torch.cat((
            torch.cos((2*math.pi*k/L)*pos[pos[:,0]<L/2.,0]/2).reshape(N1,1),
            torch.sin((2*math.pi*k/L)*pos[pos[:,0]<L/2.,0]/2).reshape(N1,1),
            torch.zeros(N1,1).type(dtype),
            torch.zeros(N1,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,0]<L/2.,:] = vonmises_quat_rand(qtarget1, torch.tensor([kappa]).type(dtype))
       
        qtarget2 = torch.cat((
            torch.cos((2*math.pi*k/L)*pos[pos[:,0]>=L/2.,0]/2).reshape(N2,1),
            -torch.sin((2*math.pi*k/L)*pos[pos[:,0]>=L/2.,0]/2).reshape(N2,1),
            torch.zeros(N2,1).type(dtype),
            torch.zeros(N2,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,0]>=L/2.,:] = vonmises_quat_rand(qtarget2, torch.tensor([kappa]).type(dtype))
    else:
        N1 = torch.sum(pos[:,0]<L[0]/2.).int()
        N2 = torch.sum(pos[:,0]>=L[0]/2.).int()
        qtarget1 = torch.cat((
            torch.cos((2*math.pi*k/L[0])*pos[pos[:,0]<L[0]/2.,0]/2).reshape(N1,1),
            torch.sin((2*math.pi*k/L[0])*pos[pos[:,0]<L[0]/2.,0]/2).reshape(N1,1),
            torch.zeros(N1,1).type(dtype),
            torch.zeros(N1,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,0]<L[0]/2.,:] = vonmises_quat_rand(qtarget1, torch.tensor([kappa]).type(dtype))
        
        qtarget2 = torch.cat((
            torch.cos((2*math.pi*k/L[0])*pos[pos[:,0]>=L[0]/2.,0]/2).reshape(N2,1),
            -torch.sin((2*math.pi*k/L[0])*pos[pos[:,0]>=L[0]/2.,0]/2).reshape(N2,1),
            torch.zeros(N2,1).type(dtype),
            torch.zeros(N2,1).type(dtype)),dim=1).type(dtype)
        bo[pos[:,0]>=L[0]/2.,:] = vonmises_quat_rand(qtarget2, torch.tensor([kappa]).type(dtype))
           
    return pos, bo