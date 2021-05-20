# by Saeed Shaker

# This is a direct translation of Yuval Tassa's Matlab code into Python:
# https://benjaminmoll.com/wp-content/uploads/2020/06/LCP.m
# It solves LCP using a Newton type method
# To be consistent across platforms and with Yuval Tassa's code,
# I have tried to make as minimal changes as I could, 
# so this code can be followed the same way as the original Matlab code does.

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

def LCP_python(M,q,l=[],u=[],x0=[],display=False):   
    tol            = 1.0e-8;
    mu             = 1e-3;
    mu_step        = 5;
    mu_min         = 1e-5;
    max_iter       = 25;
    b_tol          = 1e-6;

    n              = M.shape[0]

    if l == []:
        l = np.zeros((n,1))
    if u == []:
        u = np.ones((n,1))*np.inf
    if x0 == []:
        x00 = np.maximum(np.ones((n,1)),l)
        x0 = np.minimum(x00,u)
    
    M = sparse.csc_matrix(M)
    q = q.reshape((-1, 1))
    l = l.reshape((-1, 1))
    u = u.reshape((-1, 1))
    x0 = x0.reshape((-1, 1))
                         
    lu             = np.column_stack((l , u));
    x              = x0.copy();
    psi,phi,J    = FB(x,q,M,l,u);
    new_x          = True

    for iter1 in range(0,max_iter):
        if new_x:
            mlu      = np.min(np.column_stack((np.abs(x-l),np.abs(u-x))),1).reshape((-1, 1));
            ilu      = np.argmin(np.column_stack((np.abs(x-l),np.abs(u-x))),1).reshape((-1, 1));
            bad            = np.maximum(np.abs(phi),mlu) < b_tol;
            psi            = psi - 0.5*np.dot(phi[bad] , phi[bad])
            notbad = bad == False
            Jind = np.dot(notbad , notbad.T)
            notbad_trues = np.sum(notbad*1)
            J              = sparse.csc_matrix(np.reshape(J[Jind] , (notbad_trues,notbad_trues) ))
            phi            = phi[notbad]; 
            new_x          = False;
            nx             = x.copy();
            nx[bad]        = lu.flatten()[(bad[bad])*1+(ilu[bad]-1)*n]
        H              = np.dot(J.T , J) + mu*sparse.eye(notbad_trues);
        Jphi           = sparse.csc_matrix.dot(J.T,phi)
        d              = -spsolve(sparse.csc_matrix(H) , Jphi)
        nx[notbad]       = x[notbad] + d;
        npsi,nphi,nJ = FB(nx,q,M,l,u);
        
        r   = (psi - npsi)/ -(np.dot(Jphi.T,d) + 0.5*np.dot(sparse.csc_matrix.dot(d.T,H),d) );  # actual reduction / expected reduction

        if r < 0.3:
            mu = np.maximum(mu*mu_step,mu_min);
        if r > 0:
            x     = nx.copy();
            psi   = npsi.copy();
            phi   = nphi.copy();
            J     = nJ.copy();
            new_x = True;
            if r > 0.8: 
                mu = mu/mu_step * (mu > mu_min);
        if display:
            print('iter = ', iter1 , ' --- psi = ' , psi ,' --- r = ' , r ,' --- mu = ' , mu);
        if psi < tol:
            break;
    x = np.minimum(np.maximum(x,l),u);    
    return x

#----------------------------------------------------------

def FB(x,q,M,l,u):
    n     = x.size;
    Zl    = ((l >-np.inf) & (u==np.inf))
    Zu    = (l==-np.inf) & (u <np.inf);
    Zlu   = (l >-np.inf) & (u <np.inf);
    Zf    = (l==-np.inf) & (u==np.inf);

    a     = x.copy();
    b     = sparse.csc_matrix.dot(M,x)+q;
    a[Zl] = x[Zl]-l[Zl];
    a[Zu] = u[Zu]-x[Zu];
    b[Zu] = -b[Zu];

    if any(Zlu):
        nt     = np.sum(Zlu);
        at     = u[Zlu]-x[Zlu];
        bt     = -b[Zlu];
        st     = np.sqrt(np.power(at,2) + np.power(bt,2));
        a[Zlu] = x[Zlu]-l[Zlu];
        b[Zlu] = st -at -bt;
    
    s        = np.sqrt(np.power(a,2) + np.power(b,2));
    phi      = s - a - b;
    phi[Zu]  = -phi[Zu];
    phi[Zf]  = -b[Zf];

    psi      = 0.5*np.dot(phi.T , phi);

    if any(Zlu):
        M[Zlu,:] = -sparse.csc_matrix((at/st-np.ones((nt,1)),(np.arange(nt),Zlu[Zlu != 0])),nt,n , dtype=np.float) - sparse.csc_matrix.dot(sparse.csc_matrix((bt/st-np.ones((nt,1)),(np.arange(nt),np.arange(nt))) , dtype=np.float), M[Zlu,:]);
   
    da       = (a/s-np.ones((n,1))).reshape((-1, 1));
    db       = (b/s-np.ones((n,1))).reshape((-1, 1));
    da[Zf]   = 0;
    db[Zf]   = -1;   
    J        =  sparse.csc_matrix((np.array(da[:,0]),(np.arange(n),np.arange(n))), dtype=np.float) + sparse.csc_matrix.dot(sparse.csc_matrix((np.array(db[:,0]),(np.arange(n),np.arange(n))) , dtype=np.float) , M);
    
    return psi,phi,J