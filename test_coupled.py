"""Verification suite for coupled_model.py.  Run:  python test_coupled.py
Exits 0 if all checks pass, 1 otherwise."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from coupled_model import (NutrientField, deposit, interp, min_image, CoupledSim,
                           long_range_screened_vec, long_range_lattice,
                           short_range_celllist, R_c, R_unif, aggregation_index)
import hybrid_model as hm

results = []
def check(name, cond, detail=""):
    results.append(cond)
    print(f"  [{'PASS' if cond else '**FAIL**'}] {name}  {detail}")

print("="*74); print("T1  Field solver: manufactured solution, order in space and time"); print("="*74)
D, al, TP = 0.1, 0.5, 2*np.pi
cstar = lambda x,y,t: np.exp(-t)*(1+0.5*np.sin(TP*x)*np.cos(TP*y))
def src(x,y,t):
    g=np.sin(TP*x)*np.cos(TP*y); return np.exp(-t)*((al-1.0)*(1+0.5*g)+D*TP**2*g)
def solve(M,dt,T):
    h=1.0/M; xs=np.arange(M)*h; X,Y=np.meshgrid(xs,xs,indexing="ij")
    f=NutrientField(M,1.0,D,al,0.0,0.0,cstar(X,Y,0.0)); n=int(round(T/dt))
    for k in range(n):
        f.S=src(X,Y,(k+1)*dt); f.step(dt,np.zeros((M,M)))
    return np.sqrt(np.mean((f.rho-cstar(X,Y,n*dt))**2))
Ms=[16,32,64,128]; es=[solve(M,2e-4,0.02) for M in Ms]
ps=np.polyfit(np.log(1.0/np.array(Ms)),np.log(es),1)[0]
check("spatial order ~2", 1.8<ps<2.2, f"fitted {ps:.3f}")
dts=[0.02,0.01,0.005,0.0025]; et=[solve(128,d,0.1) for d in dts]
pt=np.polyfit(np.log(dts),np.log(et),1)[0]
check("temporal order ~1", 0.85<pt<1.2, f"fitted {pt:.3f}")

print(); print("="*74); print("T2  Particle<->grid transfer"); print("="*74)
rng=np.random.default_rng(0); L=1.0
for M in (32,64,128):
    U=deposit(rng.random((500,2))*L,L,M)
    check(f"int U dx = 1 (M={M})", abs(U.sum()*(L/M)**2-1)<1e-12, f"{U.sum()*(L/M)**2:.12f}")
errs=[]
for M in (32,64,128,256):
    xs=np.arange(M)*L/M; GX,GY=np.meshgrid(xs,xs,indexing="ij")
    G=np.sin(2*np.pi*GX)*np.cos(2*np.pi*GY); Xp=rng.random((400,2))*L
    errs.append(np.sqrt(np.mean((interp(G,Xp,L,M)-np.sin(2*np.pi*Xp[:,0])*np.cos(2*np.pi*Xp[:,1]))**2)))
pi=np.polyfit(np.log(1.0/np.array([32,64,128,256])),np.log(errs),1)[0]
check("interpolation order ~2", 1.7<pi<2.3, f"fitted {pi:.3f}")

print(); print("="*74); print("T3  Screened kernel: is minimum image controlled?"); print("="*74)
X=rng.random((400,2))*L; el,reg=0.02,0.03
for kL,tol in ((0.0,None),(5.0,None),(8.66,5e-3),(20.0,1e-4)):
    rel=np.linalg.norm(long_range_screened_vec(X,L,el,reg,kL)-long_range_lattice(X,L,el,reg,kL,4))
    rel/=np.linalg.norm(long_range_lattice(X,L,el,reg,kL,4))
    if tol is None: print(f"      kappa*L={kL:5.2f}: {rel:.3e}  (reference)")
    else: check(f"kappa*L={kL:.2f} controlled", rel<tol, f"{rel:.2e} < {tol:.0e}")
F0=long_range_screened_vec(X,L,el,reg,0.0)
check("kappa=0 reproduces the submitted kernel exactly",
      np.allclose(F0,hm.long_range_direct_vec(X,L,el,reg),rtol=0,atol=1e-15))

print(); print("="*74); print("T4  Short-range regularisation (Assumption 1)"); print("="*74)
Xs=rng.random((300,2))*L
Fr=short_range_celllist(Xs,L,12.0,0.06,reg_s=3e-4)
F0=short_range_celllist(Xs,L,12.0,0.06,reg_s=0.0)
check("reg_s=0 reproduces the submitted short-range force",
      np.allclose(F0,hm.short_range_celllist(Xs,L,12.0,0.06)))
check("reg_s=3e-4 within 1% of unregularised", np.isfinite(Fr).all() and
      np.linalg.norm(Fr-F0)/max(np.linalg.norm(F0),1e-30)<0.01,
      f"{np.linalg.norm(Fr-F0)/max(np.linalg.norm(F0),1e-30):.2e}")

print(); print("="*74); print("T5  Coupling is live"); print("="*74)
sim=CoupledSim(N=600,M=64,seed=1,dt=0.02,beta=4.0,S=1.0,source="uniform")
r0=sim.field.rho.mean(); sim.X[:,0]=sim.rng.random(sim.N)*0.25*sim.L
for _ in range(120): sim.step()
rho,M=sim.field.rho,sim.M
check("uptake depletes the field where particles are",
      rho[:M//4,:].mean()<rho[M//2:3*M//4,:].mean()*0.98,
      f"{rho[:M//4,:].mean():.4f} < {rho[M//2:3*M//4,:].mean():.4f}")
check("field is dynamic, not static", abs(rho.mean()-r0)>1e-6, f"{r0:.5f} -> {rho.mean():.5f}")
check("rho stays non-negative", (rho>=-1e-12).all(), f"min {rho.min():.2e}")

print(); print("="*74); print("T6  MECHANISM: field-mediated coupling is chemo-REPULSIVE"); print("="*74)
U=deposit(sim.X,sim.L,sim.M); dn=U-U.mean(); dr=rho-rho.mean()
corr=float(np.sum(dn*dr)/np.sqrt(np.sum(dn**2)*np.sum(dr**2)))
check("corr(delta_n, delta_rho) < 0", corr<-0.2, f"corr = {corr:+.3f}")

print(); print("="*74); print("T7  Field solver stability under strong concentrated uptake"); print("="*74)
Uc=np.zeros((64,64)); Uc[30:34,30:34]=200.0
for beta,dt in ((1.0,0.02),(10.0,0.02),(50.0,0.05)):
    f=NutrientField(64,1.0,0.02,0.5,beta,1.0,1.0)
    for _ in range(200): f.step(dt,Uc)
    check(f"stable & positive at dt*beta*maxU = {dt*beta*Uc.max():.0f}",
          np.isfinite(f.rho).all() and f.rho.min()>=-1e-12, f"min rho {f.rho.min():.3e}")

print(); print("="*74); print("T8  Diagnostics"); print("="*74)
print(f"      R_unif(L=1) = {R_unif(1.0):.5f}")
A0=aggregation_index(CoupledSim(N=400,M=64,seed=3))
check("aggregation index ~1 on a Poisson configuration", 0.8<A0<1.2, f"A = {A0:.3f}")

print(); print("="*74); print(f"  {sum(results)}/{len(results)} checks passed"); print("="*74)
sys.exit(0 if all(results) else 1)
