import math

PEAK1 = 209.5e12      # RTX 5090 dense bf16 w/ fp32 accum
NG    = 4
PEAK  = PEAK1*NG

def C(mfu, hours): return PEAK*mfu*hours*3600.0

print("="*70); print("1. COMPUTE BUDGET  (4x5090, peak %.0f TFLOPS)"%(PEAK/1e12))
for h in (3.0,4.0,5.0,6.0):
    row=[f"{C(m,h):.3e}" for m in (0.20,0.25,0.30,0.35)]
    print(f"  {h:>4.1f}h  MFU20/25/30/35: {row}")

print("\n2. COMPUTE-OPTIMAL N*  (C=6ND, r=D/N, N*=sqrt(C/6r))")
def Nstar(Cv,r): return math.sqrt(Cv/(6.0*r))
for Cv,lab in ((C(.25,4.0),"C@25%,4h"),(C(.30,4.0),"C@30%,4h"),(C(.30,5.0),"C@30%,5h"),(C(.35,5.0),"C@35%,5h")):
    print(f"  {lab:<12} C={Cv:.3e}  " + "  ".join(f"r={r}:{Nstar(Cv,r)/1e6:6.1f}M" for r in (40,30,20,10,5,2,1)))

print("\n   inverse: what r makes N=X optimal at C@30%,5h = %.2e"%C(.30,5.0))
Cv=C(.30,5.0)
for N in (150e6,350e6,600e6,1e9):
    print(f"     N={N/1e6:6.0f}M -> r = C/(6N^2) = {Cv/(6*N*N):6.3f} tok/param")

print("\n3. CHINCHILLA LOSS (Hoffmann E1.69 A406.4 B410.7 a.34 b.28) at C@30%,5h  [ILLUSTRATIVE]")
E,A,B,al,be=1.69,406.4,410.7,0.34,0.28
def L(N,D): return E+A/N**al+B/D**be
best=None
for N in [50e6,100e6,159e6,250e6,350e6,500e6,700e6,1e9]:
    D=Cv/(6*N); l=L(N,D)
    print(f"   N={N/1e6:6.0f}M D={D/1e9:6.2f}B D/N={D/N:7.2f}  L={l:.4f}")
    if best is None or l<best[1]: best=(N,l)
print(f"   min over grid: N={best[0]/1e6:.0f}M L={best[1]:.4f}")
# fine optimum
xs=[(L(N,Cv/(6*N)),N) for N in [1e6*k for k in range(50,1001,1)]]
lo=min(xs); print(f"   fine optimum N*={lo[1]/1e6:.1f}M  L={lo[0]:.4f}   L(1B)-L(N*)={L(1e9,Cv/6e9)-lo[0]:+.4f} nats")

print("\n4. E-CONFOUND: measured local slope = alpha*(1-E/L)")
for Ev in (1.69,1.82):
    for Lv in (2.6,3.0,3.4,3.8):
        att=1-Ev/Lv
        print(f"   E={Ev} L={Lv}: attenuation={att:.3f} -> measured/alpha={att*100:5.1f}%")

print("\n5. ADVANTAGE-GROWTH statistic (E cancels if E shared across arch)")
print("   Delta(N)=L_sub-L_ref = A_s/N^as - A_r/N^ar ; d(Delta)/dlnN = -(as*(Ls-E) - ar*(Lr-E))")
for LmE in (1.0,1.3,1.7):
    for dal in (0.01,0.02,0.05):
        print(f"   L-E={LmE}: d_alpha={dal:.2f} -> dDelta/dlnN = {dal*LmE:.4f} nats/e-fold")
