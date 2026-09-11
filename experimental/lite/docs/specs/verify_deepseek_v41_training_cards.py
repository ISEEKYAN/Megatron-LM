import math

def close(a,b,tol=1e-12):
 assert abs(a-b)<=tol+tol*abs(b),(a,b)
def fd(f,x,want):
 for h in (1e-5,5e-6):
  for i,g in enumerate(want):
   a=x.copy();b=x.copy();a[i]+=h;b[i]-=h
   close((f(a)-f(b))/(2*h),g,1e-8)
fd(lambda x:3*(2*x[0])+5*(2*x[0]),[3.],[16.])
fd(lambda x:2*(x[0]*x[2]+x[1]*x[3]),[2.,5.,.25,.75],[.5,1.5,4.,10.])
def pool(x):
 p=[math.exp(z) for z in x[2:]]
 return sum(a*b for a,b in zip(x[:2],p))/sum(p)
fd(pool,[2.,6.,0.,0.],[.5,.5,-1.,1.])
def attn(x):
 q,k1,k2,s=x; e=[math.exp(q*k1),math.exp(q*k2),math.exp(s)]
 return (e[0]*k1+e[1]*k2)/sum(e)
fd(attn,[0.,2.,4.,0.],[8/3,1/3,1/3,-2/3])
fd(lambda x: math.log(sum(math.exp(z) for z in x))-x[0],[0.,0.],[-.5,.5])
fd(lambda x:(2*x[0]+12*x[0])/4,[1.],[3.5])
fd(lambda x:5*x[0]+2*x[1],[3.,7.],[5.,2.])
fd(lambda x:15*x[0],[2.],[15.])
close(5*3+2*7,29)
# T9 is a declared VJP, deliberately not finite-differenced.
close(2*.25,.5); close(2*1,2)
W=[[1.,1.],[1.,1.]];M=[[0.,0.],[0.,0.]];G=[[1.,-1.],[-1.,1.]]
for step in range(2):
 M=[[.95*M[i][j]+.05*(G[i][j] if step==0 else 0) for j in range(2)] for i in range(2)]
 N=[[.95*M[i][j]+.05*(G[i][j] if step==0 else 0) for j in range(2)] for i in range(2)]
 close(M[0][0],[.05,.0475][step]);close(N[0][0],[.0975,.045125][step])
 norms=[math.sqrt(sum(v*v for v in row)) for row in N]
 U=[[v if norms[i]>.001*sum(norms)/2 else 0 for v in row] for i,row in enumerate(N)]
 for k in range(11):
  if k%2==0:
   U=[[v/(math.sqrt(sum(t*t for t in row))+1e-20) for v in row] for row in U]
  else:
   den=[math.sqrt(sum(U[i][j]**2 for i in range(2)))+1e-20 for j in range(2)]
   U=[[U[i][j]/den[j] for j in range(2)] for i in range(2)]
 W=[[W[i][j]-.18*math.sqrt(2)*U[i][j] for j in range(2)] for i in range(2)]
for i in range(2):
 for j in range(2):close(W[i][j],1-.36*G[i][j])
m=.1*4;v=.05*16
close((1-.01*.1)*2-.01*(m/.1)/(math.sqrt(v/.05)+1e-20),1.988)
print('8 smooth cards: both finite-difference steps passed; STE declared VJP checked; Sinkhorn two-step and AdamW vectors passed')
