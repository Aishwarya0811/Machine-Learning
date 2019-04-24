import numpy as np
import pandas as pd
from scipy.stats import norm, logistic, t
import cvxopt
from cvxopt import matrix
from cvxopt.solvers import qp
import warnings
from graphviz import Digraph

cvxopt.solvers.options['show_progress'] = False

def NewtonRaphson(xinit,J,H,reps=1000,tol=1e-16):
    x = xinit
    for i in range(reps):
        upd = np.linalg.solve(H(x),J(x))
        x -= upd
        if np.power(upd,2).sum()<tol: return(x,J(x),H(x),i)
    raise Exception('Newton did not converge')

def step(model,x,y,bic=False,*args):
    (n,r) = x.shape
    mod0 = model(x,y,*args)
    if bic: pen=np.log(n)
    else:   pen=2
    current = 2*r-2*mod0.logl
    ics = []
    for i in range(r):
        if   i==0: newx = x[:,1:]
        elif i==r: newx = x[:,:-1]
        else     : newx = np.hstack((x[:,:i],x[:,i+1:]))
        mod = model(newx,y,*args)
        ics += [2*(r-1)-2*mod.logl]
    ics = np.array(ics)
    if ics.min()>=current:
        return mod0
    i = ics.argmin()
    if i==0: newx = x[:,1:]
    elif i==r: newx = x[:,:-1]
    else: newx = np.hstack((x[:,:i],x[:,i+1:]))
    return step(model,newx,y,*args)

def mspe(model,xtest,ytest):
    err = ytest - model.predict(xtest)
    return np.array((err**2).mean())

def rmse(model,xtest,ytest):
    return np.sqrt(mspe(model,xtest,ytest))

def cmat(model,xtest,ytest):
    v1 = np.hstack((1-self.predict(xtest),self.predict(xtest)))
    v2 = np.hstack((1-ytest,ytest))
    return np.dot(v1.T,v2)

def precision(model,xtest,ytest):
    mat = cmat(model,xtest,ytest)
    ans = np.array(mat[1,1]/(mat[1,1]+mat[1,0]))
    return np.array(0) if ans.isnan() else ans
    
def recall(model,xtest,ytest):
    mat = cmat(model,xtest,ytest)
    ans = np.array(mat[1,1]/(mat[1,1]+mat[0,1]))
    return np.array(0) if ans.isnan() else ans
    
def accuracy(model,xtest,ytest):
    mat = cmat(model,xtest,ytest)
    ans = np.array((mat[1,1]+mat[0,0])/mat.sum())
    return np.array(0) if ans.isnan() else ans

def F1(model,xtest,ytest):
    prec = model.precision(xtest,ytest)
    recl = model.recall(xtest,ytest)
    return np.array(2*prec*recl/(prec+recl))

def kfold(model,stat,x,y,k,*args):
    n = y.shape[0]
    perm = np.random.permutation(n)
    siz = n//k
    outp = 0
    for i in range(k):
        test = perm[siz*i:siz*(i+1)]
        trainl = perm[:siz*i]
        trainu = perm[siz*(i+1):]
        train = np.hstack((trainl,trainu))
        mod = model(x[train,:],y[train],*args)
        outp += stat(mod,x[test,:],y[test])
    return outp/k
class lm0:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        (self.n,self.r) = x.shape
        xx = np.dot(x.T,x)
        xy = np.dot(x.T,y)
        self.xxi = np.linalg.inv(xx)
        self.b = np.linalg.solve(xx,xy).reshape(-1,1)
        e = y - np.dot(x,self.b)
        self.resid = e
        self.vb = self.genvariance(e)
        self.se = np.sqrt(np.diagonal(self.vb)).reshape(-1,1)
        self.tstat = np.divide(self.b,self.se)
        self.pval = 2*t.cdf(-np.abs(self.tstat),df=self.n-self.r)
        self.rsq = 1-e.var()/y.var()
        self.adjrsq = 1-(1-self.rsq)*(self.n-1)/(self.n-self.r)
        self.logl = -self.n/2*(np.log(2*np.pi*e.var())+1)
        self.aic = 2*self.r-2*self.logl
        self.bic = np.log(self.n)*self.r-2*self.logl
        nulllike = -self.n/2*(np.log(2*np.pi*y.var())+1)
        self.deviance = 2*(self.logl-nulllike)
    def genvariance(self,e):
        return e.var()*self.xxi
    def predict(self,*args):
        newx = self.__predbuild__(self,*args)
        return np.dot(newx,self.b)
    def __predbuild__(self,*args):
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            newx = args[0]
        return newx
    def tidy(self,confint=False,conflevel=0.95):
        if not confint:
            df = [self.b,self.se,self.tstat,self.pval]
        else:
            df = [self.b,self.se,self.tstat,self.pval,\
                  self.b+self.se*t.ppf((1-conflevel)/2,df=self.n-self.r),\
                  self.b-self.se*t.ppf((1-conflevel)/2,df=self.n-self.r)]
        df = [x.reshape(-1,1) for x in df]
        df = np.hstack(df)
        if not confint:
            df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val'])
        else:
            df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val','lower','upper'])
        return df
    def glance(self):
        df = pd.DataFrame(columns=['r.squared','adj.rsq','r','logl',\
                                   'aic','bic','deviance','df'])
        df.loc[0] = [self.rsq,self.adjrsq,self.r,self.logl,self.aic,\
                     self.bic,self.deviance,self.n-self.r]
        return df
    

class lm(lm0):
    def __init__(self,x,y):
        (self.n,self.r) = x.shape
        ones = np.ones((self.n,1))
        x = np.hstack((ones,x))
        super(lm,self).__init__(x,y)
    def __predbuild__(self,*args):
        newx = self.__predbuild__(self,*args)
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            newx = args[0]
            m = newx.shape[0]
            ones = np.ones((m,1))
            newx = np.hstack((ones,newx))
        return newx


        
class white(lm):
    def genvariance(self,e):
        meat = np.diagflat(e**2)
        meat = self.x.T.dot(meat).dot(self.x)
        return self.xxi.dot(meat).dot(self.xxi)

class logit:
    def __init__(self,x,y):
        (self.n,self.y) = (y.shape[0],y)
        ones = np.ones((n,1))
        self.x = np.hstack((ones,x))
        self.r = x.shape[1]
        jac = lambda b: self.__likemaker__(self.x,b)[1]
        hess = lambda b: self.__likemaker__(self.x,b)[2]
        (b,_,H,_) = NewtonRaphson(np.zeros((self.r,1)),jac,hess)
        self.b = b.reshape(-1,1)
        self.vb = -np.linalg.inv(H)
        Fhat = self.predict()
        e = self.y.reshape(-1,1) - Fhat.reshape(-1,1)
        self.resid = e
        self.se = np.sqrt(np.diagonal(self.vb)).reshape(-1,1)
        self.tstat = np.divide(self.b,self.se)
        self.pval = 2*t.cdf(-np.abs(self.tstat),df=self.n-self.r)
        self.logl = self.__likemaker__(self.x,self.b)[0][0,0]
        self.aic = 2*self.r-2*self.logl
        self.bic = np.log(self.n)*self.r-2*self.logl
        jac = lambda b: self.__likemaker__(ones,b)[1]
        hess = lambda b: self.__likemaker__(ones,b)[2]
        (bone,_,_,_) = NewtonRhapson(np.zeros((1,1)),jac,hess)
        self.nulllike = self.__likemaker__(ones,bone)[0][0,0]
        self.deviance = 2*(self.logl-self.nulllike)
        self.mcfrsq = 1-self.logl/self.nulllike
        self.blrsq = 0
        self.vzrsq = 0
        self.efrsq = 0
        self.mzrsq = 0
    def __likemaker__(self,x,b):
        (logL,dlogL,ddlogL) = (0,0,0)
        for i in range(self.n):
            xcur = x[i,:].reshape(-1,1)
            inner = xcur.T.dot(b)
            Fx = logistic.cdf(inner)
            logL += self.y[i]*np.log(Fx)+(1-y[i])*np.log(1-Fx)
            dlogL += (self.y[i]-Fx)*xcur
            ddlogL -= logistic.pdf(inner)*(xcur.dot(xcur.T))
        return(logL,dlogL,ddlogL)
    def __predbuild__(self,*args):
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            newx = args[0]
            m = newx.shape[0]
            ones = np.ones((m,1))
            newx = np.hstack((ones,newx))
        return newx
    def predict(self,*args):
        newx = self.__predbuild__(*args)
        return logistic.cdf(np.dot(newx,self.b))
    def tidy(self,confint=False,conflevel=0.95):
        if not confint:
            df = [self.b,self.se,self.tstat,self.pval]
        else:
            df = [self.b,self.se,self.tstat,self.pval,\
                  self.b+self.se*t.ppf((1-conflevel)/2,df=self.n-self.r),\
                  self.b-self.se*t.ppf((1-conflevel)/2,df=self.n-self.r)]
        df = [x.reshape(-1,1) for x in df]
        df = np.hstack(df)
        if not confint:
            df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val'])
        else:
            df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val','lower','upper'])
        return df
    def glance(self):
        df = pd.DataFrame(columns=['mcfadden.rsq','r','logl',\
                                   'aic','bic','deviance','df',\
                                   'bl.rsq','vz.rsq','ef.rsq','mz.rsq'])
        df.loc[0] = [self.mcfrsq,self.r,self.logl,self.aic,\
                     self.bic,self.deviance,self.n-self.r,\
                    self.blrsq,self.vzrsq,self.efrsq,self.mzrsq]
        return df

class probit(logit):
    def __likemaker__(self,x,b):
        (logL,dlogL,ddlogL) = (0,0,0)
        for i in range(self.n):
            xcur = x[i,:].reshape(-1,1)
            inner = xcur.T.dot(b)
            Fx = norm.cdf(inner)
            fx = norm.pdf(inner)
            etax = fx/Fx/(1-Fx)
            detax = -inner*etax-(1-2*Fx)*etax**2
            logL += self.y[i]*np.log(Fx)+(1-y[i])*np.log(1-Fx)
            dlogL += etax*(self.y[i]-Fx)*xcur
            ddlogL += (-etax*fx*+(self.y[i]-Fx)*detax)*(xcur.dot(xcur.T))
        return(logL,dlogL,ddlogL)
    
class l1reg:
    def __init__(self,x,y,thresh):
        dy = y - y.mean()
        dx = x - x.mean(0)
        b = self.lassosolve(dx,dy,thresh)
        b0 = y.mean()-x.mean(0).dot(b)
        b = np.vstack((b0,b))
        (self.n,self.r) = x.shape
        ones = np.ones((self.n,1))
        self.x = np.hstack((ones,x))
        self.r += 1
        self.y = y
        self.b = b
    def lassosolve(self,x,y,thresh):
        (n,r) = x.shape
        P = x.T.dot(x)
        q = x.T.dot(y)
        A = np.matrix([[1,-1],[-1,1]])
        P = np.kron(A,P)
        A = np.matrix([[1],[-1]])
        q = np.kron(A,q)
        G = -np.eye(2*r)
        A = np.ones((1,2*r))
        G = np.vstack((G,A))
        h = np.zeros((2*r,1))
        h = np.vstack((h,thresh))
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        b = np.matrix(qp(P,q,G,h)['x'])
        b = b[:r,0] - b[r:,0]
        return b
    def __predbuild__(self,*args):
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            newx = args[0]
            m = newx.shape[0]
            ones = np.ones((m,1))
            newx = np.hstack((ones,newx))
        return newx
    def predict(self,*args):
        newx = self.__predbuild__(*args)
        return np.dot(newx,self.b)


class l1regcv(l1reg):
    def __init__(self,x,y):
        threshmax = np.abs(lm(x,y).b[1:]).sum()
        self.threshmax = threshmax
        mspe = []
        for i in range(0,101):
            mspe = [kfold(lassosimple,lassosimple.mspe,x,y,5,threshmax*i/100)[0,0]]
        self.thresh = np.array(mspe).argmin()/100*threshmax
        super(l1regcv,self).__init__(x,y,self.thresh)

class bintree:
    def __init__(self,*args):
        if len(args)>=3: 
            raise Exception('bintree takes 0, 1, or 2 arguments')
        self.name   = args[0] if len(args)>=1 else None
        self.parent = args[1] if len(args)>=2 else None
        self.lchild = None
        self.rchild = None
        
class rptree(bintree):
    def __init__(self,x,y,level='',parent=None,maxlevs=None,test=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (self.x,self.y,self.level,self.n) = (x,y,level,y.shape[0])
            super(rptree,self).__init__(level,parent)
            if maxlevs is not None and maxlevs <= len(level)+1: 
                (self.var,self.split,self.pvalue) = (None,None,None)
                return
            (self.svar,self.split) = self.getsplit()
            if test:
                xtmp = (x[:,self.svar]<=self.split).astype(int).reshape(-1,1)
                self.pvalue = lm(xtmp,y).pval[1][0]
                if self.pvalue>=0.05: return
            else:
                self.pvalue = None
            (lft,rght) = (x[:,self.svar]<=self.split,x[:,self.svar]>self.split)
            self.lchild = rptree(x[lft,:],y[lft],level+'L',parent=self,maxlevs=maxlevs,test=test)
            self.rchild = rptree(x[rght,:],y[rght],level+'R',parent=self,maxlevs=maxlevs,test=test)
    def isterm(self):
        if self.lchild is not None and self.rchild is not None: return True
        return False
    def getsplit(self):
        (x,y) = (self.x,self.y)
        splits = []
        RSSes = []
        for i in range(x.shape[1]):
            xuse = x[:,i]
            RSS = []
            for item in np.unique(xuse):
                y1 = y[xuse<=item]
                y2 = y[xuse>item]
                v1 = y1.var()*len(y1)
                v2 = y2.var()*len(y2)
                if np.isnan(v2): v2 = 0
                RSS += [v1+v2]
            splitrow = np.array(RSS).argmin()
            splits += [np.unique(xuse)[splitrow]]
            RSSes += [RSS[splitrow]]
        rselect = np.array(RSSes).argmin()
        split = splits[rselect]
        return (rselect,split)
    def plot(self,dot=Digraph()):
        if not self.isterm():
            pval = np.round(self.pvalue,3)
            if pval==0:
                dot.node(self.level,'Split: '+str(self.svar)+'\np<0.001')
            else:
                dot.node(self.level,'Split: '+str(self.svar)+'\np='+str(pval))
            dot.node(self.level+'L')
            dot.node(self.level+'R')
            dot.edge(self.level,self.level+'L','<='+str(self.split))
            dot.edge(self.level,self.level+'R','>'+str(self.split))
            self.lchild.plot(dot)
            self.rchild.plot(dot)
        else:
            self.plot_term(dot)
        return dot
    def plot_term(self,dot):
        dot.node(self.level,"E[y|X]="+str(np.round(self.y.mean(),3))+"\nn="+str(self.n),shape='box')
    def __str__(self,outstr=''):
        outstr += self.level + '; '
        if not self.isterm():
            outstr += 'Split: '+str(self.svar)
            if self.pvalue is not None: 
                pval = np.round(self.pvalue,3)
                outstr += '; p<0.001' if pval==0 else '; p='+str(pval)
            outstr += '\n'
            outstr += str(self.lchild)
            outstr += str(self.rchild)
        else:
            outstr += "E[y|X]="+str(np.round(self.y.mean(),3))+"; n="+str(self.n)+'\n'
        return outstr
