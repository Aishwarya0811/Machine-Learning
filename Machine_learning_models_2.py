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
    current = pen*r-2*mod0.logl
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
    orgmodel = model(x,y,*args)
    n = y.shape[0]
    perm = np.random.permutation(n)
    siz = n//k
    outp = np.zeros((k,*stat(orgmodel).shape))
    for i in range(k):
        test = perm[siz*i:siz*(i+1)]
        trainl = perm[:siz*i]
        trainu = perm[siz*(i+1):]
        train = np.hstack((trainl,trainu))
        mod = model(x[train,:],y[train],*args)
        outp[i,:] = stat(mod,x[test,:],y[test])
    return outp

def bootstrap(model,stat,dgp,x,y,reps,*args):
    orgmodel = model(x,y,*args)
    e = y-orgmodel.predict()
    inputmat = np.hstack((e,x))
    (n,r) = inputmat.shape
    outp = np.zeros((reps,*stat(orgmodel).shape))
    for i in range(reps):
        resamp = np.random.randint(n,size=(n,))
        resmat = inputmat[resamp,:]
        newx = resmat[:,1:].reshape(-1,r-1)
        newe = resmat[:,0].reshape(-1,1)
        newy = dgp(newx,newe)
        rsmodel = model(newx,newy,*args)
        outp[i,:] = stat(rsmodel)
    return outp


        
## BIG TASKS PRIORITY
## TODO: Add plots to lm
## TODO: Add rptree with plots at nodes
## TODO: Cut apart and add documentation
## TODO: Add Pandas functionality
## TODO: Add beta distribution class (figure out how to make it a class)
## TODO: Add test/train split
## TODO: Add general classifier class
## TODO: Add missing data functionality to lm
## TODO: Add multivariate normal generator
## TODO: ANOVA testing
## TODO: Poisson regression
## TODO: Add nnets
## TODO: Add panel data functionality
## TODO: Add SVM
## TODO: Add gaussian mixtures
## TODO: Add random forests
## TODO: Add monte carlo helpers
## TODO: Add time series everything
## TODO: Add MAD estimator lm class
## TODO: Kernel density estimation
## TODO: Naive bayes
## TODO: Multinomial logit
## TODO: Ordered logit
## TODO: Order statistics Nonparametrics
## TODO: Gamma least squares

## SMALL TASKS
## TODO: Add boosting class
## TODO: Add bagging class
## TODO: Test probit
## TODO: Test step
## TODO: Add white hc1 2 and 3 corrections
## TODO: Add gradient descent
## TODO: Add Lasso cv plots
## TODO: Add glance to the trees
## TODO: Add tidy for tree nodes p.value at each node
## TODO: Add l1reg tidy/glance etc
## TODO: Add mean absolute error, rmspe, mean sq perc error, rms perc error
## TODO: GLS, Weighted least squares

class mean_predictor:
    def __init__(self,y):
        self.y = y 
        self.n = y.shape[0]
        self.r = y.shape[1]
        self.mean = y.mean(0)
    def predict(self,*args):
        if len(args)==0:
            return self.mean*np.ones((self.n,self.r))
        if len(args)>1:
            raise Exception('predict takes 0 or 1 argument')
        return self.mean*np.ones((args[0],self.r))
        
class lm_noconstant(mean_predictor):
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
        newx = self.__predbuild__(*args)
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
    

class lm(lm_noconstant):
    def __init__(self,x,y):
        (self.n,self.r) = x.shape
        ones = np.ones((self.n,1))
        x = np.hstack((ones,x))
        super(lm,self).__init__(x,y)
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


        
class white(lm):
    def genvariance(self,e):
        meat = np.diagflat(e**2)
        meat = self.x.T.dot(meat).dot(self.x)
        return self.xxi.dot(meat).dot(self.xxi)

class logit(mean_predictor):
    def __init__(self,x,y):
        (self.n,self.y) = (y.shape[0],y)
        ones = np.ones((self.n,1))
        self.x = np.hstack((ones,x))
        self.r = self.x.shape[1]
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
        (bone,_,_,_) = NewtonRaphson(np.zeros((1,1)),jac,hess)
        self.nulllike = self.__likemaker__(ones,bone)[0][0,0]
        self.deviance = 2*(self.logl-self.nulllike)
        self.mcfrsq = 1-self.logl/self.nulllike
        Fhat = Fhat.reshape(-1,1)
        y = y.reshape(-1,1)
        self.blrsq = np.mean(np.multiply(y,Fhat)+np.multiply(1-y,1-Fhat))
        delta = self.n/2/self.nulllike
        self.vzrsq = self.mcfrsq*(delta-1)/(delta-self.mcfrsq)
        self.efrsq = 1-((y-Fhat)**2).sum()/((y-y.mean())**2).sum()
        num = (np.dot(self.x-self.x.mean(0),self.b)**2).sum()
        self.mzrsq = num/(self.n+num)
    def __likemaker__(self,x,b):
        (logL,dlogL,ddlogL) = (0,0,0)
        for i in range(self.n):
            xcur = x[i,:].reshape(-1,1)
            inner = xcur.T.dot(b)
            Fx = logistic.cdf(inner)
            logL += self.y[i]*np.log(Fx)+(1-self.y[i])*np.log(1-Fx)
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
            logL += self.y[i]*np.log(Fx)+(1-self.y[i])*np.log(1-Fx)
            dlogL += etax*(self.y[i]-Fx)*xcur
            ddlogL += (-etax*fx+(self.y[i]-Fx)*detax)*(xcur.dot(xcur.T))
        return(logL,dlogL,ddlogL)
    
class l1reg(mean_predictor):
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
            mspe = [kfold(lassosimple,lassosimple.mspe,x,y,5,threshmax*i/100).mean()]
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
    def isterm(self):
        if self.lchild and self.rchild: return False
        return True
        
class rptree(bintree):
    def __init__(self,x,y,level='',parent=None,maxlevs=None,test=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (self.x,self.y,self.level,self.n) = (x,y,level,y.shape[0])
            super(rptree,self).__init__(level,parent)
            (self.svar,self.split) = self.getsplit()
            xtmp = (x[:,self.svar]<=self.split).astype(int).reshape(-1,1)
            if maxlevs and len(level)+1>maxlevs: return 
            try:
                self.pvalue = lm(xtmp,y).pval[1][0]
            except:
                self.pvalue = 0.5
            if test and self.pvalue>=0.05: return
            (lft,rght) = (x[:,self.svar]<=self.split,x[:,self.svar]>self.split)
            self.lchild = rptree(x[lft,:],y[lft],level+'L',parent=self,maxlevs=maxlevs,test=test)
            self.rchild = rptree(x[rght,:],y[rght],level+'R',parent=self,maxlevs=maxlevs,test=test)
    
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
            if not RSS:
                splits += [None]
                RSSes += [1e16]
                continue
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
    def predict(self,*args):
        if len(args)==0:
            newx = self.x
        else:
            newx = args[0]
        outp = []
        for row in range(newx.shape[0]):
            outp += [self.simple_predict(newx[row,:])]
        return(np.array(outp).reshape(-1,1))
    def simple_predict(self,newx):
        if self.isterm(): return self.y.mean()
        if newx[self.svar]<=self.split:
            return self.lchild.simple_predict(newx)
        else:
            return self.rchild.simple_predict(newx)
        
        
        
class pca:
    def __init__(self,x,scale=False,maxr=None):
        if not maxr:
            maxr = min(x.shape[1]-1,10)
        self.maxr = maxr
        self.x = x
        (self.n,self.r) = x.shape
        self.z = x - x.mean(0)
        if scale: self.z = np.divide(self.z,self.z.std(0))
        self.vcov = np.dot(self.z.T,self.z)/(self.n-1)
        (self.eign,self.rotation) = np.linalg.eigh(self.vcov)
        (self.eign,self.rotation) = self.flip_eigen(self.eign,self.rotation)
        (self.eratio,self.gratio) = self.setup_eratios(self.eign,self.rotation)
        self.pc = np.dot(self.z,self.rotation)
    def setr(self,val):
        self.r = val
        self.rotation = self.rotation[:,:self.r]
        self.pc = self.pc[:,:self.r]
    def setup_eratios(self,eign,rotn):
        eign = np.vstack((0,eign))
        vars = eign.sum()-np.cumsum(eign)
        vars = vars.reshape(-1,1)
        eign[0,0] = eign.sum()/np.log(self.n)
        eratio = eign[:-1]/eign[1:]
        eratio = eratio[:self.maxr]
        gratio = np.log(1+np.divide(eign,vars))
        gratio = gratio[:self.maxr]
        return(eratio,gratio)
    def flip_eigen(self,eign,rotn):
        eign = eign.reshape(-1,1)
        eign = np.flipud(eign)
        rotn = np.fliplr(rotn)
        rotn = rotn[:,:self.maxr]
        return(eign,rotn)
        
class kmeans_simple_dumb:
    def __init__(self,x,k,maxiters):
        self.x = x
        self.k = k
        n = x.shape[0]
        self.n = n
        self.clusters = np.random.randint(k,size=(n,))
        for i in range(maxiters):
            self.means = self.get_means(x,self.clusters,k)
            distz = self.get_dists(x,self.means,k)
            oldclu = self.clusters
            self.clusters = distz.argmin(1)
            if np.power(oldclu-self.clusters,2).sum() < 1: break
        self.iterations = i
        self.wss = distz.min(1).sum()
    def get_means(self,x,clusters,k):
        meanz = []
        for i in range(k):
            meanz += [x[clusters==i,:].mean(0)]
        meanz = np.matrix(meanz)
        return meanz
    def get_dists(self,x,means,k):
        distz = np.zeros((x.shape[0],k))
        for i in range(k):
            diffs = x-means[i,:]
            distz[:,i] = np.sqrt(np.power(diffs,2).sum(1)).reshape(-1)
        return distz
class kmeans_simple(kmeans_simple_dumb):
    def __init__(self,x,k,nstart,maxiters):
        super(kmeans_simple,self).__init__(x,k,maxiters)
        results = []
        for i in range(nstart):
            results += [kmeans_simple_dumb(x,k,maxiters)]
        counts = np.zeros((nstart,))
        for i in range(nstart):
            mat = results[i].means.copy()
            mat.sort()
            for j in range(nstart):
                newmat = results[j].means.copy()
                newmat.sort()
                diffs = mat-newmat
                if np.power(diffs,2).sum() <= 1e-2:
                    counts[i] += 1
        result = results[counts.argmax()]
        self.means = result.means
        self.clusters = result.clusters
        self.wss = result.wss

class kmeans(kmeans_simple):
    def __init__(self,x,maxk=10,nstart=10,maxiters=100):
        self.x = x
        self.n = x.shape[0]
        self.r = x.shape[1]
        self.models = {}
        self.wss = {}
        for i in range(1,maxk):
            self.models[i] = kmeans_simple(x,i,nstart,maxiters)
            self.wss[i] = self.models[i].wss
    def setk(self,k):
        self.k = k
        self.model = self.models[k]
        self.means = self.model.means
        self.clusters = self.model.clusters
