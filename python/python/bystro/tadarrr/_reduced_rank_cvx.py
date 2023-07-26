"""
This implements vanilla reduced rank regression and its dual objective 
using cvxpy.

Objects
-------
RRRPrimalCvx(K=1.0):
    min ||Y-XB||^2
    norm(B)_nuc \le K

RRRDualCvx(mu=1.0)
    min ||Y - XB||^2 + mu*norm(B)_nuc

Methods
-------
None
"""
import numpy as np
import cvxpy as cp
from datetime import datetime as dt
import time
from ..utils._misc import fill_dict, pretty_string_dict
from ._base import BaseReducedRankRegression

__version__ = "1.0.0"


class RRRPrimalCvx(BaseReducedRankRegression):
    def __init__(self, K=1.0):
        """
        This implements a version of reduced rank regression. Given 
        covariates $X\in\mathbb{R}^{Nxp}$ where $N$ is the number of 
        observations and $p$ is the number of predictors, we wish to 
        predict $Y\in\mathbb{R}^{Nxq}$. One of the simplest models is 
        linear regression where we model $Y=XB+E$ where 
        $B\in\mathbb{R}^{pxq}$ is a matrix of coefficients and $E$ is the 
        error matrix. This can be solved as 

        min ||Y-XB||^2
        B

        which has an analytic solution $(X^TX)^{-1}X^TY$. This is just the 
        multivariate least squares solution. The problem is that if $p$ and
        $q$ are relatively large we have an enormous number of parameters 
        $#=p*q$. Reduced rank regression can substantially reduce the number
        of parameters. We let $B=UV$ where $U\in\mathbb{R}^{pxr}$ and 
        $V\in\mathbb{R}^{rxq}$. If $r<<min(p,q)$ this will result in a 
        substantially reduced number of parameters which will have 
        substantially better predictive properties. The objective then 
        becomes

        min ||Y-XUV||^2
        U,V

        Unfortunately, this optimization problem is not convex because the 
        domain is not a convex set. We can turn this into a convex problem 
        by replacing the rank constraint in to a constraint on something 
        called a nuclear norm, defined as 

        ||X||_nuc = \sum s_i(X)

        Where $s_i(X)$ are the singular values of X. The objective then 
        becomes

        min ||Y-XB||^2
        norm(B)_nuc \le K

        We then minimize this objective using cvxpy, which solves convex 
        optimization problems in an efficient manner.

        Attributes
        ----------
        K : float,default=1.0
            The constraint on the norm of the coefficients.

        Usage
        -----
        N = 10000
        p,q,R = 30,5,2
        sigma = 1.0
        U = rand.randn(p,R)
        V = rand.randn(R,q)

        B = np.dot(U,V)
        X = rand.randn(N,p)
        Y_hat = np.dot(X,B)
        Y = Y_hat + sigma*rand.randn(N,q)

        model = RRR_primal_cvx(K=10.0)
        model.fit(X,Y,verbose=False)
        y_pred = model.predict(X,K=10.0)
        """
        self.K = float(K)

        self.creationDate = dt.now()
        self.fitted = False
        self.version = __version__

    def __repr__(self):
        out_str = "RRRPrimalCvx object\n"
        out_str += "K=%0.3f\n" % self.K
        return out_str

    def fit(self, X, Y, options={"verbose": False}):
        """
        Given X and Y, this fits the model

        min ||Y - XB||^2
        s.t norm(B)_nuc le K

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict

        options : dict
            verbose - bool,default=False
                Whether you want to print out fitting information
        """
        N, p = X.shape
        n, q = Y.shape
        if N != n:
            raise ValueError("Dimensions of X and Y do not match")
        self.p, self.q = p, q

        # Declare our variables
        B = cp.Variable((p, q))

        # This is the primal objective cost
        cost = cp.sum_squares(Y - X @ B) / N

        # These are the constraints on the decomposition variables
        low_rank = cp.normNuc(B)
        constraint_list = [low_rank <= self.K * p]

        # Actually declare the convex optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraint_list)

        # Solve the problem
        start_time = time.time()
        prob.solve(verbose=options["verbose"])
        self.elapsed_time = time.time() - start_time

        # Save the optimal value and the
        self.optimal_value = prob.value
        self.B = B.value
        self.fitted = True
        return self


class RRRDualCvx(BaseReducedRankRegression):
    def __init__(self, mu=1.0):
        """
        This replaces the primal optimization problem (constrained 
        optimization problem) of RRR_primal_cvx. That is the objective 
        becomes 

        min ||Y - XB||^2 + mu*norm(B)_nuc

        This can be derived as taking the Lagrangian of the primal problem 
        (replace constraint with Lagrange multipliers, then noting that 
        the K*mu term is a constant). This is still a convex problem. So
        instead of a constrained optimization problem, we now have a 
        penalization scheme with a tuning parameter.

        Attributes
        ----------
        mu : float,default=1.0
            The penalization strength

        Usage
        -----
        N = 10000
        p,q,R = 30,5,2
        sigma = 1.0
        U = rand.randn(p,R)
        V = rand.randn(R,q)

        B = np.dot(U,V)
        X = rand.randn(N,p)
        Y_hat = np.dot(X,B)
        Y = Y_hat + sigma*rand.randn(N,q)

        model = RRR_dual_cvx()
        model.fit(X,Y,verbose=False)
        y_pred = model.predict(X,K=10.0)
        mse = np.mean((y_pred-Y)**2)
        """
        self.mu = float(mu)

        self.creationDate = dt.now()
        self.fitted = False

    def __repr__(self):
        out_str = "RRR_dual_cvx object\n"
        out_str += "mu=%0.3f\n" % self.mu
        return out_str

    def fit(self, X, Y, options={"verbose": False}):
        """
        Given X and Y, this fits the model

        min ||Y - XB||^2 + mu*norm(B)_nuc

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict

        options : dict
            verbose - bool,default=False
                Whether you want to print out fitting information
        """
        N, p = X.shape
        n, q = Y.shape
        if N != n:
            raise ValueError("Samples in X and Y do not match")

        # Declare our variables
        B = cp.Variable((p, q))

        # This is the primal objective cost
        cost = cp.sum_squares(Y - X @ B) / N

        # These are the penalization terms rather than constraints
        low_rank = self.mu * cp.normNuc(B)

        cost_reg = cost + low_rank

        # Actually declare the convex optimization problem
        prob = cp.Problem(cp.Minimize(cost_reg))

        # Solve the problem
        start_time = time.time()
        prob.solve(verbose=options["verbose"])
        self.elapsed_time = time.time() - start_time

        # Save the optimal value and the
        self.optimal_value = prob.value
        self.B = B.value
        self.fitted = True
        return self
