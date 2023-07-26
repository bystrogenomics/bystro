"""
This implements a version of reduced rank regression in Pytorch. This
implementation has two advantages over the other implementation, (1) it
can use a GPU and (2) it uses stochastic training which is substantially
faster for large datasets (in terms of the number of samples).

Objects
-------

Methods
-------
None
"""
import numpy as np
import numpy.linalg as la
from datetime import datetime as dt
from ._base import BaseReducedRankRegression
import cloudpickle
from sklearn.linear_model import Ridge


class ReducedRankAnalyticNP(BaseReducedRankRegression):
    def __init__(self, L):
        """

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

        model = RRR_dual_tf()
        model.fit(X,Y)
        y_pred = model.predict(X,K=10.0)
        mse = np.mean((y_pred-Y)**2)
        """
        self.L = L
        self.creationDate = dt.now()
        self.fitted = False

    def __repr__(self):
        out_str = "ReducedRankAnalyticNP object\n"
        return out_str

    def fit(self, X, Y):
        """
        Given X and Y, this fits the model

        min ||Y - XB||^2 

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables, should be demeaned

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict, should be demeaned

        loss_function - function(X,X_hat)->tf.Float
            A loss function representing the difference between X 
            and Yhat

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        Returns
        -------
        self
        """
        N = X.shape[0]
        self.p, self.q = X.shape[1], Y.shape[1]
        SigmaXX = 1 / N * np.dot(X.T, X) + 0.00001 * np.eye(self.p)
        SigmaXY = 1 / N * np.dot(X.T, Y)

        SecondHalf = la.solve(SigmaXX, SigmaXY)
        SigmaYZY = np.dot(SigmaXY.T, SecondHalf)
        mod = Ridge()
        mod.fit(X, Y)
        SecondHalf = mod.coef_.T

        Yhat = np.dot(X, SecondHalf)
        U, S, VT = la.svd(Yhat, full_matrices=False)
        VT_sub = VT[: self.L]
        self.B = np.dot(SecondHalf, np.dot(VT_sub.T, VT_sub))

    def unpickle(self, load_name):
        """ 
        Having saved our model parameters using save_model, we can now
        load the parameters into a new object

        Parameters
        ----------
        load_name : str
            The name of the file with saved parameters
        """
        load_dictionary = cloudpickle.load(open(load_name, "rb"))
        self.B = load_dictionary["model"].B
        self.fitted = True

    def _test_inputs(self, X, Y, loss_function):
        """
        This performs error checking on inputs for fit

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables, should be demeaned

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict, should be demeaned

        loss_function - function(X,X_hat)->tf.Float
            A loss function representing the difference between X
            and Yhat
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Samples X != Samples Y")
