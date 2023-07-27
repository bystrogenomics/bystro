"""
This provides the base class of any reduced rank regression model, whether 
it be vanilla or some of the penalized extensions. This only creates the 
base methods that can be used independent of inference strategy. 

Implementing an extension model requires that the following methods be 
implemented
    fit - Learns the model given data 

Objects
-------
BaseReducedRankRegression()
    Base class

Methods
-------
None
"""
import numpy as np
from numpy import linalg as la
from copy import deepcopy
from datetime import datetime as dt
from datetime import timezone
import cloudpickle

from bystro.tadarrr._template_sgd_np import _BaseSGDModel


class BaseReducedRankRegression(object):
    def get_nuclear_norm(self):
        """
        Returns the nuclear norm of the coefficients, a convex relaxation 
        of the ``rank''.

        Parameters
        ----------
        None

        Returns
        -------
        nuclear_norm : float
            The nuclear norm
        """
        nuclear_norm = la.norm(self.B, ord="nuc")
        return nuclear_norm

    def predict(self, X):
        """
        Predicts Y given learned coefficients B

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables

        Returns
        -------
        Y_hat : np.array-like,shape=(N,q)
            The predicted values
        """
        Y_hat = np.dot(X, self.B)
        return Y_hat

    def predict_subset(self, X, idxs):
        """
        Predicts Y given learned coefficients B

        Parameters
        ----------
        X : np.array-like,shape=(N,sum(idxs))
            The subset predictor variables

        idxs : np.array-like,shape=(self.p,)
            The covariates to consider

        Returns
        -------
        Y_hat : np.array-like,shape=(N,q)
            The predicted values
        """
        B_sub = self.B[idxs == 1]
        Y_hat = np.dot(X, B_sub)
        return Y_hat

    def mse(self, X, Y):
        """
        Evaluates the MSE of the predictions made by reduced rank 
        regression

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict

        Returns
        -------
        Y_hat : np.array-like,shape=(N,q)
            The predicted values
        """
        Y_hat = self.predict(X)
        Ysq = np.square(Y - Y_hat)
        Y_sum = np.sum(Ysq, axis=0)
        mse = np.mean(Y_sum)
        return mse

    def pickle(self, fileName):
        """
        Saves a (trained) model

        Parameters
        ----------
        fileName : string
            What the file name you save it to is

        Returns
        -------
        None
        """
        myDict = {"model": self}
        with open(fileName, "wb") as f:
            cloudpickle.dump(myDict, f)

    def save_coefficients_csv(self, fileName):
        """
        Saves the coefficient matrix as csv files. 

        Parameters
        ----------
        fileName : string
            Saving the file as a csv.

        Returns
        -------
        None
        """
        if self.fitted:
            np.savetxt(fileName + "_B.csv", self.B, fmt="%0.8f", delimiter=",")
        else:
            print("Model not fitted")


class BaseReducedRankRegressionSGD(BaseReducedRankRegression, _BaseSGDModel):
    def __init__(self, training_options={}):
        self.training_options = self._fill_training_options(training_options)

        self.creationDate = dt.now(tz=timezone.utc)

    def _initialize_losses(self):
        """
        This initializes the arrays to store losses

        Attributes
        ----------
        losses : np.array,size=(td['n_iterations'],)
            Total loss including regularization terms

        losses_recon : np.array,size=(td['n_iterations'],)
            Prediction loss

        losses_reg : np.array,size=(td['n_iterations'],)
            Regularization
        """
        n_iterations = self.training_options["n_iterations"]
        self.losses = np.zeros(n_iterations)
        self.losses_recon = np.zeros(n_iterations)
        self.losses_reg = np.zeros(n_iterations)

    def _fill_training_options(self, training_options):
        """
        """
        default_dict = {
            "learning_rate": 1e-2,
            "method": "Nadam",
            "momentum": 0.9,
            "decay_options": {
                "decay": "exponential",
                "steps": 500,
                "rate": 0.96,
                "staircase": True,
            },
            "gpu_memory": 1024,
            "n_iterations": 5000,
            "batch_size": 512,
            "adaptive": False,
        }
        return fill_dict(training_options, default_dict)


def fill_dict(mydict, default_dict):
    """
    For any key in default_dict that does not appear in mydict add the key

    Parameters
    ----------
    mydict :
        Dictionary to modify

    default_dict:
        Dictionary with default answers

    Returns
    -------
    mydict : dictionary
        The dictionary with added keys
    """
    z = deepcopy(default_dict)
    z.update(mydict)
    return z


def simple_batcher_xy(batchSize, X, Y):
    N = X.shape[0]
    rng = np.random.default_rng()
    idx = rng.choice(N, batchSize, replace=False)
    X_batch = X[idx]
    Y_batch = Y[idx]
    return X_batch, Y_batch
