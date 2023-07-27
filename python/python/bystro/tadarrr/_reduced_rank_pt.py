"""
This implements a version of reduced rank regression in Pytorch. This
implementation has two advantages over the other implementation, (1) it
can use a GPU and (2) it uses stochastic training which is substantially
faster for large datasets (in terms of the number of samples).

Objects
-------
RRRDualPT(mu=1.0,training_options={})
    min ||Y - XB||^2 + mu*norm(B)_nuc

SparseLowRankRegressionPT(mul=1.0,mus=1.0,training_options={})
        This decomposes regression into a low rank and sparse component
        min ||Y - XB||^2 + mu*norm(L)_nuc + mus*norm(S)_1
        s.t B = L + S

Methods
-------
None
"""
from tqdm import trange
import numpy as np
import torch
import cloudpickle
from torch import nn

from bystro.tadarrr._base import BaseReducedRankRegressionSGD, simple_batcher_xy
from bystro.tadarrr._reduced_rank_np import ReducedRankAnalyticNP


class ReducedRankPT(BaseReducedRankRegressionSGD):
    def __init__(self, mu=1.0, training_options={}):
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

        model = RRR_dual_tf()
        model.fit(X,Y)
        y_pred = model.predict(X,K=10.0)
        mse = np.mean((y_pred-Y)**2)
        """
        self.mu = float(mu)
        super.__int__(training_options=training_options)

    def __repr__(self):
        out_str = "RRRDualPT object\n"
        return out_str

    def fit(self, X, Y, loss_function=nn.MSELoss(), progress_bar=True):
        """
        Given X and Y, this fits the model

        min ||Y - XB||^2 + mu*norm(B)_1 

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
        td = self.training_options
        self._test_inputs(X, Y, loss_function)
        X, Y = self._transform_training_data(X, Y)

        # Declare our variables
        mod_init = ReducedRankAnalyticNP(int(np.minimum(X.shape[0], 5)))
        mod_init.fit(X, Y)
        B_ = torch.tensor(mod_init.B, requires_grad=True)
        trainable_variables = [B_]

        myrange = trange if progress_bar else range

        self._initialize_losses()

        optimizer = torch.optim.SGD(
            trainable_variables, lr=td["learning_rate"], momentum=0.9
        )

        for i in myrange(td["n_iterations"]):
            X_batch, Y_batch = simple_batcher_xy(td["batch_size"], X, Y)

            Y_recon = torch.matmul(X_batch, B_)
            loss_recon = loss_function(Y_batch, Y_recon)

            # Regularization losses
            loss_reg = torch.linalg.matrix_norm(B_, ord="nuc")
            loss = loss_recon + self.mu * loss_reg / self.q

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, loss, loss_recon, loss_reg)

        loss_reg = torch.linalg.matrix_norm(B_, ord="nuc")
        Y_recon = torch.matmul(X, B_)
        loss_recon = loss_function(Y, Y_recon)
        loss = loss_recon + self.mu * loss_reg

        self.optimal_value = loss.detach().numpy()
        self.optimal_loss = loss_recon.detach().numpy()

        self._save_variables(trainable_variables)
        return self

    def unpickle(self, load_name):
        """ 
        Having saved our model parameters using save_model, we can now
        load the parameters into a new object

        Parameters
        ----------
        load_name : str
            The name of the file with saved parameters
        """
        with open(load_name, "rb") as f:
            load_dictionary = cloudpickle.load(f)
        self.B = load_dictionary["model"].B

    def _save_variables(self, training_variables):
        """
        This saves the final parameter values after training

        Parameters
        ----------
        training_variables :list,len=1
            The single trained variable 

        Attributes
        ----------
        B : np.array,(q,r)
            The regression coefficients
        """
        self.B = training_variables[0].detach().numpy()

    def _save_losses(self, i, loss, loss_recon, loss_reg):
        """
        This saves the respective losses at each iteration

        Parameters
        ----------
        i : int 
            The current iteration

        loss : tf.Float
            The total loss minimized by optimizer

        loss_recon : tf.Float
            The reconstruction loss

        loss_reg : tf.Foat
            The nuclear norm loss
        """
        self.losses[i] = loss.detach().numpy()
        self.losses_recon[i] = loss_recon.detach().numpy()
        self.losses_reg[i] = loss_reg.detach().numpy()

    def _test_inputs(self, X, Y):
        """
        This performs error checking on inputs for fit

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables, should be demeaned

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict, should be demeaned
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Samples X != Samples Y")

    def _transform_training_data(self, X, Y):
        """
        This converts training data to adequate format

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The predictor variables, should be demeaned

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict, should be demeaned

        Returns
        -------
        X : np.array-like,shape=(N,p)
            The predictor variables, should be demeaned

        Y : np.array-like,shape=(N,q)
            The variables we wish to predict, should be demeaned
        """
        self.n_samples, self.p = X.shape
        self.q = Y.shape[1]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y
