"""
This implements a version of reduced rank regression in Tensorflow. This
implementation has two advantages over the other implementation, (1) it
can use a GPU and (2) it uses stochastic training which is substantially
faster for large datasets (in terms of the number of samples).

Objects
-------
RRRDualTf(mu=1.0,training_options={})
    min ||Y - XB||^2 + mu*norm(B)_nuc

SparseLowRankRegressionTF(mul=1.0,mus=1.0,training_options={})
        This decomposes regression into a low rank and sparse component
        min ||Y - XB||^2 + mu*norm(L)_nuc + mus*norm(S)_1
        s.t B = L + S

Methods
-------
None

"""
from tqdm import trange
import numpy as np
import tensorflow as tf
from bystro.tadarrr._base import BaseReducedRankRegressionSGD, simple_batcher_xy
from tensorflow import keras
import cloudpickle


def loss_norm_lq_tf(X, Xhat=None, q=2, weights=None, safe=True):
    """
    Computes the Lq loss (omitting 1/q)

    Parameters
    ----------
    X : tf.array-like(p,r)
        Matrix of data

    Xhat : tf.array-like(p,r)
        Optional matrix of predictions (default=0)

    q : float>0
        The power

    weights : array,default=None
        The weights to place on each observation

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    loss : tf.Float
        The loss
    """
    if safe:
        X = X.astype(np.float32)

    if Xhat is None:
        Xhat = 0

    if len(X.shape) > 1:
        if q == 1:
            loss_un = tf.reduce_mean(tf.abs(X - Xhat), axis=1)
        elif q == 2:
            loss_un = tf.reduce_mean(tf.square(X - Xhat), axis=1)
        else:
            loss_un = tf.reduce_mean(tf.math.pow(tf.abs(X - Xhat), q), axis=1)
    else:
        if q == 1:
            loss_un = tf.abs(X - Xhat)
        elif q == 2:
            loss_un = tf.square(X - Xhat)
        else:
            loss_un = tf.math.pow(tf.abs(X - Xhat), q)

    loss = (
        tf.reduce_mean(loss_un)
        if weights is None
        else tf.reduce_mean(weights * loss_un)
    )

    return loss


class RRRDualTf(BaseReducedRankRegressionSGD):
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
        super().__init__(training_options=training_options)

    def __repr__(self):
        out_str = "RRRDualTf object\n"
        return out_str

    def fit(self, X, Y, loss_function=loss_norm_lq_tf, progress_bar=True):
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
        limit_gpu(td["gpu_memory"])
        self._test_inputs(X, Y, loss_function)
        X, Y = self._transform_training_data(X, Y)

        # Declare our variables
        rng = np.random.default_rng()
        p, q = X.shape[1], Y.shape[1]
        B_ = tf.Variable(rng.normal(size=(p, q)).astype(np.float32))
        trainable_variables = [B_]

        self._initialize_losses()

        myrange = trange if progress_bar else range

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        for i in myrange(td["n_iterations"]):
            X_batch, Y_batch = simple_batcher_xy(td["batch_size"], X, Y)

            with tf.GradientTape() as tape:
                Y_recon = tf.matmul(X_batch, B_)
                loss_recon = loss_function(Y_batch, Y_recon)
                loss_recon = tf.reduce_mean(tf.square(Y_recon - Y_batch))

                # Regularization losses
                loss_reg = loss_norm_nuclear_tf(B_)
                loss = loss_recon + self.mu * loss_reg

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, loss, loss_recon, loss_reg)

        loss_reg = loss_norm_nuclear_tf(B_)
        Y_recon = tf.matmul(X, B_)
        loss_recon = loss_function(Y, Y_recon)
        loss = loss_recon + self.mu * loss_reg

        self.optimal_value = loss.numpy()
        self.optimal_loss = loss_recon.numpy()

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
        self.B = training_variables[0].numpy()

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
        self.losses[i] = loss.numpy()
        self.losses_recon[i] = loss_recon.numpy()
        self.losses_reg[i] = loss_reg.numpy()

    def _test_inputs(self, X, Y):
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
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        return X, Y


class SparseLowRankRegressionTF(BaseReducedRankRegressionSGD):
    def __init__(self, mul=1.0, mus=1.0, training_options={}):
        """
        This decomposes regression into a low rank and sparse component

        min ||Y - XB||^2 + mu*norm(L)_nuc + mus*norm(S)_1
        s.t B = L + S

        Attributes
        ----------
        mul : float,default=1.0
            The low rank penalization strength

        mus : float,default=1.0
            The sparse penalization strength
        """
        self.mul = float(mul)
        self.mus = float(mus)
        super().__init__(training_options=training_options)

    def __repr__(self):
        out_str = "SparseLowRankRegressionTF object\n"
        return out_str

    def fit(self, X, Y, loss_function=loss_norm_lq_tf, progress_bar=True):
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
        limit_gpu(td["gpu_memory"])
        self._test_inputs(X, Y, loss_function)
        X, Y = self._transform_training_data(X, Y)
        p, q = X.shape[1], Y.shape[1]
        rng = np.random.default_rng()

        # Declare our variables
        S_ = tf.Variable(rng.normal(shape=(p, q)).astype(np.float32))
        L_ = tf.Variable(rng.normal(shape=(p, q)).astype(np.float32))
        trainable_variables = [S_, L_]

        self._initialize_losses()

        myrange = trange if progress_bar else range

        optimizer = return_optimizer_tf(td["method"], td["learning_rate"])

        for i in myrange(td["n_iterations"]):
            X_batch, Y_batch = simple_batcher_xy(td["batch_size"], X, Y)

            with tf.GradientTape() as tape:
                B_ = S_ + L_

                Y_recon = tf.matmul(X_batch, B_)
                loss_recon = loss_function(Y_batch, Y_recon)
                loss_recon = tf.reduce_mean(tf.square(Y_recon - Y_batch))

                # Regularization losses
                reg_L = loss_norm_nuclear_tf(L_)
                reg_S = tf.linalg.norm(S_, ord=1)

                loss = loss_recon + self.mul * reg_L + self.mus * reg_S

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            self._save_losses(i, loss, loss_recon, reg_L, reg_S)

        B_ = S_ + L_
        loss_reg = loss_norm_nuclear_tf(B_)
        Y_recon = tf.matmul(X, B_)
        loss_recon = loss_function(Y, Y_recon)
        loss = loss_recon + self.mul * reg_L + self.mus * loss_reg

        self.optimal_value = loss.numpy()
        self.optimal_loss = loss_recon.numpy()

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
        self.L_ = load_dictionary["model"].L_
        self.S_ = load_dictionary["model"].S_

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
        self.losses_reg_nuc = np.zeros(n_iterations)
        self.losses_reg_sparsity = np.zeros(n_iterations)

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
        self.S_ = training_variables[0].numpy()
        self.L_ = training_variables[1].numpy()
        self.B = self.S_ + self.L_

    def _save_losses(self, i, loss, loss_recon, loss_nuc, loss_sparse):
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

        loss_nuc : tf.Float
            The nuclear norm loss

        loss_sparse : tf.Float
            The L1 sparsity loss
        """
        self.losses[i] = loss.numpy()
        self.losses_recon[i] = loss_recon.numpy()
        self.losses_reg_nuc[i] = loss_nuc.numpy()
        self.losses_reg_sparsity[i] = loss_sparse.numpy()

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

        Y : np.array-like,shape=(N,q) The variables we wish to predict, should be demeaned """
        self.n_samples, self.p = X.shape
        self.q = Y.shape[1]
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        return X, Y


def loss_norm_nuclear_tf(X, safe=True):
    """ 
    This computes the nuclear norm via an SVD. This is the convex relaxation
    of rank(X).

    Paramters
    ---------
    X : tf.matrix,shape=(m,n)
        The input matrix

    safe : bool,default=True
        Should typecasting occur?

    Returns
    -------
    norm : tf.Float
        The estimated norm
    """
    if safe:
        X = tf.cast(X, tf.float32)
    s = tf.linalg.svd(X, compute_uv=False)
    norm = tf.reduce_sum(s)
    return norm


def limit_gpu(gpuMem):
    """
    Limits the GPU memory to a certain amount

    Parameters
    ----------
    gpuMem : int
        MB of memory to allocate
    """
    gpuMem = int(gpuMem)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpuMem
                    )
                ],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                " Physical GPUs, ",
                len(logical_gpus),
                " Logical GPUs",
            )
        except RuntimeError as e:
            print(e)


def return_optimizer_tf(trainingMethod, learningRate):
    """
    Creates a Keras optimizer with specific parameters

    Parameters
    ----------
    trainingMethod : str \in {'Nadam','Adam', 'SGD'}
        The SGD method

    learningRate : float
        The learning rate of optimization

    Returns
    -------
    optimizer : keras optimizer
    """
    if trainingMethod == "Nadam":
        optimizer = keras.optimizers.Nadam(learning_rate=learningRate)
    elif trainingMethod == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    elif trainingMethod == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learningRate)
    else:
        raise ValueError("Unrecognized learning strategy %s", trainingMethod)
    return optimizer
