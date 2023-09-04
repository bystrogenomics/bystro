"""
This implements a base class for any model using stochastic gradient 
descent-based techniques for inference. 

Objects
-------
_BaseSGDModel(object):

Methods
-------
None
"""
import abc
import cloudpickle  # type: ignore


class _BaseSGDModel(object):
    def __init__(self, training_options=None):
        """
        The base class of a model relying on stochastic gradient descent for
        inference
        """
        if training_options is None:
            training_options = {}
        self.training_options = self._fill_training_options(training_options)

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        Method for fitting

        Parameters
        ----------
        *args:
           List of arguments

        *kwargs:
           Key word arguments
        """

    def pickle(self, path):
        """
        Method for saving the model

        Parameters
        ----------
        path : str
            The directory to save the model to
        """
        mydict = {"model": self}
        with open(path, "wb") as f:
            cloudpickle.dump(mydict, f)

    @abc.abstractmethod
    def unpickle(self, path):
        """
        Method for loading the model

        Parameters
        ----------
        path : str
            The directory to load the model from
        """

    def _fill_training_options(self, training_options):
        """
        This fills any relevant parameters for the learning algorithm

        Parameters
        ----------
        training_options : dict

        Returns
        -------
        training_opts : dict
        """
        default_options = {"n_iterations": 5000}
        training_opts = {**default_options, **training_options}
        return training_opts

    @abc.abstractmethod
    def _save_variables(self, training_variables):
        """
        This saves the final parameter values after training

        Parameters
        ----------
        training_variables :list
            The variables trained
        """
        raise NotImplementedError("_save_variables")

    @abc.abstractmethod
    def _initialize_save_losses(self):
        """
        This method initializes the arrays to track relevant variables
        during training

        Parameters
        ----------
        """
        raise NotImplementedError("_initialize_save_losses")

    @abc.abstractmethod
    def _save_losses(self, *args):
        """
        This saves the respective losses at each iteration

        Parameters
        ----------
        """
        raise NotImplementedError("_save_losses")

    @abc.abstractmethod
    def _test_inputs(self, *args):
        """
        This performs error checking on inputs for fit

        Parameters
        ----------
        """
        raise NotImplementedError("_transform_training_data")

    @abc.abstractmethod
    def _transform_training_data(self, *args):
        """
        This converts training data to adequate format

        Parameters
        ----------
        """
        raise NotImplementedError("_transform_training_data")
