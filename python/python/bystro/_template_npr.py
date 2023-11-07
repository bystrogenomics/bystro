"""
This provides a basic template for any model that uses numpyro as an 
inference method. It has several methods that should be filled by any
object extending the template, namely
    fit
    _fill_hp_options

These are the methods for running the samples given data and providing 
hyperameter selections respectively.

Objects
-------
BaseNumpyroModel(mcmc_options=None,hp_options=None)
    This is the template Numpyro model

"""
import abc
import cloudpickle  # type: ignore
import numpyro  # type: ignore


class BaseNumpyroModel(abc.ABC):
    """
    The template for a numpyro-based model
    """

    def __init__(self, mcmc_options=None, hp_options=None):
        """

        Parameters
        ----------

        Returns
        -------

        """
        if mcmc_options is None:
            mcmc_options = {}
        if hp_options is None:
            hp_options = {}
        self.mcmc_options = self._fill_mcmc_options(mcmc_options)
        self.hp_options = self._fill_hp_options(hp_options)
        self.samples = None
        self._model = None
        self.model_kwargs = None

    @abc.abstractmethod
    def fit(self, *args):
        """

        Parameters
        ----------

        Returns
        -------

        """

    def render_model(self):
        """
        This provides a graphical representation of the model

        Parameters
        ----------

        Returns
        -------

        """
        assert self._model is not None, "Fit model first"
        return numpyro.render_model(self._model, model_kwargs=self.model_kwargs)

    def pickle(self, path):
        """
        This saves samples from a fit model

        Parameters
        ----------

        Returns
        -------

        """
        mydict = {"model": self}
        with open(path, "wb") as f:
            cloudpickle.dump(mydict, f)

    @classmethod
    def unpickle(cls, path):
        """
        This loads samples from a previously saved model

        Parameters
        ----------

        Returns
        -------

        """
        with open(path, "rb") as f:
            myDict = cloudpickle.load(f)
        return myDict["model"]

    def _fill_mcmc_options(self, mcmc_options):
        """
        This fills in default MCMC options of the sampler. Further methods
        might override these but these are common/basic enough to leave in
        as an implemented method.

        Parameters
        ----------

        Returns
        -------

        """
        default_options = {
            "num_chains": 1,
            "num_warmup": 500,
            "num_samples": 2000,
        }
        mopts = {**default_options, **mcmc_options}
        return mopts

    @abc.abstractmethod
    def _fill_hp_options(self, hp_options):
        """
        This fills in default hyperparameters of the model. Since these are
        not conserved between models we leave this as an abstract method
        to be filled in per model.

        Parameters
        ----------

        Returns
        -------

        """
