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
_BaseNPRModel(object)
    This is the template Numpyro model

"""
import abc
import cloudpickle  # type: ignore
import numpyro  # type: ignore


class _BaseNumpyroModel:
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

    def fit(self, *args):
        """

        Parameters
        ----------

        Returns
        -------

        """

    def render_model(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        assert self._model is not None, "Fit model first"
        return numpyro.render_model(self._model, model_kwargs=self.model_kwargs)

    def pickle(self, path):
        """

        Parameters
        ----------

        Returns
        -------

        """
        assert self.samples is not None, "Fit model first"
        with open(path, "wb") as f:
            cloudpickle.dump(self.samples, f)

    def unpickle(self, path):
        """

        Parameters
        ----------

        Returns
        -------

        """
        with open(path, "rb") as f:
            return cloudpickle.load(f)

    def _fill_mcmc_options(self, mcmc_options):
        """

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
        if mopts["num_chains"] > 1:
            print("You tried to set num_chains>1. This has been ignored.")
            mopts["num_chains"] = 1
        return mopts

    @abc.abstractmethod
    def _fill_hp_options(self, hp_options):
        """

        Parameters
        ----------

        Returns
        -------

        """
