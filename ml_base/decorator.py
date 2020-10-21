"""Base class for building decorators for MLModel objects."""
from ml_base.ml_model import MLModel


class MLModelDecorator(MLModel):
    """Base class for ML model decorator code.

    .. note::
        The default behavior of the MLModelDecorator base class is to do nothing and to forward the method call to
        the model that is is wrapping. Any subtypes of MLModelDecorator that would like to add on to the behavior
        of the model needs to override the default implementations in the MLModelDecorator base class.

    """

    def __init__(self, model: MLModel, **kwargs):
        """Initialize MLModelDecorator instance.

        .. note::
            This method receives the model instance and stores the reference. It also receives all keyword arguments
            and stores them in a dictionary called "_configuration". This configuration dictionary can be used to modify
            the behavior of the decorator object at runtime.

        """
        if not isinstance(model, MLModel):
            raise ValueError("Only objects of type MLModel can be wrapped with MLModelDecorator instances.")
        self._model = model
        self._configuration = kwargs

    @property
    def display_name(self) -> str:
        """Property that returns a display name for the model."""
        return self._model.display_name

    @property
    def qualified_name(self) -> str:
        """Property that returns the qualified name of the model."""
        return self._model.qualified_name

    @property
    def description(self) -> str:
        """Property that returns a description of the model."""
        return self._model.description

    @property
    def version(self) -> str:
        """Property that returns the model's version as a string."""
        return self._model.version

    @property
    def input_schema(self):
        """Property that returns the schema that is accepted by the predict() method."""
        return self._model.input_schema

    @property
    def output_schema(self):
        """Property returns the schema that is returned by the predict() method."""
        return self._model.output_schema

    def predict(self, data):
        """Predict with the model.

        :param data: data used by the model for making a prediction
        :type data: object --  can be any python type
        :rtype: python object -- can be any python type

        """
        return self._model.predict(data=data)
