*****************
Decorator Example
*****************

Creating a Decorator
####################

Decorators are objects that allow us to extend the functionality of other objects at runtime without having to modify
the objects that are being decorated. The decorator pattern is a well-known object-oriented design pattern that helps
to make code more flexible and reusable.

Notice that we are not working with Python decorators, which are used to decorate functions and methods are load time
only. The decorators we will work with are run-time decorators since they are applied during the runtime of the program.

Creating an MLModel Class
#########################

The objects we want to decorate are MLModel objects, so we'll start by defining a simple example MLModel class to work
with::

    import os
    import pickle
    from numpy import array
    from ml_base.ml_model import MLModel, MLModelSchemaValidationException

    class IrisModel(MLModel):
        @property
        def display_name(self):
            return "Iris Model"

        @property
        def qualified_name(self):
            return "iris_model"

        @property
        def description(self):
            return "A model to predict the species of a flower based on its measurements."

        @property
        def version(self):
            return "1.0.0"

        @property
        def input_schema(self):
            return ModelInput

        @property
        def output_schema(self):
            return ModelOutput

        def __init__(self):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = open(os.path.join(dir_path, "svc_model.pickle"), 'rb')
            self._svm_model = pickle.load(file)
            file.close()

        def predict(self, data: dict):
            model_input = ModelInput(**data)
            X = array([model_input.sepal_length, model_input.sepal_width, model_input.petal_length, model_input.petal_width]).reshape(1, -1)
            y_hat = int(self._svm_model.predict(X)[0])
            targets = ["Iris setosa", "Iris versicolor", "Iris virginica"]
            species = targets[y_hat]
            return ModelOutput(species=species)

This is the same class we worked with in the previous example. Now we'll instantiate it to make sure that everything
works::

    >>> model = IrisModel()
    >>> prediction = model.predict(data={
            "sepal_length":5.1,
            "sepal_width":2.2,
            "petal_length": 1.2,
            "petal_width": 1.3})

Creating a Decorator Class
##########################

To create a decorator for MLModel classes, we'll inherit from the MLModelDecorator class::

    from ml_base import MLModelDecorator
    from ml_base.ml_model import MLModelException


    class CatchExceptionsDecorator(MLModelDecorator):
        """Decorator that catches exceptions thrown by the predict method of the model and raises an
        MLModelException instead."""

        def predict(self, data):
            try:
                return self._model.predict(data=data)
            except Exception as e:
                raise MLModelException(e)

The decorator class "wraps" around the interface of the class that it is decorating. In the decorator above,
we're wrapping around the "predict" method of the MLModel class and catching any exceptions that are raised by
the MLModel object when the predict method is called.

Now all we need is to instantiate the MLModel class and the decorator to try it out::

    >>> model = IrisModel()
    >>> decorator = CatchExceptionsDecorator(model=model)

    >>> # making a failing prediction
    >>> decorator.predict(data={
        "sepal_length":1.0,
        "sepal_width":1.1,
        "petal_length": 1.2,
        "petal_width": 1.3})

When the code above runs, it raises this exception::

    Traceback (most recent call last):
      File "<input>", line 5, in <module>
      File "<input>", line 13, in predict
    ml_base.ml_model.MLModelException: 2 validation errors for ModelInput
      sepal_length
    ensure this value is greater than 5.0 (type=value_error.number.not_gt; limit_value=5.0)
      sepal_width
    ensure this value is greater than 2.0 (type=value_error.number.not_gt; limit_value=2.0)

Now, instead of getting a ValidationError from the predict method, we'll get an MLModelException. The decorator
wrapped around the functionality of the IrisModel class and add a bit of its own functionality.

Creating a More Complex Decorator Class
#######################################

Now we can try a slightly more complex decorator::

    class AddStringDecorator(MLModelDecorator):
        """Decorator that adds a string to the display_name, qualified_name, description, and version string
        returned by the model object."""

        @property
        def display_name(self) -> str:
            return self._model.display_name + self._configuration["string"]

        @property
        def qualified_name(self) -> str:
            return self._model.qualified_name + self._configuration["string"]

        @property
        def description(self) -> str:
            return self._model.description + self._configuration["string"]

        @property
        def version(self) -> str:
            return self._model.version + self._configuration["string"]

The decorator adds a string to the display_name, qualified_name, description, and version properties of the model
object::

    >>> model = IrisModel()
    >>> decorator = AddStringDecorator(model=model, string=" extra string")

Now when we access the properties, we'll get the string we configured added to the end::

    >>> decorator.version
    '1.0.0 extra string'
    >>> decorator.description
    'A model to predict the species of a flower based on its measurements. extra string'

The keyword arguments are added to an internal property of the decorator by the default constructor of the
MLModelDecorator class.

Adding the Decorated Model to ModelManager
##########################################

Whenever a decorator is needed, we'll instantiate the model object first, then instantiate the decorator object
providing it with a reference to the model object. Wherever we need to use the model object, we can reference the
decorator object instead and the model object's methods will be called by the decorator.

Because we can use a reference to the decorator wherever it would be appropiate to use a direct reference to the
model object itself, we can also add the decorator object to the ModelManager singleton::

    >>> from ml_base.utilities import ModelManager
    >>> model_manager = ModelManager()
    >>> model_manager.add_model(decorator)
    >>> model_manager.get_model_metadata("iris_model extra string")
    {'display_name': 'Iris Model extra string', 'qualified_name': 'iris_model extra string', ...

The ModelManager is able to work with the decorated model object because it has the same interface as MLModel.
Notice that we had to reference the qualified_name of the model as "iris_model extra string" because the decorator
is adding to the qualified name of the model.