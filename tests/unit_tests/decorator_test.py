import unittest

from ml_base.ml_model import MLModelException
from tests.mocks import SomeClass, MLModelMock, SimpleDecorator, ModelInput, ModelOutput, AddStringDecorator, \
    CatchExceptionsDecorator


class DecoratorTests(unittest.TestCase):

    def test_decorating_an_object_that_is_not_of_type_ml_model(self):
        """Testing decorating an object that is not of type MLModel."""
        # arrange
        model = SomeClass()

        # act, assert
        with self.assertRaises(ValueError) as context:
            decorator = SimpleDecorator(model=model)

    def test_with_simple_decorator(self):
        """Testing that the SimpleDecorator class works."""
        # arrange
        model = MLModelMock()
        decorator = SimpleDecorator(model=model)

        # act
        display_name = decorator.display_name
        qualified_name = decorator.qualified_name
        description = decorator.description
        version = decorator.version
        input_schema = decorator.input_schema
        output_schema = decorator.output_schema
        prediction = decorator.predict(data={"sepal_length": 6.0,
                                             "sepal_width": 4.0,
                                             "petal_length": 2.0,
                                             "petal_width": 1.0})

        # assert
        self.assertTrue(display_name == "display_name")
        self.assertTrue(qualified_name == "qualified_name")
        self.assertTrue(description == "description")
        self.assertTrue(version == "1.0.0")
        self.assertTrue(input_schema == ModelInput)
        self.assertTrue(output_schema == ModelOutput)
        self.assertTrue(type(prediction) is ModelOutput)

    def test_with_add_string_decorator(self):
        """Testing that the AddStringDecorator class works."""
        # arrange
        model = MLModelMock()
        decorator = AddStringDecorator(model=model, string=" test")

        # act
        display_name = decorator.display_name
        qualified_name = decorator.qualified_name
        description = decorator.description
        version = decorator.version
        input_schema = decorator.input_schema
        output_schema = decorator.output_schema
        prediction = decorator.predict(data={"sepal_length": 6.0,
                                             "sepal_width": 4.0,
                                             "petal_length": 2.0,
                                             "petal_width": 1.0})

        # assert
        self.assertTrue(display_name == "display_name test")
        self.assertTrue(qualified_name == "qualified_name test")
        self.assertTrue(description == "description test")
        self.assertTrue(version == "1.0.0 test")
        self.assertTrue(input_schema == ModelInput)
        self.assertTrue(output_schema == ModelOutput)
        self.assertTrue(type(prediction) is ModelOutput)

    def test_with_catch_exception_decorator(self):
        """Testing that the CatchExceptionsDecorator class works."""
        # arrange
        model = MLModelMock()
        decorator = CatchExceptionsDecorator(model=model)

        # act
        display_name = decorator.display_name
        qualified_name = decorator.qualified_name
        description = decorator.description
        version = decorator.version
        input_schema = decorator.input_schema
        output_schema = decorator.output_schema
        with self.assertRaises(MLModelException) as context:
            prediction = decorator.predict(data={"sepal_length": 6.0,
                                                 "sepal_width": 4.0,
                                                 "petal_length": 2.0,
                                                 "petal_width": "asdf"})

        # assert
        self.assertTrue(display_name == "display_name")
        self.assertTrue(qualified_name == "qualified_name")
        self.assertTrue(description == "description")
        self.assertTrue(version == "1.0.0")
        self.assertTrue(input_schema == ModelInput)
        self.assertTrue(output_schema == ModelOutput)


if __name__ == '__main__':
    unittest.main()
