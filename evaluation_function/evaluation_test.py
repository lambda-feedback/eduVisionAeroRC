import unittest

from .evaluation import Params, evaluation_function

class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """
    def test_evaluation(self):
        # Use a sample image from the internet (or an empty string if you don't have one)
        # Use local file as image for test
        import os
        local_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation_test.py.jpg"))
        response = [{
            "comment": "",
            "name": "evaluation_test.py.jpg",
            "size": os.path.getsize(local_image_path),
            "type": "image/jpeg",
            "url": f"file://{local_image_path}"
        }]
        answer = "test_answer"
        params = Params(target="test_class", show_target=True, draw_images=False, debug=True)

        result = evaluation_function(response, answer, params).to_dict()

        # Check if the result is a dictionary and contains the key 'is_correct' and 'feedback_items'
        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        self.assertIn("feedback", result)

    def test_no_response_returns_feedback_without_crashing(self):
        result = evaluation_function([], "test_answer", Params()).to_dict()

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_correct"])
        self.assertIn("feedback", result)
        self.assertIn("No Image Provided", result["feedback"])

    def test_legacy_param_names_still_work(self):
        # `return_images`/`show_target` were the param names documented before
        # the `draw_images` rename; make sure old question configs still work.
        import os
        local_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation_test.py.jpg"))
        response = [{
            "comment": "",
            "name": "evaluation_test.py.jpg",
            "size": os.path.getsize(local_image_path),
            "type": "image/jpeg",
            "url": f"file://{local_image_path}",
        }]
        params = Params(target="test_class", return_images=False, show_target=False)

        result = evaluation_function(response, "test_answer", params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("is_correct", result)
        # show_target=False must suppress the "Target" feedback section
        self.assertNotIn("Target component", result["feedback"])
