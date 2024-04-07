import unittest
import json
import os

from post_process_utils import post_process_xsum


class Args:
    """Mock arguments class to simulate argument parsing."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


class TestPostProcessXSum(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.args = Args(output_dir=self.output_dir)

    def tearDown(self):
        """Clean up after each test."""
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            os.remove(file_path)
        os.rmdir(self.output_dir)

    # python -m unittest post_process_utils_test.TestPostProcessXSum.test_post_process_xsum -v
    def test_post_process_xsum(self):
        """Test that post_process_xsum processes and writes expected output."""
        golds = ["The quick brown fox jumps over the lazy dog."]
        preds = ["A quick brown fox jumps over the lazy dog."]

        post_process_xsum(self.args, golds, preds)

        output_file_path = os.path.join(self.output_dir, "result_summary.json")
        self.assertTrue(
            os.path.isfile(output_file_path), "Output file was not created."
        )

        with open(output_file_path, "r") as f:
            result_data = json.load(f)

        expected_data = {
            "rouge1": 88.8889,
            "rouge2": 87.5,
            "rougeL": 88.8889,
            "rougeLsum": 88.8889,
        }

        assert result_data == expected_data


if __name__ == "__main__":
    unittest.main()
