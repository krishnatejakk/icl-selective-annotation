import unittest
import json
import os

import torch

from inference_utils import inference_for_xsum


class MockTokenizer:
    """A manual mock for the tokenizer."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text, return_tensors):
        # Mocking tokenization to convert text to a list of token ids
        input_ids = torch.ones(1, 10)

        class MockTokenizerReturnValue:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return MockTokenizerReturnValue(input_ids)

    def batch_decode(self, tokens):
        # Mocking decoding to convert token ids back to text
        return ["g" for t in tokens]


class MockInferenceModel:
    """A manual mock for the inference model."""

    def generate(
        self,
        input_ids,
        do_sample,
        temperature,
        max_length,
        output_scores,
        return_dict_in_generate,
    ):
        output_len = min(max_length, len(input_ids[0]) + 10)
        gen_tokens = [*input_ids.tolist()[0], *([1] * (output_len - len(input_ids[0])))]

        class MockInferenceModelReturnValue:
            def __init__(self, gen_tokens):
                self.sequences = torch.tensor(gen_tokens)

        return MockInferenceModelReturnValue(gen_tokens)


class TestInferenceForXSum(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.prompt_cache_dir = "test_prompts"
        self.output_dir = "test_output"
        os.makedirs(self.prompt_cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup a mock prompt file
        self.mock_prompt_file = "mock_prompt.json"
        mock_prompt_content = [
            None,
            "Test prompt for unit testing.",
            {"summary": "Expected summary for testing."},
        ]
        with open(os.path.join(self.prompt_cache_dir, self.mock_prompt_file), "w") as f:
            json.dump(mock_prompt_content, f)

        # Setup mock tokenizer and model
        self.vocab = {
            "Test": 1,
            "prompt": 2,
            "for": 3,
            "unit": 4,
            "testing.": 5,
            "generated": 1,
        }
        self.mock_tokenizer = MockTokenizer(self.vocab)
        self.mock_model = MockInferenceModel()

    def tearDown(self):
        """Clean up after each test."""
        os.remove(os.path.join(self.prompt_cache_dir, self.mock_prompt_file))
        os.rmdir(self.prompt_cache_dir)
        for filename in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, filename))
        os.rmdir(self.output_dir)

    # python -m unittest inference_utils_test.TestInferenceForXSum.test_inference_for_xsum -v
    def test_inference_for_xsum(self):
        """Test the inference process and output file generation."""
        device = "cpu"  # Simplified for this test
        gold, pred = inference_for_xsum(
            self.mock_tokenizer,
            self.mock_model,
            device,
            self.prompt_cache_dir,
            self.output_dir,
            self.mock_prompt_file,
        )

        # Check if output file is created and has correct contents
        output_file_path = os.path.join(self.output_dir, self.mock_prompt_file)
        self.assertTrue(
            os.path.isfile(output_file_path), "Output file was not created."
        )

        with open(output_file_path, "r") as f:
            output_content = json.load(f)

        expected_output_content = [
            "g g g g g g g g g g",
            "g g g g g g g g g g",
            "Expected summary for testing.",
            10,
            20,
        ]

        assert output_content == expected_output_content, output_content

        assert gold == "Expected summary for testing.", gold
        assert pred == "g g g g g g g g g g", pred


if __name__ == "__main__":
    unittest.main()
