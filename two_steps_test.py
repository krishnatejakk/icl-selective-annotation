import unittest
import torch
from two_steps import selective_annotation


class TestSelectiveAnnotation(unittest.TestCase):
    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_facility_location_selection -v
    def test_facility_location_selection(self):
        # Generate synthetic embeddings using PyTorch
        num_embeddings = 300
        embedding_dim = 128
        embeddings = torch.randn(num_embeddings, embedding_dim)

        # Define args and kwargs to mimic your actual function call
        args = type("", (), {})()
        args.selective_annotation_method = "facility_location"
        args.annotation_size = 150
        kwargs = {"embeddings": embeddings}

        # Call the selective_annotation function
        selected_indices = selective_annotation(args, **kwargs)

        print(selected_indices)

        assert type(selected_indices) == list
        assert type(selected_indices[0]) == int
        assert len(selected_indices) == args.annotation_size

    # python -m unittest two_steps_test.TestSelectiveAnnotation.test_mfl_selection -v
    def test_mfl_selection(self):
        args = type("", (), {})()
        args.selective_annotation_method = "mfl"
        args.annotation_size = 10

        num_data_points = 100  # Example number of data points
        embedding_dim = 128  # Example embedding dimension
        embeddings = torch.rand(num_data_points, embedding_dim)

        kwargs = {"embeddings": embeddings}

        selected_indices = selective_annotation(args, **kwargs)

        assert type(selected_indices) == list
        assert len(selected_indices) == args.annotation_size


if __name__ == "__main__":
    unittest.main()
