import os
from unittest import TestCase, mock
from unittest.mock import patch
from bystro.api.auth import CachedAuth
from bystro.api.proteomics import upload_proteomics_dataset, HTTP_STATUS_OK

current_script_path = os.path.dirname(__file__)
experiment_file = os.path.join(current_script_path, "dummy_annotation_file.txt")

# Example test case
class TestUploadProteomicsDataset(TestCase):
    @patch("requests.post")
    @patch("requests.get")
    @patch("uuid.uuid4", mock.MagicMock(side_effect=["proteomics_uuid", "annotation_uuid"]))
    @patch(
        "bystro.api.proteomics.authenticate",
        return_value=(
            CachedAuth(url="foo", access_token="bar", email="stuff"),
            {"Authorization": "Bearer token"},
        ),
    )
    def test_upload_proteomics_dataset(self, mock_authenticate, mock_requests_get, mock_requests_post):
        somascan_file = os.path.join(current_script_path, "example_data.adat")

        # Mock responses
        mock_experiment_response = mock.MagicMock()
        mock_experiment_response.status_code = HTTP_STATUS_OK
        mock_experiment_response.json.return_value = {"experimentID": "dummy_experiment_id"}

        mock_proteomics_response = mock.MagicMock()
        mock_proteomics_response.status_code = HTTP_STATUS_OK
        mock_proteomics_response.json.return_value = {"proteomicsID": "proteomics_uuid"}

        mock_annotation_response = mock.MagicMock()
        mock_annotation_response.status_code = HTTP_STATUS_OK
        mock_annotation_response.json.return_value = {"name": "Test Annotation"}

        # Configure the mock to return a response with an OK status code.
        mock_requests_post.return_value = mock_proteomics_response
        mock_requests_get.return_value = mock_annotation_response

        # Call the function under test
        result = upload_proteomics_dataset(
            protein_abundance_file=somascan_file,
            experiment_annotation_file=experiment_file,
            annotation_job_id="dummy_annotation_id",
            experiment_name="Test Experiment",
        )

        # Assert the call
        mock_authenticate.assert_called()
        mock_requests_post.assert_called()
        mock_requests_get.assert_called()

        self.assertIn("proteomicsID", result)
        self.assertEqual(result["proteomicsID"], "proteomics_uuid")


# Add more test cases to cover different branches and edge cases
