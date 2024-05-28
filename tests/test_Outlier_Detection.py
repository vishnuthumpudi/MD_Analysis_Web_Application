import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os

# Mock functions
def fitting_classifiers(number):
    pass

def detect_outliers(df, number):
    return df

# Your main function here...

class TestMain(unittest.TestCase):
    @patch('streamlit.set_page_config')
    @patch('streamlit.markdown')
    @patch('streamlit.divider')
    @patch('streamlit.file_uploader')
    @patch('streamlit.number_input')
    @patch('streamlit.checkbox')
    @patch('streamlit.button')
    @patch('streamlit.warning')
    @patch('streamlit.success')
    @patch('streamlit.toast')
    def test_main(self, mock_toast, mock_success, mock_warning, mock_button, mock_checkbox, mock_number_input, mock_file_uploader, mock_divider, mock_markdown, mock_set_page_config):
        # Mocking Streamlit methods
        mock_file_uploader.return_value = MagicMock(spec=tempfile._TemporaryFileWrapper)
        mock_number_input.return_value = 0.1
        mock_checkbox.return_value = True
        mock_button.return_value = True

        # Mocking CSV file
        df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            df.to_csv(tmp_csv.name, index=False)
            mock_file_uploader.return_value.name = tmp_csv.name

        # Call the main function
        main()

        # Assertions
        mock_set_page_config.assert_called_once()
        mock_markdown.assert_called_once()
        mock_divider.assert_called_once()
        mock_file_uploader.assert_called_once_with("Upload your CSV file", type=["csv"])
        mock_number_input.assert_called_once_with("Enter the outlier fraction", min_value=0.0, max_value=1.0, step=0.01, placeholder="Type a number...")
        mock_checkbox.assert_called()
        mock_button.assert_called_once()
        mock_success.assert_called()
        mock_toast.assert_called()

        # Clean up
        os.remove(tmp_csv.name)

if __name__ == '__main__':
    unittest.main()