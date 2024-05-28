import pytest
import tempfile
import pandas as pd
from unittest import mock
from your_module import main  # replace with the actual module name

# Mocking Streamlit methods
st = mock.MagicMock()
modules = {
    "streamlit": st,
    "streamlit.set_page_config": st.set_page_config,
    "streamlit.markdown": st.markdown,
    "streamlit.divider": st.divider,
    "streamlit.file_uploader": st.file_uploader,
    "streamlit.number_input": st.number_input,
    "streamlit.checkbox": st.checkbox,
    "streamlit.button": st.button,
    "streamlit.warning": st.warning,
    "streamlit.success": st.success,
    "streamlit.spinner": st.spinner,
    "streamlit.write": st.write,
    "streamlit.dataframe": st.dataframe,
    "streamlit.table": st.table,
    "streamlit.toast": st.toast,
    "streamlit.error": st.error,
}

@mock.patch.dict("sys.modules", modules)
def test_main():
    # Mocking the file uploaders
    st.file_uploader.return_value = tempfile.NamedTemporaryFile(delete=False, suffix=".xtc")
    
    # Mocking the checkboxes
    st.checkbox.return_value = True
    
    # Mocking the button
    st.button.return_value = True
    
    # Call the main function
    main()
    
    # Assert that the file uploader was called with the correct arguments
    st.file_uploader.assert_any_call("Upload your Trajectory file", type=["xtc","trr","dcd"])
    st.file_uploader.assert_any_call("Upload your PDB file", type=["pdb"])
    
    # Assert that the checkboxes were called with the correct arguments
    st.checkbox.assert_any_call("Compute Root Mean Square Deviation")
    st.checkbox.assert_any_call("Compute Root Mean Square Fluctuation")
    st.checkbox.assert_any_call("Compute Solvent Accessible Surface Area")
    st.checkbox.assert_any_call("Compute Radius of Gyration")
    st.checkbox.assert_any_call("Compute Hydrogen Bonds")
    st.checkbox.assert_any_call("Compute Native Contacts")
    
    # Assert that the button was called with the correct argument
    st.button.assert_called_with("Process")
    
    # Assert that the warnings were displayed
    st.warning.assert_any_call("Computing RMSF takes longer time than expected", icon="⚠️")
    st.warning.assert_any_call("Computing SASA takes longer time than expected", icon="⚠️")
    
    # Assert that the success message was displayed
    st.success.assert_called_with("Successfully Uploaded the trajectory and the PDB file")
    
    # Assert that the spinner was displayed
    st.spinner.assert_called_with('Wait a while...')
    
    # Assert that the dataframe was displayed
    st.dataframe.assert_called()
    
    # Assert that the table was displayed
    st.table.assert_called()
    
    # Assert that the toast was displayed
    st.toast.assert_called_with('Hooray! We have computed all your selected features', icon='🎉')