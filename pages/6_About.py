import streamlit as st

def main():
    st.set_page_config(page_title="Tool for Analysis of MD Data",page_icon="chart_with_upwards_trend",layout="wide")
    st.header("About")
    st.markdown("This is a web application for the analysis of Molecular Dynamics Data using Machine Learning.")
    st.write("This web application can be used to perform the following tasks:")
    st.write("1. Feature Extraction")
    st.write("2. Clustering")
    st.write("3. Classification")
    st.write("4. Predictive Modeling")
    st.write("5. Visualization")
    st.write("6. Consensus Detection")
    st.write("7. Outlier Detection")
    st.write("8. Consensus Outlier Detection")
    st.text('Contact us: ')
    st.markdown("Author : Dr. Ashok Palaniappan")
    st.markdown('<p><a href="mailto:apalania@scbt.sastra.edu"> apalania@scbt.sastra.edu</a></p> ', unsafe_allow_html=True)
    st.markdown("Author : Mr. Vishnu Thumpudi")
    st.markdown('<p><a href="mailto:vishnuthumpudi@gmail.com"> vishnuthumpudi@gmail.com</a></p> ', unsafe_allow_html=True)

if __name__ == '__main__':
    main()