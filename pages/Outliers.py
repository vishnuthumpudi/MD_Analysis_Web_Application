import streamlit as st

def main():
    st.set_page_config(page_title="Topology Info",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("# Hello and Welcome to our outlier detection tool...🎈")
    st.write("We are using 12 different types of Algorithms for detection of anomaly")
    with st.sidebar:
        st.subheader("Upload Your File Here")
        csv_file = st.file_uploader("Please Upload your CSV file",type=['csv'])
        button = st.button('Detect')
    if button:
        st.write('The Page is yet to develop')
            


if __name__ == '__main__':
    main()