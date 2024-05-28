import streamlit as st

def main():
    st.set_page_config(page_title="Tool for Analysis of MD Data",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("# Home Page")
    st.divider()
    st.header("Want to know more about Molecular Dynamics Simulation?")
    lst = ["MD simulations predict how every atom in a protein or other molecular system will move over time.",
           "These simulations can capture a wide variety of important biomolecular processes, including conformational change, ligand binding, protein folding etc., revealing the positions of all the atoms at femtosecond temporal resolution.",
           "Eg : Gromacs, Plumed, Amber"]
    s = ""
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    st.divider()
    st.header("Want to know more about Machine learning?")
    lst = ["Its a branch of Artificial Intelligence that enables computers to “self-learn” from training data and improve over time, without being explicitly programmed.",
           "The ML Algorithms are having ability to detect patterns in data and used to learn from them in order to make predictions.",
           "Eg : Regression, Random forest Classifier"]
    s = ""
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    st.divider()
    st.header("Contribution of Machine Learning in Molecular Dynamics")
    lst = ["ML contributes by automating feature extraction, clustering, classification, and predictive modeling. ",
           "It finds applications in drug discovery, materials science, and biophysics, aiding in the prediction of binding affinities, materials properties, and protein dynamics.",
           "Understanding complicated molecular behavior from molecular dynamics simulations could be improved using machine learning (ML) methods."]
    s = ""
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    st.divider()

#The topology file defines the molecular structure of the system under study. This includes details such as atom types, atomic charges, bond connectivity, and possibly parameters for non-bonded interactions like van der Waals forces and electrostatic interactions.
if __name__ == '__main__':
    main()