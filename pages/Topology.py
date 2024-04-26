import streamlit as st
import mdtraj as md
import tempfile

def main():
    st.set_page_config(page_title="Topology Info",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("# Do you wanna know more about the topology file ?🎈")
    lst = ["The topology file defines the molecular structure of the system under study. This includes details such as atom types, atomic charges, bond connectivity, and possibly parameters for non-bonded interactions like van der Waals forces and electrostatic interactions.",
           "The topology file specifies the force field parameters used in the MD simulation. This encompasses the specific functional forms and parameter values for bonds, angles, dihedrals, and non-bonded interactions ",
           "The topology file often includes information about the overall composition of the system, such as the number of molecules of different types (e.g., water, proteins, ions) and any constraints applied to the system ",
           "Eg : *.pdb, *.psf"]
    s = ""
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    with st.sidebar:
        st.subheader("Upload Your Files Here")
        pdb_file = st.file_uploader("Upload your PDB File here",type=['pdb'])
        button = st.button('Process')
    
    if button:
        if pdb_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_pdb:
                tmp_pdb.write(pdb_file.getvalue())
                pdb_path = tmp_pdb.name
            with st.spinner('Wait a while...'):
                topology = md.load(pdb_path).topology
                st.success("Successfully loaded the topology file")
                table, bonds = topology.to_dataframe() # converting into a dataframe
                st.write("Here are the first five rows of the topology file")
                st.dataframe(table.head())
                st.write("Here are the last five rows of the topology file")
                st.dataframe(table.tail())
                st.balloons()
    

if __name__ == '__main__':
    main()