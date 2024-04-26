import streamlit as st  
import mdtraj as md
import time 
import os
import tempfile
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from itertools import combinations

def compute_rmsd(trajectory):
    rmsds = md.rmsd(trajectory, trajectory, 0)
    st.success("Successfully computed rmsd")
    frames = np.arange(len(rmsds))  # Frame indices
    p = figure(
    title='Frame vs RMSD',
    x_axis_label='Frames',
    y_axis_label='RMSD')
    p.line(frames,rmsds, legend_label='RMSD ', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return rmsds

def compute_rog(trajectory):
    rog = md.compute_rg(trajectory, masses=None)
    st.success("Successfully computed Radius of Gyration")
    frames = np.arange(len(rog))
    p = figure(
    title='Frame vs RoG',
    x_axis_label='Frames',
    y_axis_label='RoG')
    p.line(frames,rog, legend_label='RoG', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return rog

def compute_rmsf(trajectory):
    rmsf = md.rmsf(trajectory, trajectory, 0)
    st.success("Successfully computed RMSF")
    atom_numbers = np.arange(1, len(rmsf) + 1)
    p = figure(
    title='Atoms vs RMSF',
    x_axis_label='Atoms',
    y_axis_label='RMSF')
    p.line(atom_numbers,rmsf, legend_label='RMSF', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return rmsf

def compute_sasa(trajectory):
    sasa = md.shrake_rupley(trajectory)
    st.success("Successfully computed SaSa")
    total_sasa = sasa.sum(axis=1)
    p = figure(
    title='Time vs SaSa',
    x_axis_label='Time [ps]',
    y_axis_label='Total SASA (nm)^2')
    p.line(trajectory.time,total_sasa, legend_label='SaSa', line_width=2)
    st.bokeh_chart(p, use_container_width=True)
    return sasa

def compute_hbond(trajectory):
    h_bonds = md.baker_hubbard(trajectory)
    st.success("Successfully computed Hydrogen Bonds")
    st.write("Total number of hydrogen bonds formed over the simulation : %d" % len(h_bonds))
    st.text("Here below you can find the residues which contain hydrogen bonds: ")
    label = lambda hbond : '%s -- %s' % (trajectory.topology.atom(hbond[0]), trajectory.topology.atom(hbond[2]))
    for hbond in h_bonds:
        st.write(label(hbond))

def best_hummer_q(traj, native):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]
    
    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        
    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        
    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """
    
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    
    # get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i,j) for (i,j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 3])
    
    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    # print("Number of native contacts", len(native_contacts))
    
    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)
    
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q ,native_contacts

def compute_native(trajectory):
    q, native = best_hummer_q(trajectory, trajectory[2])
    st.success("Successfully computed contacts that determine protein folding")
    st.write("Fraction of native contacts determing native folding mechanism in the trajectory are : %d" % len(native))

def main():
    st.set_page_config(page_title="Tool for Analysis of MD Data",page_icon="chart_with_upwards_trend",layout="wide")
    st.markdown("# Hello and Welcome To Feature Extraction Web Page🎈")
    st.divider()
    st.header("Do you wanna more about trajectory file format?")
    lst = ["The trajectory file records the three-dimensional coordinates (x, y, z) of each atom in the system at discrete time steps throughout the simulation.",
           "These coordinates provide a detailed snapshot of the molecular configuration at every point in time, allowing researchers to analyze the system's structural evolution, conformational changes, and dynamics.",
           " In addition to atomic positions, trajectory files often include velocity information for each atom at each time step. Velocities are crucial for calculating kinetic properties and can be used to derive thermodynamic quantities such as temperature.",
           "Trajectory files typically store metadata related to the simulation parameters and settings, including the integration time step, total simulation time, and possibly details about the force field and simulation conditions.",
           "Ex : *.trr. *.xtc, *.dcd"]
    s = ""
    for i in lst:
        s += "- " + i + "\n"
    st.markdown(s)
    st.divider()
    with st.sidebar:
        st.subheader("Upload Your Files Here")
        xtc_file = st.file_uploader("Upload your XTC File here",type=['xtc'])
        pdb_file = st.file_uploader("Upload your PDB File here",type=['pdb'])
        st.text('Select the features: ')
        rmsd_box = st.checkbox('Compute RMSD')
        rog_box = st.checkbox('Compute RoG')
        rmsf_box = st.checkbox('Compute RMSF')
        sasa_box = st.checkbox('Compute SASA')
        hbond_box = st.checkbox('Compute H-Bonds')
        native_box = st.checkbox('Compute Native Contacts')
        button = st.button('Process')

    if sasa_box:
        st.warning('Computing SASA takes longer time than expected', icon="⚠️")
        # sasa = compute_sasa(trajectory)
    
    if button: 
        if xtc_file and pdb_file:
        # Save uploaded files to a temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xtc") as tmp_xtc:
                    tmp_xtc.write(xtc_file.getvalue())
                    xtc_path = tmp_xtc.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_pdb:
                tmp_pdb.write(pdb_file.getvalue())
                pdb_path = tmp_pdb.name
            with st.spinner('Wait a while...'):
                trajectory = md.load_xtc(xtc_path,top=pdb_path)
                st.success("Successfully loaded the trajectory and we are ready to start analysis")
                st.write(f"Number of frames: {trajectory.n_frames}")
                st.write(f"Number of atoms: {trajectory.n_atoms}")
                st.write(f"Number of residues: {trajectory.n_residues}")
            if rmsd_box:
                rmsd = compute_rmsd(trajectory)
                data = {
                    "Frame": range(1, len(rmsd) + 1),
                    "RMSD": rmsd
                }
                df = pd.DataFrame(data)
                st.text("Description on RMSD")
                st.write(df['RMSD'].describe())
            if rog_box:
                rog = compute_rog(trajectory)
                df['RoG'] = rog
                st.text("Description on RoG")
                st.write(df['RoG'].describe())
            if rmsf_box:
                rmsf = compute_rmsf(trajectory)
                st.text("You can download the below dataframe by clicking on download symbol")
                daf = pd.DataFrame({'Atom' : range(1, len(rmsf) + 1), 'RMSF' : rmsf})
                daf
                st.text("Description on RMSF")
                st.write(daf['RMSF'].describe())
            if hbond_box:
                compute_hbond(trajectory)
            if sasa_box:
                sasa = compute_sasa(trajectory)
                df['SaSa'] = sasa
                st.text("Description on SaSa")
                st.write(df['SaSa'].describe())
            if native_box:
                compute_native(trajectory)
            st.text("You can download the below dataframe by clicking on download symbol")
            df
            st.info("We have successfully computed all the features you have choosen",icon="ℹ️")
            st.snow()
            
if __name__ == '__main__':
    main()