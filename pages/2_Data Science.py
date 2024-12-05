# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:32:27 2024

@author: docke
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:19:34 2024

@author: docke
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import altair as alt

df = pd.read_csv("combined_data.csv")
df.drop(columns=["Unnamed: 0",],axis=1,inplace=True) # this is an extra index column
df_no_unc = df.drop(columns=["radius_unc", "MASS EXCESS UNC", "BINDING ENERGY UNC", "ATOMIC MASS UNC"],axis=1)

df_no_unc_subset = df_no_unc[["radius_val","MASS EXCESS","BINDING ENERGY/A", "ATOMIC MASS", " half_life [s]"]]
df_no_unc_subset[" half_life [s]"] = np.log(df_no_unc_subset[" half_life [s]"]) # make log scale due to variation in values
scaler = StandardScaler()
scaler.fit(df_no_unc_subset)
standardized = scaler.transform(df_no_unc_subset)
df_scaled = pd.DataFrame(standardized,columns=["radius_val","MASS EXCESS","BINDING ENERGY/A", "ATOMIC MASS", " half_life [s]"])
df_scaled = pd.concat([df_scaled, df_no_unc[['z', 'n', 'a', 'N-Z', ' jp', ' decay', 'radioactive']]], axis=1)

df_scaled_lowmass = df_scaled[df_scaled["n"] >= 18]
df_scaled_lowmass = df_scaled_lowmass[df_scaled_lowmass["n"] <= 30]

st.title("Technical Background")
st.write('''
    Outline methadology used for the modeling.
''')
tab1, tab2, tab3 = st.tabs(["Cleaning Data", "Feature Engineering", "Model Selection/Validation"])
