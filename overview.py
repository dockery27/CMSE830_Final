# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:36:56 2024

@author: docke
"""

import streamlit as st

st.title("Modeling Nuclear Charge Radius")
#st.write("Test")


st.write('''
         This webpage explores the relationship between the nuclear charge radius and other nuclear 
         parameters, and utilizes these parameters to model the trend of radii. The model is compared to 
         the standard model in the field of nuclear physics. Nuclear charge radius is the average distance of 
         protons in the nucleus from the center, and is an important quantity measured to understand the form 
         of the nuclear forces.
         ''')

st.markdown(
    """
    ### Organization
    1. **Production** - The production page is a self-contained product for general presentations. 
    The page starts with a quick introduction of the importance of this project and then contains tabs for 
    the data processing, which are described below. Last, a conclusion is included below the tabs.
        - Trends and Correlations - The relationship between different features and charge radius is explored.
        - Principle Components - A principal component analysis is performed on the numeric features in the dataset.
        - Regression - Regression is performed on a few key variables with clear trends relative to the charge radius.
        - Random Forest - Evaluation of a random forest model to predict charge radius. This still needs to be included. 
    2. **Data Science** - This page focuses on the technical background of the project. It starts by summarizing the 
    overall method used to analyze this data. Then specific issues are explored in tabs.
        - Data Cleaning - Three data sets were merged and missing data was handled. The specifics of this process is discussed.
        - Feature Engineering - Creation and selection of important features for this model is discussed.
        - Model Selection/Validation - Choice of the regression models and random forest paramerters are discussed and statistical validation is discussed.
"""
)