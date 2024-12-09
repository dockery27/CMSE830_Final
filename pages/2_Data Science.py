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
from lmfit.models import PowerLawModel, ConstantModel, PolynomialModel

df = pd.read_csv("combined_data.csv")
df.drop(columns=["Unnamed: 0",],axis=1,inplace=True) # this is an extra index column
df_no_unc = df.drop(columns=["radius_unc", "MASS EXCESS UNC", "BINDING ENERGY UNC", "ATOMIC MASS UNC"],axis=1)

df_no_unc_subset = df_no_unc[["radius_val","MASS EXCESS","BINDING ENERGY/A", "ATOMIC MASS", " half_life [s]"]]
df_no_unc_subset[" half_life [s]"] = np.log(df_no_unc_subset[" half_life [s]"]) # make log scale due to variation in values
scaler = StandardScaler()
scaler.fit(df_no_unc_subset)
standardized = scaler.transform(df_no_unc_subset)
df_scaled = pd.DataFrame(standardized,columns=["radius_val","MASS EXCESS","BINDING ENERGY/A", "ATOMIC MASS", " half_life [s]"])
df_scaled = pd.concat([df_scaled, df_no_unc[['z', 'n', 'a', 'N-Z', ' jp', ' decay', 'radioactive', 'decay_encoded', 'spin', 'parity']]], axis=1)


st.title("Technical Background")
st.write('''
    Data was compiled from the international atomic energy agency which had three sources on 
    charge radius, mass, and life-time. Cleaning and imputation were applied to merge all three 
    datasets into a useable form. Several features were encoded and engineered for useful applications 
    to the modeling.
    
    Exploratory plots were made on the relationship of key variables to the charge radius. These were 
    primarily scatterplots, histograms, and violin plots. Fittable relationships were identified and regression was 
    performed to test the ability to explain the charge radius variation.
    
    All variables were fed into a random forest regressor model to test the performance against simple regression models, 
    and the simple model generally used in the field. The random forest was found to outperform the regression models by the 
    R^2 value.
''')
tab1, tab2, tab3 = st.tabs(["Cleaning Data", "Feature Engineering", "Model Selection/Validation"])

with tab1:
    st.write('''
        To combine the three datasets, each set was first cleaned individually before combining. 
        The charge radius dataset was cleaned first as this contains the target variable charge radius. 
        The first few rows of the dataset are shown below which contained the charge radius and some preliminary values. 
    ''')
    df_radii = pd.read_excel("charge_radii.xlsx")
    st.dataframe(df_radii.head())
    
    st.write('''
        Preliminary values (values not yet accepted by peer review) were merged with the existing values. 
        In cases where both preliminary and accepted values were reported, the accepted values were taken.
        
        Next, the nuclear mass dataset was cleaned and combined into the radius dataset. The first few rows of the dataset 
        are shown below.''')
    df_mass = pd.read_excel("nuclear_mass.xlsx")
    st.dataframe(df_mass.head())    
        
    st.write('''A few special characters and spaces had to be removed to have float datatypes in the dataframe. Due to a downloading error, corrections were 
        made to the mass value. More observations are present in the mass data including excited nuclear states, so only the 
        ground nuclear states were considered (to match the radii dataset). The dataframe was prepared to merge by matching the 
        number of protons and number of neutrons in each observation to that of the radius dataframe. 
    ''')
    
    
    st.write('''
        Lastly, the life-time datset was cleaned to prepare for merging. The original form is shown below.
        ''')
        
    df_lifetime = pd.read_csv("lifetime.csv")
    st.dataframe(df_lifetime.head())
        
    st.write('''
        A number of additional 
        columns are present in the dataset for additional decay pathways which were moved due to 
        the large degree of missingness. The main decay pathway was included.
        
        For the life-time values, stable nuclei have a NaN value. To deal with this, a mean imputation was used 
        on the life-time column. Imputated observations (stable nuclei) were given a value of 0 in a new feature radioactive, 
        and the radioactive nuceli were given a 1. The null values in the decay feature were also given the label "stable."
        
        Similarly to the mass data, the life-time data was looped over and ground states of the observations matching the 
        radius dataset were chosen.
    ''')
    
    
    
    st.write('''
        Having cleaned each dataset individually, the datasets were then combined into one for 
        easier analysis by appending the columns. Summary statistics of each column are shown below. 
        This includes the decay_encoding, spin, and parity features which were engineered. The 
        feature engineering of these variables are discussed in the following tab.
    ''')
    
    st.dataframe(df.describe())
    
    st.write('''
        Before being applied to the models, several more changes were made to the data. The life-time 
        values spanning many orders of magnitude, so the log of the life-time was taken. Numeric featues 
        were z-scored including the atomic mass, binding energy, life-time, mass excess, and charge radius. Categorical 
        features were not modified by the z-score. 
        
        The uncertainty on the measured parameters was dropped before evaluating models on the data. The final version of 
        the dataset is shown below as well as summary statistics of each feature.
    ''')
    
    st.dataframe(df_scaled.head())
    st.dataframe(df_scaled.describe())
    
    st.markdown('''
        Below is a breif description of each feature included in the analysis.
        * radius_val - The charge radius value in fm (1 m = $10^15$ m)
        * MASS EXCESS - Difference in the mass from the expected value, where the mass per nucleon of $^{12}C$ is taken as standard.
        * BINDING ENERGY/A - The energy required to seperate each nucleon from the nucleus.
        * ATOMIC MASS - The atomic mass of the nucleus in a.m.u.
        * half life [s] - Log of the half-life, which is the time for half of the population to decay.
        * z - The number of protons.
        * n - The nubmer of neutrons.
        * a - The mass number, protons + neutrons.
        * N-Z - The difference between the number of neutrons and protons.
        * jp - The spin and parity of the nucleus.
        * decay - The mode of decay for the nucleus.
        * radioactive - True/False if the nucleus is radioactive or not.
        * decay_encoded - The categorical number encoding for the decay feature.
        * spin - The nuclear spin of each nucleus.
        * parity - The parity of each nucleus with 1 as postive and -1 as negative.
    ''')
    
    
with tab2:
    st.write('''
        As mentioned in the data prepartion tab, serveral features were engineered to support 
        the data analysis. The radioactive feature was added to identify whether a nucleus is 
        radioactive or not. This corresponds to entries that have a missing life-time value. To do this, 
        NaN values in the life-time feature was imputed and encoded in the radioactive feature.
        
        There are multiple decay pathways possible depending on which nucleus is decaying. In order to 
        be compatible with the random forest model, the different decay pathways were encoded into categorical numbers from 
        0 to 7. 
        
        Finally, the jp feature in the life-time data describes the spin and parity of the nucleus. The short-hand 
        notation in the jp column is not condusive to the random forest model. This column was split into the numeric portion 
        for the spin, and prelimary values (contained in parenthesis) were taken as the final value. Parity was encoded with the + 
        corresponding to 1 and the - correspong to -1.
        
        Summary statistics of the engineered features are available in the data cleaning tab.
    ''')
    
with tab3:
    st.write('''
        Regression models were chosen for four variables which showed a clear trend with the 
        charge radius: mass number, atomic mass, mass excess, and binding energy. For the mass number 
        and atomic mass, the mathematical form of the trend line was taken from nuclear physics principles. A constant had to 
        be added to the typical form to allow for negative values, which originate from the z-scaling. In the below equation, A is the 
        amplitude, k is the power of the exponential, c is the offset, and x is the atomic mass / mass number.
    ''')
    
    st.latex("f(x; A,k,c) = A*x^k + c")
    
    st.write('''
             Nonlinear regression was performed utilizing this form, and the fit quality 
             was evaluated by comparing the R^2 and X^2 values. Best fit lines are available 
             on the Production page. 
             
             For the mass excess and binding energy, polynomial regression was utilized since 
             no clear physics principle was available. Due to the orientation of the trendlines, the 
             x and y values were swaped in the regression to allow for functional fits. That is, the x value 
             was taken as the charge radius. The equation for the polynomial regression is below where $c_i$ is 
             the ith coefficient.
             ''')
             
    st.latex("f(x; c_n) = \sum_{i=0}^n x^i * c_i")
    
    st.write('''
             The number of terms was varied for each regression until a fit was obtained 
             containing a reasonable trend. For the 
             ''')
    with st.expander("Mass Excess"):
         df_sorted = df_scaled.sort_values("radius_val",axis=0)
         x=df_sorted["radius_val"]
         y=df_sorted["MASS EXCESS"]
         
         fig1, axs = plt.subplots(2,2,figsize=(6,6))
         ax1 = axs[0][0]
         ax2 = axs[0][1]
         ax3 = axs[1][0]
         ax4 = axs[1][1]
         
         mod1 = PolynomialModel(degree=1)
         params1 = mod1.make_params()
         params1["c0"].set(value=1)
         params1["c1"].set(value=0)
         
         mod2 = PolynomialModel(degree=2)
         params2 = mod2.make_params()
         params2["c0"].set(value=1)
         params2["c1"].set(value=0)
         params2["c2"].set(value=.1)
         
         mod3 = PolynomialModel(degree=3)
         params3 = mod3.make_params()
         params3["c0"].set(value=1)
         params3["c1"].set(value=0)
         params3["c2"].set(value=.1)
         params3["c3"].set(value=-.1)
         
         mod4 = PolynomialModel(degree=4)
         params4 = mod4.make_params()
         params4["c0"].set(value=1)
         params4["c1"].set(value=0)
         params4["c2"].set(value=.1)
         params4["c3"].set(value=-.1)
         params4["c4"].set(value=-.1)
         
         result1 = mod1.fit(y, params1, x=x)
         y_plot1 = result1.eval(x=x)
         
         result2 = mod2.fit(y, params2, x=x)
         y_plot2 = result2.eval(x=x)
         
         result3 = mod3.fit(y, params3, x=x)
         y_plot3 = result3.eval(x=x)
         
         result4 = mod4.fit(y, params4, x=x)
         y_plot4 = result4.eval(x=x)
     
         ax1.plot(x,y,"b.",label="Data")
         ax1.plot(x,y_plot1,"r-",label="Best fit")
         
         ax2.plot(x,y,"b.",label="Data")
         ax2.plot(x,y_plot2,"r-",label="Best fit")
         
         ax3.plot(x,y,"b.",label="Data")
         ax3.plot(x,y_plot3,"r-",label="Best fit")
         
         ax4.plot(x,y,"b.",label="Data")
         ax4.plot(x,y_plot4,"r-",label="Best fit")
         
         st.pyplot(fig1)
         st.caption('''
                    The best fit result of the polynomial regression is shown for polynomials of 
                    degree 1 (top left), 2 (top right), 3 (bottom left), and 4 (bottom right). Based 
                    on this, the degree four polynomial was selected as the best model to use.
                    ''')
                    
    with st.expander("Binding Energy"):
         df_sorted = df_scaled.sort_values("radius_val",axis=0)
         x=df_sorted["radius_val"]
         y=df_sorted["BINDING ENERGY/A"]
         
         fig2, axs = plt.subplots(1,2,figsize=(6,3))
         ax1 = axs[0]
         ax2 = axs[1]
         
         mod1 = PolynomialModel(degree=1)
         params1 = mod1.make_params()
         params1["c0"].set(value=1)
         params1["c1"].set(value=0)
         
         mod2 = PolynomialModel(degree=2)
         params2 = mod2.make_params()
         params2["c0"].set(value=1)
         params2["c1"].set(value=0)
         params2["c2"].set(value=.1)
         
         result1 = mod1.fit(y, params1, x=x)
         y_plot1 = result1.eval(x=x)
         
         result2 = mod2.fit(y, params2, x=x)
         y_plot2 = result2.eval(x=x)
         
         ax1.plot(x,y,"b.",label="Data")
         ax1.plot(x,y_plot1,"r-",label="Best fit")
         
         ax2.plot(x,y,"b.",label="Data")
         ax2.plot(x,y_plot2,"r-",label="Best fit")
         
         st.pyplot(fig2)
         st.caption('''
                    The best fit result of the polynomial regression is shown for polynomials of 
                    degree 1 (left) and 2 (right). Based on this, the degree two polynomial was selected as the best model to use.
                    ''')
    st.write('''
             The random forest model was constructed with 100 random trees with bootstrapping allowed. 
             To evaluate the success of the model, the R^2 was evaluated and compared to the regression models. The training 
             and testing were repeated several times to estimate the impact of the random split on the achieved accuracy. All features 
             in the dataset, as discussed in the cleaning tab, were included in the random forest model.
             ''')