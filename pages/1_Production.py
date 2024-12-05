
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import altair as alt
import lmfit 
from lmfit.models import PowerLawModel, ConstantModel, PolynomialModel

def create_biplot(X_scaled, feature_names, pc1=0, pc2=1, scale_arrows=5,title="Biplot"):
    # Perform SVD
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    # Calculate scores (projected data points)
    scores = U * S
    
    # Get loadings (eigenvectors)
    loadings = Vt.T
    
    # Calculate explained variance
    var_exp = S**2 / np.sum(S**2) # calculate the variance explianed by each feature
    cum_var_exp = np.cumsum(var_exp) # calculate the cumulative variance explained at each feature

    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Plot scores
    #ax.scatter(scores[:, pc1], scores[:, pc2], alpha=0.7)
    sns.scatterplot(x=scores[:, pc1], y=scores[:, pc2], alpha=0.8, ax=ax, hue=df_scaled[" decay"], palette="pastel")
    
    # Plot feature vectors
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0,
                loadings[i, pc1] * scale_arrows,
                loadings[i, pc2] * scale_arrows,
                color='r', alpha=0.5)
        ax.text(loadings[i, pc1] * scale_arrows * 1.15,
                loadings[i, pc2] * scale_arrows * 1.15,
                feature, color='r')
    
    # Add labels and title
    ax.set_xlabel(f'PC{pc1+1} ({var_exp[pc1]:.1%} explained var.)')
    ax.set_ylabel(f'PC{pc2+1} ({var_exp[pc2]:.1%} explained var.)')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True)
    
    # Make equal aspect ratio
    ax.set_aspect('equal')
    
    return fig, ax

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



st.title("Modeling Nuclear Charge Radius")
st.markdown('''
    The strong nuclear force is the least understood fundamental force and is responsible for the 
    the existence of all matter heavier than hydrogen. Modern theories of the strong force are tested against 
    observables of the nucleus, of which nuclear charge radius (R) is one of the most important.
    
    The nuclear charge radius is a measure of how far the protons are from the center of the nucleus, and here the 
    mean-square distance is used. In nuclear physics, the charge radius is typically calculated from purely theoretical 
    principles. The below equation is taken as the general trend where (A) is the number of protons + neutrons (mass number), 
    and computationally intensive methods are used for specific cases. ''')   
st.latex("R = 1.2 A^{1/3}")
st.markdown('''    
    Here, we develop a model based on other nuclear observables to improve upon the simple model above. The correlations between charge radius 
    and nuclear mass, decay mode, binding energy, and more properties are shown. Polynomial regression and random forest models are used to fit the 
    available charge radius data.
    
    Data was compiled from the international atomic energy agency databases for charge radii, nuclear mass,
    and half lives. This process is explained in detail in the github and data science page.
''')
tab1, tab2, tab3, tab4 = st.tabs(["Trends and Correlations", "Principle Components", "Regression", "Random Forests"])

with tab1:
    st.write('''
        The nuclear charge radius has a general trend that is typically modeled by the mass number. This relationship is shown in the first 
        dropdown and other observables are explored in the following dropdowns.
    ''')
    with st.expander("Mass Number"):
        st.write('''
                 The mass number refers to the number of protons and neutrons in the nucleus. This is 
                 an integer number, and in this dataset it varies from 1 to 248 with a median of 137.
        ''')
        chart = alt.Chart(df_scaled).mark_point().encode(
            x=alt.X('a', title="Mass Number"),
            y=alt.Y('radius_val', title="Charge Radius (z-scaled)"),
            color=' decay',
            tooltip=['a', 'radius_val', 'z', 'n', ' decay']
            ).interactive()
        st.altair_chart(chart)
        st.caption('''
                   The standard model taken for nuclear charge radius follows the trend in 
                   this figure. The color of the points refers to the mode of radioactive decay 
                   of the nucleus.
        ''')
        
    with st.expander("Atomic Mass"):
        st.write('''
                 The atomic mass is the real mass of the nucleus as opposed to the number of protons and neutrons. This is 
                 a discreet value and was z-scaled to vary between -2 and 2.
        ''')
        chart = alt.Chart(df_scaled).mark_point().encode(
            x=alt.X('ATOMIC MASS', title="Atomic Mass"),
            y=alt.Y('radius_val', title="Charge Radius (z-scaled)"),
            color=' decay',
            tooltip=['a', 'radius_val', 'z', 'n', ' decay']
            ).interactive()
        st.altair_chart(chart)
        st.caption('''
                   The real mass value follows the standard mass number model. 
                   The color of the points refers to the mode of radioactive decay 
                   of the nucleus.
        ''')
        
    with st.expander("Radioactivity"):
        st.write('''
                 Nuclei can either be stable (do not decay) or radioactive (decay to a different nucleus). 
                 Here we see the effect of radioactivity on the nuclear charge radius.
        ''')
        fig = plt.figure(figsize=(6,6))
        sns.histplot(df_scaled,x="radius_val",hue="radioactive")
        plt.xlabel("Charge Radius (z-scaled)",fontsize="x-large")
        plt.ylabel("Count",fontsize="x-large")
        st.pyplot(plt)
        st.caption('''
                   A histrogram of the charge radii where the radioactive distribution (1) is 
                   compared to the stable distribution (0). The radioactive distribution extends to larger 
                   radii than the stable distribution, but there are no other major differences.
        ''')
    
    with st.expander("Decay Mode"):
        st.write('''
                 Radioactive nuceli can decay in a variety of pathways. In the below plot, the charge radii 
                 of different decay pathways is compared.
        ''')
        plt.figure(figsize=(8,4))
        sns.violinplot(df_scaled,x=" decay",y="radius_val")
        plt.xlabel("Decay Mode",fontsize="x-large")
        plt.ylabel("Charge Radius (z-scaled)",fontsize="x-large")
        st.pyplot(plt)
        st.caption('''
                   Each decay mode spans various radii distributions. Alpha and double beta-plus 
                   decay are confined to large radii. However, beta-minus and stable nuclei span much 
                   of the range of radii. Other decay modes span intermediate regions.
        ''')
    
    with st.expander("Mass Excess"):
        st.write('''
                 Mass excess is a measure of how much the nuclear mass deviates from the expected quantity 
                 for the mass number of the nucleus. Carbon-12 is taken as the standard mass value for computing 
                 mass excess.
        ''')
        chart2 = alt.Chart(df_scaled).mark_point().encode(
            x=alt.X('MASS EXCESS', title='Mass excess (z-scaled)'),
            y=alt.Y('radius_val', title='Charge Radius (z-scaled)'),
            color=' decay',
            tooltip=['MASS EXCESS', 'radius_val', 'z', 'n', ' decay', ]
            ).interactive()
        st.altair_chart(chart2)
        st.caption('''
                   Charge radii is plotted as a function of the mass excess value, which has been z-scaled. 
                   The charge radii roughly follow a rotated parabola relationship with the mass excess values.
        ''')
    
    
    with st.expander("Binding Energy per Nucleon"):
        st.write('''
                 Binding energy is a measure of how tighly bound individual nucleons are in the nucleus. Binding energy 
                 per nucleon peaks around the iron region of the nuclear chart and decreases at high and low masses.
        ''')
        chart3 = alt.Chart(df_scaled).mark_point().encode(
            x=alt.X('BINDING ENERGY/A', title="Binding enrgy per nucleon (z-scaled)"),
            y=alt.Y('radius_val', title='Charge radius (z-scaled)'),
            color=' decay',
            tooltip=['BINDING ENERGY/A', 'radius_val', 'z', 'n', ' decay']
            ).interactive()
        st.altair_chart(chart3)
        st.caption('''
                   Charge radii is plotted as a function of the binding energy per nucleon. The colors encode the decay 
                   method of the nucleus. An interesting trend is observed between the two variables which is hard to model 
                   mathematically.
        ''')
        
    with st.expander("Half Life"):
        st.write('''
                 Not working now; need to update.
        ''')
        
    st.write('''
        The above plots show that the standard global trend of mass number strongly correlates with the charge radius values. 
        Interestingly, trends are also observed between the decay mode, mass excess, and binding energy per nucleon.
    ''')
with tab2:
    st.write('''
             The charge radius, atomic mass, mass excess, binding energy, and half life are 
             all continuous numbers shown to have correlated relationships in the dataset. To further characterize 
             their relationships, a principle component analysis is performed. Biplots are created to show which 
             variables are combined into principle components.
             ''')
             
    with st.expander("Biplot of All Variabels"):
        fig, ax = create_biplot(standardized, ["radius_val","MASS EXCESS","BINDING ENERGY/A", "ATOMIC MASS", " half_life [s]"],
                                title="")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption('''
                   Data is plotted as a scatter plot of the second principle components against the 
                   first principle component. The color of each point corresponds to the decay mode. 
                   Original features of the dataset are plotted as arrows in the projection direction. 
                   Atomic mass most strongly correlates with the first principle component and binding energy most 
                   strongly correlates with the second principle component.
        ''')
        
     
    with st.expander("Biplot Without Target"):
        fig, ax = create_biplot(standardized[:,1:], ["MASS EXCESS","BINDING ENERGY/A", "ATOMIC MASS", " half_life [s]"],
                                title="")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption('''
                   The figure is anaalogous to the version above, but the target has been removed in order to see if any inputs are 
                   repetetive. In this version, mass excess correlates most with the first principle component, and  
                   atomic mass and half life correlate most with the second principle component.
        ''')
        
    st.write('''
             Based on the principle component analysis, the most variation in the data with all variables included is explained by 
             the atomic mass and binding energy. This suggests these quantities may be useful in understanding the variations 
             in charge radii. In addition, a secondary principle component analysis without the target shows that none 
             of the parameters are strongly correlated with each other. Only the half life and atomic mass share a similar (but opposite) 
             location on the biplot.
             ''')
             
with tab3:
    st.write('''
             A first attempt to construct an improved model is performed by  
             regressing the scatterplots that showed a clear relationship with charge radius. The charge radii vs mass number is 
             fit with a power law to compare to. Additionally, fits are performed on the atomic mass, mass excess, and binding energy.
             ''')
             
    with st.expander("Mass Number"):
        st.write('''
                 To represent the standard charge radius approximation used, a regression is performed 
                 on a power law model where the exponent and amplitude are free parameters. The power 
                 law equation is written below. An additional shift term must be added for the power 
                 law to reach negative numbers.
                 ''')
        st.latex("f(x; A,k,c) = A*x^k + c")
        y=df_scaled["radius_val"]
        x=df_scaled["a"]
        
        mod = PowerLawModel() + ConstantModel()
        params = mod.make_params()
        params["exponent"].set(value=1/3)
        params["amplitude"].set(value=1)
        result = mod.fit(y, params, x=x)
        
        x2 = np.linspace(0,250,500)
        y_plot = result.eval(x=x2)

        fig = plt.figure(figsize=(6,6))
        plt.plot(x,y,"b.",label="Data")
        plt.plot(x2,y_plot,"r-",label="Best fit")
        plt.xlabel("Mass Number",fontsize="large")
        plt.ylabel("Charge Radius (z-scaled)",fontsize="large")
        plt.legend(fontsize="large")
        plt.text(50,-4,"R^2 = "+str(round(result.rsquared,4)))
        plt.text(50,-5,"X^2 = "+str(round(result.chisqr,4)))
        
        st.pyplot(fig)
        st.caption('''
                   The best fit result of the power law is the standard method used to 
                   estimate the charge radius. All other regressions performed will be 
                   compared to the R^2 and X^2 found here.
                   ''')
        
    with st.expander("Atomic Mass"):
        st.write('''
                 An analogous power law regression is performed on the atomic mass data with 
                 the same equation as above. Since the atomic mass data had been z-scaled, the x-values 
                 in the regression were shifted to positive values to avoid errors.
                 ''')
                 
        df_sorted = df_scaled.sort_values("ATOMIC MASS",axis=0)
        y=df_sorted["radius_val"]
        x=df_sorted["ATOMIC MASS"]-min(df_sorted["ATOMIC MASS"]) # shift to only positive values
        
        mod = PowerLawModel() + ConstantModel()
        params = mod.make_params()
        params["exponent"].set(value=1/3)
        params["amplitude"].set(value=1)
        result = mod.fit(y, params, x=x)
        
        x2 = np.linspace(0,4.2,500)
        y_plot = result.eval(x=x2)

        fig = plt.figure(figsize=(6,6))
        plt.plot(x,y,"b.",label="Data")
        plt.plot(x2,y_plot,"r-",label="Best fit")
        plt.xlabel("Atomic Mass (z-scaled and shifted)",fontsize="large")
        plt.ylabel("Charge Radius (z-scaled)",fontsize="large")
        plt.legend(fontsize="large")
        plt.text(1,-4,"R^2 = "+str(round(result.rsquared,4)))
        plt.text(1,-5,"X^2 = "+str(round(result.chisqr,4)))
        
        st.pyplot(fig)
        st.caption('''
                   The best fit result of the atomic mass data with a power law is plotted 
                   against data. The R^2 and X^2 fit statistics are provided on the figure.
                   ''')
         
    with st.expander("Mass Excess"):
        st.write('''
                 A polynomial function was used to fit the charge radius as a function of 
                 mass excess. Due to the orientation of the graph in the first tab, the regression 
                 was performed with opposite x and y compared to the previous examples. A fourth 
                 degree polynomial had to be used to achieve reasonable fit results for this distribution.
                 ''')
        
        df_sorted = df_scaled.sort_values("radius_val",axis=0)
        x=df_sorted["radius_val"]
        y=df_sorted["MASS EXCESS"]
        
        mod = PolynomialModel(degree=4)
        params = mod.make_params()
        params["c0"].set(value=-1)
        params["c1"].set(value=0)
        params["c2"].set(value=.1)
        params["c3"].set(value=-.1)
        params["c4"].set(value=-.1)
        
        result = mod.fit(y, params, x=x)
        
        x2 = np.linspace(-6,1.5,500)
        y_plot = result.eval(x=x2)

        fig = plt.figure(figsize=(6,6))
        plt.plot(x,y,"b.",label="Data")
        plt.plot(x2,y_plot,"r-",label="Best fit")
        plt.xlabel("Charge Radius (z-scaled)",fontsize="large")
        plt.ylabel("Mass excess (z-scaled)",fontsize="large")
        plt.legend(fontsize="large")
        plt.text(-5,0.5,"R^2 = "+str(round(result.rsquared,4)))
        plt.text(-5,-0.5,"X^2 = "+str(round(result.chisqr,4)))
       
        st.pyplot(fig)
        st.caption('''
                   The best fit result of the mass excess data is plotted with a fourth degree 
                   polynomi best fit. The R^2 and X^2 fit statistics are provided on the figure and are 
                   worse than the mass data but still informitive.
                   ''')
       
         
    with st.expander("Binding Energy"):
        st.write('''
                 A polynomial function was used to fit the charge radius as a function of 
                 binding energy. Due to the orientation of the graph in the first tab, the regression 
                 was performed with opposite x and y compared to the previous examples. A second 
                 degree polynomial was used to achieve reasonable fit results for this distribution.
                 ''')
        
        df_sorted = df_scaled.sort_values("radius_val",axis=0)
        x=df_sorted["radius_val"]
        y=df_sorted["BINDING ENERGY/A"]
        
        mod = PolynomialModel(degree=2)
        params = mod.make_params()
        params["c0"].set(value=-1)
        params["c1"].set(value=0)
        params["c2"].set(value=.1)
        
        result = mod.fit(y, params, x=x)
        
        x2 = np.linspace(-6,1.5,500)
        y_plot = result.eval(x=x2)

        fig = plt.figure(figsize=(6,6))
        plt.plot(x,y,"b.",label="Data")
        plt.plot(x2,y_plot,"r-",label="Best fit")
        plt.xlabel("Charge Radius (z-scaled)",fontsize="large")
        plt.ylabel("Mass excess (z-scaled)",fontsize="large")
        plt.legend(fontsize="large")
        plt.text(-4,-12.5,"R^2 = "+str(round(result.rsquared,4)))
        plt.text(-4,-15,"X^2 = "+str(round(result.chisqr,4)))
       
        st.pyplot(fig)
        st.caption('''
                   The best fit result of the binding energy data is plotted with a second degree 
                   polynomi best fit. The R^2 and X^2 fit statistics are provided on the figure and are 
                   worse than the mass data but still informitive.
                   ''')
                   
    st.write('''
             All four regression models show good relationships to the charge radius data. Although 
             the standard mass number model performs well,the atomic mass data slightly outperforms 
             this model. This is a logical improvement since mass number is a quick approximation of the 
             real atomic mass.
             
             Given all four models showed reasonable relationships, further improvements may be improved 
             by combining the inputs. A more advanced machine learning model will be useful to combine these features 
             and some of the categorical features discussed earlier. Random forest models will be applicable to this 
             situation and may achieve better results.
             ''')   
    
with tab4:
    st.write("Need to implement random forest model.")

st.markdown("### Conclusion")
st.write('''
         The nuclear charge radius is an important quantity in moder nuclear physics to benchmark 
         theories of the strong nuclear force. Here we explore the relationship of charge radius to 
         different nuclear observables. The standard model based on the mass number is shown to be accurate, 
         but improvement are found using realistic mass values. Additionally relationships were identified with 
         the mass excess, half life, and binding energy that can also be used to predict the charge radius.
         
         By combining all of these features into a PCA, it was shown that linear combinations have the 
         best effect in explaining variance. This suggests using a more advance machine learning model and a 
         random forest model is being implemented.
         ''')
