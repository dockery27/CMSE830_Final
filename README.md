# CMSE830_Final
Final project for CMSE 830

In this project, I am exploring the connection between the nuclear charge radius and other properties of the nucleus. Measurements of nuclear charge radii are the focus of my PhD thesis project. The nuclear charge radius provides information about the proton distribution in the nucleus. Throughout the nuclear chart, the charge radius can be modeled by a simple power law forumula, but there are significant local deviations from this trend. Here, other nuclear observables are modeled against the charge radius to find an improved prediction. Regression is tried on several parameters and a random forest is implemented to predict on all features included in the dataset.

I gathered data from international atomic energy agency, which maintains several tabels of nuclear data. I used the charge radius (1), mass (2), and lifetime (3) tables which are linked below and included in the github.

https://www-nds.iaea.org/radii/; charge_radii.xlsx
https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt; nuclear_mass.xlsx
https://www-nds.iaea.org/relnsd/NdsEnsdf/QueryForm.html; lifetime.csv
I combined these tables into a single cleaned data set with data_cleaning.ipynb program, and that data is saved as combined_data.csv. I generated visualizations from the eda_visualizaiton.ipynb, and encoprporated many of these visualizations in the trends and correlations section of the streamlit app. For the PCA and regression, the pca_fitting.ipynb contains the fits that were incorporated in the app. Last, the random_forest.ipynb file was used to generate the random forest model. The streamlit app is run from the overview.py file, and the additional pages to the app are contained in the pages folder. The pages are 1_Production.py which is the final product website, and 2_Data Science.py which includes the technical information. In order to run the notebooks, the following packages are needed:

numpy
pandas
matplotlib
scikit-learn
seaborn
altair
lmfit
