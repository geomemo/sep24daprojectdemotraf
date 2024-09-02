
#Libaries
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import random
#@st.cache_data
#def generate_random_value(x): 
#  return random.uniform(0, x) 
#a = generate_random_value(10) 
#b = generate_random_value(20) 
#st.write(a) 
#st.write(b)
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN

    

# Load the CSV file into a DataFrame
df = pd.read_csv("PopTra_cluster_corx7.csv", delimiter=',')


image = Image.open("sidebar_image1.png")
st.sidebar.image(image, use_column_width=True)
st.sidebar.markdown("### Data Analyst DA_June24 Bootcamp")
#st.sidebar.markdown("### DEMOTRAFIC")
st.sidebar.markdown("M. Girtten")
#st.sidebar.markdown("(L. Zintel)")
st.title("Investigating Relationships Between Demographic Data and Traffic Data in Germany")
st.write("A Data-Driven Approach to Urban and Rural Transportation Analysis")

st.sidebar.title("Table of contents")
pages = ["00 Presentation", "01 Problem Statement", "02 Data Collection", "03 Overview and Visualization", "04 Data Cleaning and Preparation", "05 Data Exploration",  "06 Model Selection", "07 Model Analysis",  "08 Critical Appraisal", "09 Conclusion and Future Work", "10 last words"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:    # Presentation of Project")
    image = Image.open('Screenshot 2024-08-20 144249.png')
    st.write("## Presentation of Project")
    st.write("This project was undertaken by a Geospatial Analyst and GIS Expert from Germany, with extensive knowledge in demographic data and traffic planning. Initially focused on the VRR-Area in North Rhine-Westphalia (NRW), the analysis was expanded to cover a broader region across Germany to ensure a robust dataset for Machine Learning. The core problem addressed in this analysis is the challenge of understanding how population density correlates with vehicle density in both urban and rural settings. Such insights are crucial for optimizing traffic management, urban planning, and public transportation infrastructure.")
    st.image(image, caption='Screenshot, Population (Area Types) and Traffic (Vehicle Density Level) Map in PowerBI', use_column_width=True)
    
    

if page == pages[1]:    # Problem Statement
    st.write("## Problem Statement")
    
    st.write("""
            
    Urban areas in Germany and specially in the VRR-Area are experiencing increasing vehicle congestion, while rural areas may struggle with inadequate transportation infrastructure. Understanding the relationship between population density and vehicle density is crucial for developing targeted urban planning and traffic management strategies that can address these challenges effectively.         
    
    #### Expected Outcomes:
    The analysis will provide insights into how demographic factors influence vehicle distribution, helping urban planners and policymakers make informed decisions. This could lead to improved traffic flow in urban areas, optimized public transportation systems in rural regions, and overall better socio-economic outcomes by reducing congestion, lowering emissions, and enhancing the quality of life for residents. The findings could also support sustainable urban development and contribute to long-term environmental goals.         
    The expected outcomes of this project include identifying patterns and clusters that can inform policy decisions, such as where to invest in public transportation or how to manage traffic congestion more effectively. These insights could have significant socio-economic benefits, including improved quality of life, reduced environmental impact, and more efficient use of resources. By integrating advanced data analysis techniques, this project aims to provide actionable recommendations for urban planners, traffic managers, and policymakers.
    """)
    
    #image = Image.open('introduction.png')
    #st.image(image, caption='introduction', use_column_width=True)
       
    
    
if page == pages[2]:    # Data Collection
    st.write("## Data Collection")
    #st.write("Summary of  Data Collection")
    st.write("""
     #### Data Source and Raw Data Description
    - **Landesdatenbank NRW** ([landesdatenbank.nrw.de](https://www.landesdatenbank.nrw.de)): Provided detailed demographic and economic statistics specific to North Rhine-Westphalia (NRW).
    - **Regionalstatistik** ([regionalstatistik.de](https://www.regionalstatistik.de)): Offered comprehensive regional statistics across Germany, including traffic and population data.
    - **Statistikportal** ([statistikportal.de](https://www.statistikportal.de/en)): A central portal for accessing statistical data across German regions, crucial for cross-referencing demographic information.
    - **ArcGIS** ([arcgis.com](https://services2.arcgis.com/jUpNdisbWqRpMo35/arcgis/rest/services/Kreisgrenzen_2023/FeatureServer)): Supplied geospatial data on district boundaries across Germany, essential for spatial analysis.             
     
     
             
    #### The raw datasets included:
    - **Kreisgrenzen**: Geospatial data defining district boundaries.
    - **Car_18_20**: Traffic data capturing vehicle counts for 2018-2020 across districts.
    - **Pop_18_20**: Population data reflecting demographic trends during the same period.

    These datasets were integral to analyzing the relationship between population density and vehicle density, aiding in understanding urban and rural dynamics within NRW and later all district in Germany .     
             
    """)
    
    #image = Image.open('Screenshot 2024-08-20 202537.png')
    #st.image(image, caption='Data Source', use_column_width=True)
    
       
    
if page == pages[3]:    # Presentation of data
    st.write("## Presentation of Data")
    st.write("""
    #### Key Features Used:
    - **Population Density (Pop_Dens)**: Indicates the number of people per square kilometer, essential for understanding the concentration of population in urban vs. rural areas.
    - **Vehicle Density (Veh_Dens)**: Measures the number of vehicles per square kilometer, a critical indicator of traffic load in different regions.
    - **Vehicles per 1,000 People (Veh_1kpop)**: This ratio highlights vehicle ownership trends across different population densities.
    - **Total Population (Pop_Sum)**: Provides a measure of the overall population size in each district.
    - **Total Vehicles (Veh_Sum)**: Represents the aggregate number of vehicles in each district.
    
    """)
    
    
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())
        
    st.write("## Data Visualization")
    #st.write(" Summary of Data Visualization ")
    
    # Extract the relevant columns
    x = df['Pop_Dens']
    y = df['Veh_Dens']
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.xlabel('Population Density')
    plt.ylabel('Vehicle Density')
    plt.title('Scatter Plot of Population Density vs Vehicle Density')
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    st.write("The scatter plot shows a clear positive correlation between population density and vehicle density. As population density increases, vehicle density also tends to increase. This relationship makes sense because areas with higher population density are likely to have more vehicles per unit area due to higher demand for transportation.")

    # Add a new section in the Streamlit app for the pairplot visualization
    # if page == "Data Vizualization":
    # st.write("Pairwise Scatter Plots")

    # 1 "outcomment" the next lines for a pairplot and used an imange of the results (plots) instead-->2
    # 2 Reason: runing the code takes too much time in the streamlit app. 
    # variables_to_plot = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

    # Create pairwise scatter plots using seaborn
    # pairplot_fig = sns.pairplot(df[variables_to_plot])

    # Display the pairplot in Streamlit
    # st.pyplot(pairplot_fig)
    # st.write("The plots show clear relationships between total population and total vehicles, as well as between population density and vehicle density. However, the vehicles per 1,000 people (Veh_1kpop) metric does not strongly correlate with the density measures, suggesting that factors other than density alone affect vehicle ownership rates across different regions.")

    image = Image.open('pairPlot.png')
    st.image(image, caption='Pair Plot', use_column_width=True)
    st.write("The plots show clear relationships between total population and total vehicles, as well as between population density and vehicle density. However, the vehicles per 1,000 people (Veh_1kpop) metric does not strongly correlate with the density measures, suggesting that factors other than density alone affect vehicle ownership rates across different regions.")

    # Load the CSV file into a DataFrame
    data = pd.read_csv("PopTra_cluster_corx4.csv", delimiter=';', encoding='utf-8')

        # Select the relevant features for the heatmap
    features_for_heatmap = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

    # Calculate the correlation matrix
    corr_matrix = data[features_for_heatmap].corr()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')

    # Add a title
    plt.title('Correlation Heatmap of Population and Vehicle Metrics')

    # Display the heatmap in Streamlit
    st.write("Correlation Heatmap")
    st.pyplot(heatmap.figure)
    st.write("The heatmap shows strong positive correlations between population density and vehicle density (1.00), and between total population and total vehicles (0.97). This indicates that denser and larger populations generally lead to higher vehicle counts. Weak negative correlations appear between vehicles per 1000 people and total population (-0.32), suggesting that areas with larger populations tend to have fewer vehicles per capita. Near-zero correlations between population density and total population suggest that population size does not necessarily correspond to density, likely due to varying urban and rural patterns across regions.")


    st.write("""
    #### Additional Features
    **Area Type (AreaType)** The EU's Urban-Rural Typology classifies areas based on population density and size, using a population grid approach.

    1. Urban Areas: Defined as regions with a population density greater than 300 inhabitants per square kilometer.
    2. Intermediate Areas: These are areas with a population density between 100 and 300 inhabitants per square kilometer.
    3. Rural Areas: Defined as areas with a population density of less than 100 inhabitants per square kilometer.

    For further details, you can explore the EU's detailed explanation [here](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Urban-rural_typology).
    """)
    image = Image.open('AreaType_ger18-20.png')
    st.image(image, caption='Area Types in German Districts from 2018-2020 [geopandas, girtten, 08-24]', use_column_width=True)
    

               
    st.write("""
    The **Vehicle Density Level (VDL)**, categorizing vehicle population data into five levels: "Very Low," "Low," "Moderate," "High," and "Very High." This classification helps in assessing and comparing different regions or clusters based on their vehicle density, which can be useful for urban planning, traffic management, or analyzing transportation needs. By grouping vehicle data into these levels, it simplifies the analysis and makes it easier to identify areas with high or low vehicle density for targeted interventions.""")
    image = Image.open('VDL_DAX_Cat.png')
    st.image(image, caption='Vehicle Density Level with DAX', use_column_width=False)    
    
    image = Image.open('VDL_18-20.png')
    st.image(image, caption='VDL in German Districts from 2018-2020 [geopandas, girtten, 08-24]', use_column_width=True)

    image = Image.open('Veh_1kP.png')
    st.image(image, caption='Vehicle per 1k Pop in German Districts from 2018-2020 [geopandas, girtten, 08-24]', use_column_width=True)


    image = Image.open('PBI_1.gif')
    st.image(image, caption='Area Type and VDL in PowerBI, girtten, 09_24', use_column_width=False) 

if page == pages[4]:    # Cleaning and Preparation
    st.write("## Cleaning and Preparation")
    st.write("""
    #### Data Format and Structure:
    - **CSV Files**: The primary dataset was stored in CSV format, ensuring compatibility with a wide range of data processing tools.
    - **Preprocessing**: The data underwent extensive preprocessing, including handling missing values, standardization, and normalization, to prepare it for clustering and other ML methods.
    
    #### Data Processing and Integration for VRR-Region Analysis

    ##### Edit Phase 1: Initial Data Preparation

    In the initial phase, population and traffic data were standardized, normalized, and formatted, with missing values addressed to enable table merging. The data was filtered to focus on the VRR-Region in North Rhine-Westphalia, Germany, resulting in the "Pop-Traf-Table" containing 23 districts. This process was documented in the project proposal (June) and the Project Methodology Report (July). However, 23 districts were insufficient for machine learning applications.

    ##### Edit Phase 2: Data Expansion

    To overcome data limitations, the dataset was expanded to include all districts in Germany (over 400) and extended the years covered from 2018 to 2020. This required extensive resources to standardize, normalize, format, and handle missing values again, resulting in the "PopTraf_2018-20" table. This expansion resolved the row count issue but still lacked the necessary features.

    ##### Edit Phase 3: Feature Enhancement

    In the final phase, additional features were incorporated using geographical data for area calculations (Kreisgrenzen). Three new features were added: area in km², population density, and vehicle density per 1,000 people. The data was further encoded, normalized, and formatted, resulting in the "Pop_Tra_cluster" table, suitable for clustering machine learning methods.

    This comprehensive process ensured the dataset was robust and suitable for advanced analytical applications.
    
    
    """)
    

    image = Image.open('Flowchart_project.png')
    st.image(image, caption='Flowchart Data Preparing', use_column_width=True)
    
    
    
if page == pages[5]:    # Data Exploration
    st.write(" ## Data Exploration")
    st.write("This section standardizes the selected features in the dataset before applying PCA and K-Means Clustering.")

    # Display the original data
    st.write("### Original Data Sample:")
    st.dataframe(df.head())

    # Select the relevant features for PCA and Clustering
    features = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

    # Standardize the features before applying PCA and Clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Convert the scaled data back to a DataFrame for easier viewing
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    # Display the standardized data
    st.write("### Standardized Data Sample:")
    st.dataframe(scaled_df.head())

    # Reduce the data to 2 components for visualization using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

    # Optionally, add labels to the DataFrame for easier visualization
    pca_df['District_Name'] = df['District_Name']

    # Display the PCA results
    st.write("### PCA Results:")
    st.dataframe(pca_df.head())

    # Plot the principal components
    st.write("### PCA: 2D Visualization of Districts")
    
    #---Code replaced with imange, due to loading speed. 
    # plt.figure(figsize=(10, 7))
    # plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)

    # # Optionally, label each point with the district name (if the plot is not too crowded)
    # for i in range(pca_df.shape[0]):
    #     plt.text(pca_df['PC1'][i], pca_df['PC2'][i], pca_df['District_Name'][i], fontsize=9, alpha=0.7)

    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA: 2D Visualization of Districts')
    # plt.grid(True)

    # # Display the plot in Streamlit
    # st.pyplot(plt)
    
    image = Image.open('PCA_2d.png')
    st.image(image, caption='PCA 2D Plot', use_column_width=True)
    
    

    # K-Means Clustering: Apply the Elbow and Silhouette Methods to find the optimal number of clusters
    st.write("### Elbow and Silhouette Methods for K-Means Clustering")

# # Create two columns for the Elbow and Silhouette plots ---Code replaced with imange, due to loading speed. 
#     col1, col2 = st.columns(2)

#     # Elbow Plot
#     with col1:
#         # Calculate the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters
#         wcss = []
#         for i in range(1, 11):
#             kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#             kmeans.fit(scaled_data)
#             wcss.append(kmeans.inertia_)

#         # Plot the Elbow Method graph
#         plt.figure(figsize=(10, 7))
#         plt.plot(range(1, 11), wcss, marker='o')
#         plt.title('Elbow Method for Optimal Number of Clusters')
#         plt.xlabel('Number of Clusters')
#         plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
#         plt.grid(True)
        
#         # Display the Elbow Method plot in Streamlit
#         st.pyplot(plt)

#     # Silhouette Plot
#     with col2:
#         # Calculate the silhouette score for different numbers of clusters
#         silhouette_scores = []
#         range_n_clusters = range(2, 11)

#         for n_clusters in range_n_clusters:
#             kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
#             cluster_labels = kmeans.fit_predict(scaled_data)
#             silhouette_avg = silhouette_score(scaled_data, cluster_labels)
#             silhouette_scores.append(silhouette_avg)

#         # Plot the Silhouette Score for each number of clusters
#         plt.figure(figsize=(10, 7))
#         plt.plot(range_n_clusters, silhouette_scores, marker='o')
#         plt.title('Silhouette Score for Optimal Number of Clusters')
#         plt.xlabel('Number of Clusters')
#         plt.ylabel('Silhouette Score')
#         plt.grid(True)

#         # Display the Silhouette Score plot in Streamlit
#         st.pyplot(plt) 
        
    image = Image.open('Ellbow_silhouette.png')
    st.image(image, caption='Elbow and Silhouette Method', use_column_width=True)
    
    # Identify and display the optimal number of clusters
    #optimal_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
    #st.write(f'The optimal number of clusters based on the highest silhouette score is: {optimal_clusters}')
    st.write("Choosing 4 Clusters as indicated in the Elbow Method simplifies the model and reduces the risk of overfitting, while still capturing the major groupings in the data, but 2 clusters, as identified by the Silhouette Method, are the better choice for this project.")



#if page == pages[6]:    # Data Preparation Pipeline
#    st.write("## Data Preparation Pipeline")
#    st.write("Summary to --Data Preparation Pipeline--")
    
    

if page == pages[6]:    # Model Selection"
    st.write("## Model Selection")
    st.write("Comparison of Machine Learning Models for Traffic and Demographic Analysis")
    
    

# Title of the section
   # st.title("Comparison of Machine Learning Models for Traffic and Demographic Analysis")

# Introduction
    st.write("""
    In this project, various machine learning methods were considered to analyze the relationships between demographic data and traffic patterns across Germany. 
    The chosen method, **clustering**, was selected based on its ability to effectively group districts based on demographic and traffic-related features.
    Below, we compare the characteristics and outcomes of different models and discuss why clustering was ultimately chosen.
    Initially, regression models were considered for this project due to their ability to predict outcomes based on known target variables. However, different regression MLs were dropped due to organizational reasons.
             
    """)

# Table comparing different models
    st.subheader("Model Comparison")

# Define the comparison table
    st.markdown("""
    | **Model Type**    | **Description**                                       | **Pros** | **Cons** |
    |-------------------|-------------------------------------------------------|----------|----------|
    | **Clustering**    | Groups data based on similarities, Identifies natural groupings in data without predefined labels, Used to segment areas into urban/rural clusters | Captures complex, non-linear relationships; Flexible with different data types | Requires careful tuning of parameters like the number of clusters |
    | **Regression**    | Predicts a continuous outcome based on input features, Assumes a linear or non-linear relationship between input and output, Used for predicting traffic volumes, population density | Simple and interpretable; well-understood method | May not capture complex patterns in the data |
    | **Classification**| Categorizes data into predefined classes,  Assumes distinct classes with clear boundaries, Used for classifying areas as urban or rural | Works well when classes are well-defined; easy to implement | May struggle with overlapping classes or subtle differences |
    | **Decision Trees**| Uses tree-like models to make decisions based on input features,  Breaks down decision processes into a series of questions, Used for both classification and regression tasks | Easy to interpret and visualize; handles non-linear data | Prone to overfitting; can be unstable with small data changes |
    """)

# Explain the choice of clustering
    st.subheader("Why Clustering?")

    st.write("""
    The clustering method, particularly K-Means and DBSCAN, was selected for its effectiveness in capturing the complex structure of Germany’s demographic and traffic data. 
    Unlike regression and classification, which require predefined labels or assume specific relationships, clustering allows the discovery of natural groupings within the data.
    
    The clustering ML approach offered a sophisticated means of analyzing vehicle ownership in rural areas, uncovering natural groupings and complex patterns that conventional calculations might miss. This method provided actionable insights into how different rural areas vary in vehicle ownership, leading to more tailored policy recommendations. 
    The clustering method was adopted as an additional ML approach, partly because, unlike regression models, it does not require target variables. Clustering methods are particularly powerful for geospatial data, revealing spatial structures and patterns not immediately apparent. This approach was crucial for identifying distinct groups within the data and guiding policy and infrastructure development tailored to each rural area's unique characteristics. The choice of clustering ML enabled a more flexible and exploratory analysis, better suited to understanding the complexity of vehicle ownership patterns across Germany's rural regions.         
    """)

# Performance characteristics
    st.subheader("Performance Characteristics of Clustering")

    st.write("""
    - **Silhouette Score**: High, indicating well-defined clusters.
    - **Core vs. Border Points**: Effectively distinguished urban (core) and rural (border/noise) areas.
    - **Visualization Insights**:
    - **Cluster Plot**: Shows distinct clusters, clearly separating urban and rural areas.
    - **Silhouette Plot**: Confirms clustering quality with most points having high silhouette scores.
    """)

# Key Findings and Implications
    st.subheader("Key Findings and Implications")

    st.write("""
    - **Urban Areas**:
        - High population and vehicle density clusters.
        - Suggests need for targeted infrastructure improvements.
    - **Rural Areas**:
        - Lower density but higher vehicle ownership.
        - Highlights the need for better road infrastructure.
    - **Strategic Implications**:
        - Urban areas may require enhanced public transport.
        - Rural areas should focus on road network improvements.
        """)

    

if page == pages[7]:    # Model Analysis
    st.write("## Model Analysis")
    st.write("This section analyzes the clustering results using various clustering methods including K-Means, DBSCAN, and Hierarchical Clustering.")
    #st.subheader("K-Means Clustering")
    st.write("""
    **K-Means** helps identify distinct groups based on vehicle and population density, offering a clear overview of data structure. 
    This method is effective for partitioning data into well-separated clusters, making it ideal for categorizing regions with similar characteristics.
    """)

    # DBSCAN Explanation
    #st.subheader("DBSCAN Clustering")
    st.write("""
    **DBSCAN** is excellent for detecting clusters of varying shapes and densities, crucial for geospatial data. 
    It can also identify outliers like urban cores and sparsely populated areas, which are often critical in understanding regional traffic patterns.
    """)

    # Hierarchical Clustering Explanation
    #st.subheader("Hierarchical Clustering")
    st.write("""
    **Hierarchical Clustering** allows for exploring data at multiple levels, revealing the nested structure of rural and urban regions. 
    This method is useful for understanding the relationships between regions as you can visualize how clusters merge at different thresholds.
    """)

    # Conclusion
    st.write("""
    Together, these methods provide a thorough analysis of the demographic and traffic data, enabling targeted and strategic decisions for urban and rural planning across Germany.
    """)



    # Standardize the features again for use in this page
    features = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Define the optimal number of clusters directly for page 8 use
    optimal_clusters = 2

    # Run the KMeans algorithm again with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Group by the 'Cluster' column and compute the mean for each numeric column
    st.write("### Cluster Means of Numeric Features")
    numeric_columns = df.select_dtypes(include=[int, float]).columns
    cluster_means = df.groupby('Cluster')[numeric_columns].mean()

    # Display the result
    st.dataframe(cluster_means)

    # Now, integrate the provided K-Means clustering snippet

    # Specify the number of clusters
    k = 2

    # Initialize the K-Means model
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)

    # Fit the K-Means model to the data
    kmeans.fit(scaled_data)

    # You can now display or analyze the results of this clustering further
    st.write(f"K-Means Clustering with k={k}")
    df['Cluster'] = kmeans.labels_
    st.dataframe(df[['District_Name', 'Cluster']].head(6))

    # Reduce the data to 2 components for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    # Plot the clustered data
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis', marker='o')
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)
    st.write("The DBSCAN results show distinct clusters, likely separating urban areas from rural ones. Most districts form a tight cluster, indicating similar traffic patterns, while outliers represent regions with different demographics or traffic characteristics. This supports the project's goal of distinguishing urban and rural areas based on vehicle and population density, demonstrating DBSCAN's effectiveness in identifying regions with unique socio-economic profiles. These insights are valuable for targeted urban planning and traffic management strategies.")


# Get the cluster centers in the original feature space
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)

    # Display the cluster centers
    st.write("### Cluster Centers")
    st.dataframe(cluster_centers_df)

    # Group the original data by cluster and calculate summary statistics
    cluster_summary = df.groupby('Cluster')[features].mean()

    # Display the summary statistics for each cluster
    st.write("### Cluster Summary Statistics")
    st.dataframe(cluster_summary)

    # Plot box plots to compare the distribution of features between clusters
    st.write("### Comparison of Feature Distributions Across Clusters")
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='Cluster', y=feature, data=df)
        plt.title(f'Comparison of {feature} across Clusters')
    plt.tight_layout()

    # Display the box plots in Streamlit
    st.pyplot(plt)
    st.write("Cluster 1 --Urban Areas--:    High Population Density: Regions in this cluster have high population densities.     High Vehicle Density: These regions also show high vehicle densities.     Lower Vehicles per 1000 People: Despite high vehicle density, vehicle ownership per capita is lower, likely due to better public transportation options.     Higher Total Population and Vehicles: Reflecting the larger, more urbanized areas.")
    st.write("Cluster 0 --Rural Areas--:    Low Population Density: Regions in this cluster have very low population densities.     Low Vehicle Density: Vehicle density is also low.     Higher Vehicles per 1000 People: Rural areas show higher vehicle ownership per capita, possibly due to the necessity of personal vehicles in areas with limited public transportation.     Lower Total Population and Vehicles: These regions are less populated with fewer total vehicles.")
    st.write("#### Insights for Strategic Decisions: ")
    st.write("Urban Planning: For Cluster 1, focus on managing high population and vehicle densities, potentially by improving public transportation to reduce the need for personal vehicles. Rural Development: In Cluster 0, consider infrastructure improvements to support the higher per capita vehicle ownership, such as better road maintenance and access to services. Public Transportation: The difference in vehicle ownership per capita between the clusters suggests a need to tailor public transportation solutions based on regional characteristics ")


    st.title("DBSCAN Clustering Analysis")
    st.write("This section analyzes the clustering results using the DBSCAN method.")

    # Load the dataset
    df = pd.read_csv("PopTra_cluster_corx4.csv", delimiter=';', encoding='utf-8')

    # Select the relevant features for clustering
    features = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Set the DBSCAN parameters
    eps = 0.5  # Adjust based on your data
    min_samples = 5  # Adjust based on your data

    # Initialize DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model
    df['Cluster'] = dbscan.fit_predict(scaled_data)

    # Display the first few rows with the cluster assignments
    st.write("### DBSCAN Cluster Assignments")
    st.dataframe(df.head())

    # Count the number of points in each cluster
    cluster_counts = df['Cluster'].value_counts()

    # Identify the number of outliers
    outliers_count = (df['Cluster'] == -1).sum()

    st.write("### Cluster Counts")
    st.write(cluster_counts)

    st.write(f"### Number of Outliers: {outliers_count}")

    # Reduce the data to 2 components for visualization using PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    # Plot the clustered data
    st.write("### DBSCAN Clustering Results")
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis', marker='o')
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)
    st.write("The DBSCAN clustering plot shows distinct groups of districts, likely separating urban areas (denser clusters) from rural ones (more dispersed points). Some outliers represent regions with unique demographic or traffic patterns, underscoring DBSCAN's ability to identify non-spherical clusters and noise, valuable for targeted regional planning and traffic management.")

    # Add Hierarchical Clustering
    st.write("### Hierarchical Clustering and Dendrogram")

    # Perform hierarchical/agglomerative clustering
    Z = linkage(scaled_data, method='ward')

    # # Create a dendrogram to visualize the clustering  ---replaced code with imange, due to imporve load speed
    # plt.figure(figsize=(12, 7))
    # dendrogram(Z, labels=df.index, leaf_rotation=90, leaf_font_size=10)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample index')
    # plt.ylabel('Distance')

    # # Display the dendrogram plot in Streamlit
    # st.pyplot(plt)
    
    image = Image.open('dendrogram.png')
    st.image(image, caption='Dendrogram', use_column_width=True)

    # Choose a distance threshold or number of clusters
    max_d = 5  # You can adjust this based on the dendrogram
    clusters = fcluster(Z, max_d, criterion='distance')

    # Add the cluster labels to your dataset
    df['Cluster'] = clusters

    # Display the first few rows with the cluster assignments
    st.write("### Cluster Assignments Based on Dendrogram")
    st.dataframe(df.head())
    st.write("The dendrogram shows hierarchical clustering of districts based on demographic and vehicle density features. Three main clusters emerge, with significant differences between them. The blue cluster is distinct from the green, indicating a strong separation, likely between urban and rural areas. The green and red clusters further differentiate within these categories, reflecting varying levels of urbanization or vehicle density.")



# if page == pages[8]:    # Data Visualizationt
#     st.write("## Data Visualization")
#     #st.write(" Summary of Data Visualization ")
    
#     # Extract the relevant columns
#     x = df['Pop_Dens']
#     y = df['Veh_Dens']
    
#     # Create the scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y)
#     plt.xlabel('Population Density')
#     plt.ylabel('Vehicle Density')
#     plt.title('Scatter Plot of Population Density vs Vehicle Density')
    
#     # Display the plot in Streamlit
#     st.pyplot(plt)
#     st.write("The scatter plot shows a clear positive correlation between population density and vehicle density. As population density increases, vehicle density also tends to increase. This relationship makes sense because areas with higher population density are likely to have more vehicles per unit area due to higher demand for transportation.")

# # Add a new section in the Streamlit app for the pairplot visualization
# #if page == "Data Vizualization":
# #    st.write("Pairwise Scatter Plots")

#     #1 "outcomment" the next lines for a pairplot and used an imange of the results (plots) instead-->2
#     #2 Reason: runing the code takes too much time in the streamlit app. 
#     #variables_to_plot = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

#     # Create pairwise scatter plots using seaborn
#     #pairplot_fig = sns.pairplot(df[variables_to_plot])

#     # Display the pairplot in Streamlit
#     #st.pyplot(pairplot_fig)
#     #st.write("The plots show clear relationships between total population and total vehicles, as well as between population density and vehicle density. However, the vehicles per 1,000 people (Veh_1kpop) metric does not strongly correlate with the density measures, suggesting that factors other than density alone affect vehicle ownership rates across different regions.")

#     image = Image.open('pairPlot.png')
#     st.image(image, caption='Pair Plot', use_column_width=True)
#     st.write("The plots show clear relationships between total population and total vehicles, as well as between population density and vehicle density. However, the vehicles per 1,000 people (Veh_1kpop) metric does not strongly correlate with the density measures, suggesting that factors other than density alone affect vehicle ownership rates across different regions.")

#     # Load the CSV file into a DataFrame
#     data = pd.read_csv("PopTra_cluster_corx4.csv", delimiter=';', encoding='utf-8')

#         # Select the relevant features for the heatmap
#     features_for_heatmap = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

#     # Calculate the correlation matrix
#     corr_matrix = data[features_for_heatmap].corr()

#     # Create a heatmap
#     plt.figure(figsize=(10, 8))
#     heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')

#     # Add a title
#     plt.title('Correlation Heatmap of Population and Vehicle Metrics')

#     # Display the heatmap in Streamlit
#     st.write("Correlation Heatmap")
#     st.pyplot(heatmap.figure)
#     st.write("The heatmap shows strong positive correlations between population density and vehicle density (1.00), and between total population and total vehicles (0.97). This indicates that denser and larger populations generally lead to higher vehicle counts. Weak negative correlations appear between vehicles per 1000 people and total population (-0.32), suggesting that areas with larger populations tend to have fewer vehicles per capita. Near-zero correlations between population density and total population suggest that population size does not necessarily correspond to density, likely due to varying urban and rural patterns across regions.")



if page == pages[8]:   # Critical Appraisal"
    st.write("## Critical Appraisal")
    st.write("""
    #### Strengths:
    DBSCAN Success: Effectively identified meaningful clusters, handling non-spherical shapes and outliers.
    Actionable Insights: Provided valuable guidance for urban and rural planning.
    #### Limitations:
    Data: Limited by the availability and granularity of data; some regions may have incomplete information.
    Assumptions: Uniformity in vehicle and population density assumed; might oversimplify dynamics.
    Model: DBSCAN required careful tuning; may not generalize across all datasets.
    #### Future Perspectives:
    More Data: Incorporate economic indicators, public transport data, and real-time traffic data.
    Advanced Models: Explore hybrid models or more sophisticated clustering techniques.
    Temporal Analysis: Investigate how patterns evolve over time for long-term insights.
    #### Preemptive Considerations:
    Scope and Constraints: Focused on Germany with available data; broader application may need adjustments.
    Data Quality: Addressed within the project's scope; acknowledged limitations.
    """)
    
    
    
if page == pages[9]:   # Conclusion and Future Work
    st.write("## Conclusion and Future Work")
    st.write("""
    #### Key Findings and Implications:
    Urban Planning: High vehicle and population density clusters in Germany's urban areas suggest the need for enhanced public transportation and traffic management.
    Rural Planning: Lower density areas indicate a focus on maintaining accessibility and optimizing road infrastructure for vehicle use.
    Policy Implications: Targeted infrastructure investments based on regional needs.
    As the project aimed to explore the intricate relationships between demographic factors and traffic data across Germany, the following enhancements could have been implemented with additional time and resources:""")

     
    st.write("""
    **Inclusion of Additional Demographic Variables:** Expanding the scope of demographic factors would provide a more comprehensive analysis of traffic patterns. Potential additions could include socio-economic indicators such as income levels, employment rates, age distribution, and household sizes. These variables would help in understanding how economic and social factors influence vehicle ownership and usage across different regions.
    """)
    st.write("""
    **Longer and More Recent Time Series:** The current analysis was limited to data from 2018-2020. Extending the timeline to include more recent data would provide insights into trends and changes over time, especially considering the impact of significant events like the COVID-19 pandemic on traffic patterns. A broader historical perspective would also help in identifying long-term shifts in demographic and traffic data.        
    """)
    st.write("""
    **Incorporation of Comparative Data from Other Regions:** Expanding the analysis to include data from other European regions or global counterparts would allow for a comparative study. This would provide context for Germany’s traffic and demographic trends, helping to identify unique factors and commonalities on a broader scale.
    """)
    st.write("""
    **Disaggregation of Vehicle Types:** The current project aggregated all vehicle types into a single category. A more detailed analysis that differentiates between various vehicle types (e.g., cars, trucks, motorcycles) could uncover specific patterns in ownership and usage. This granularity would provide more actionable insights, particularly in regions where certain vehicle types dominate.
    """)
    st.write("""
    **Enhanced Mapping and Geospatial Tools:** Although some geospatial analysis was conducted using GeoPandas and Power BI, a more detailed and interactive approach could be adopted. Tools like Folium, ArcPy, and more advanced Power BI features would enable the creation of dynamic maps that better visualize the spatial relationships between demographic and traffic data. These tools would allow stakeholders to explore the data more interactively, facilitating better decision-making.        
    """)
    st.write("""
    **Application of Multiple Machine Learning Methods:** The project primarily focused on clustering methods (K-Means, DBSCAN, Hierarchical Clustering). Given more time, incorporating additional ML techniques such as regression models, decision trees, and neural networks would allow for a comparative analysis of the results. This would provide a more robust understanding of the data and validate the findings across different modeling approaches.
    """)         
    st.write("""        
    #### Significance of the Project:
    Informs Policy Decisions: Provides critical insights for designing effective urban and rural interventions.
    Data-Driven Planning: Demonstrates the power of data analytics in uncovering actionable patterns for strategic planning.
    """)
    # ### Forward-Looking Statement:
    # Future Work: Incorporate additional data (economic, environmental) to further refine clusters.
    # Broader Impact: Contributes to regional planning initiatives, offering a scalable model for data-driven developm

    # """)


if page == pages[10]:   # Conclusion and Future Work
    st.write("## Some comments")
    st.write("1 - Own Topic/Project")
    st.write("2 - Teamwork")
    st.write("3 - Communication")
    st.write("4 - Time Managemet")
