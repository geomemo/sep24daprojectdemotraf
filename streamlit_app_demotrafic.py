# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import base64
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN

# Load the CSV file into a DataFrame
df = pd.read_csv("PopTra_cluster_corx7.csv", delimiter=',')

sidebar_image = Image.open("sidebar_image1.png")  
# Changed to use_container_width
st.sidebar.image(sidebar_image, use_container_width=True)
st.sidebar.markdown("**Data Analyst DA_June24 Bootcamp**")
st.sidebar.markdown("M. Girtten - Geography M.Sc.<br>Lars Zintel - Economics M.Sc.", unsafe_allow_html=True) 
st.sidebar.markdown("Friday 13th, September 2024") 
st.title("Investigating Relationships Between Demographic Data and Traffic Data in Germany")
st.write("A Data-Driven Approach to Urban and Rural Traffic Analysis")

st.sidebar.title("Table of contents")
pages = [
    "👋 Introduction", 
    "📈 Preparation and Presentation", 
    "⚙️ Data Exploration and Analysis", 
    "📌 Conclusion and Perspectives"
]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:    # Introduction
    
    intro_image = Image.open('Screenshot 2024-08-20 144249.png')
    # Changed to use_container_width
    st.image(intro_image, caption='Screenshot, Population (Area Types) and Traffic (Vehicle Density Level) Map in PowerBI', use_container_width=True)

    # Introduction text
    st.write("## Introduction")
    st.write("""
        This project was undertaken by a Geospatial Analyst and GIS Expert from Germany, with extensive knowledge in demographic data and traffic planning. Initially focused on the VRR-Area in North Rhine-Westphalia (NRW) with 23 Districts, the analysis was expanded to cover a broader region across Germany (about 400 Districts) to ensure a robust dataset for Machine Learning. The core problem addressed in this analysis is the challenge of understanding how population density correlates with vehicle density in both urban and rural settings. Such insights are crucial for optimizing traffic management, urban planning, and public mobility infrastructure.

        Unlike typical Data-Analyst projects that primarily address business problems, this analysis is uniquely focused on a **socio-economic** issue. The project emphasizes the interplay between population density and vehicle density, aiming to derive insights that are pivotal not just for business strategy but for societal welfare and policy-making. By exploring these correlations, the project seeks to contribute to the broader understanding of urban and rural development, with a view toward enhancing public mobility infrastructure and improving overall quality of life.    
             """)

    st.write("## Problem Statement")
    st.write("""
    Urban areas in Germany, especially in the VRR-Area, are experiencing increasing vehicle congestion, while rural areas may struggle with inadequate transportation infrastructure. Understanding the relationship between population density and vehicle density is crucial for developing targeted urban planning and traffic management strategies that can address these challenges effectively.         

    #### Expected Outcomes:
    The analysis will provide insights into how demographic factors influence vehicle distribution, helping urban planners and policymakers make informed decisions. This could lead to improved traffic flow in urban areas, optimized public transportation systems in rural regions, and overall better socio-economic outcomes by reducing congestion, lowering emissions, and enhancing the quality of life for residents. The findings could also support sustainable urban development and contribute to long-term environmental goals.         
    The expected outcomes of this project include identifying patterns and clusters that can inform policy decisions, such as where to invest in public transportation or how to manage traffic congestion more effectively. These insights could have significant socio-economic benefits, including improved quality of life, reduced environmental impact, and more efficient use of resources. By integrating advanced data analysis techniques, this project aims to provide actionable recommendations for urban planners, traffic managers, and policymakers.
    """)

if page == pages[1]:    # Preparation and Presentation
    st.write("## Preparation and Presentation")
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

    These datasets were integral to analyzing the relationship between population density and vehicle density, aiding in understanding urban and rural dynamics within NRW and later all district in Germany.
    """)

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
    pairplot_image = Image.open('pairPlot.png')
    # Changed to use_container_width
    st.image(pairplot_image, caption='Pair Plot', use_container_width=True)
    st.write("The plots show clear relationships between total population and total vehicles, as well as between population density and vehicle density. However, the vehicles per 1,000 people (Veh_1kpop) metric does not strongly correlate with the density measures, suggesting that factors other than density alone affect vehicle ownership rates across different regions.")

    data = pd.read_csv("PopTra_cluster_corx7.csv", delimiter=',', encoding='utf-8')
    features_for_heatmap = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']
    corr_matrix = data[features_for_heatmap].corr()

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
    plt.title('Correlation Heatmap of Population and Vehicle Metrics')
    st.write("Correlation Heatmap")
    st.pyplot(heatmap.figure)
    st.write("The heatmap shows strong positive correlations between population density and vehicle density (1.00), and between total population and total vehicles (0.97). ...")
    st.write("*Veh_1kpop* differs from density metrics ...")

    st.write("""
    #### Additional Features
    **Area Type (AreaType)** The EU's Urban-Rural Typology classifies areas based on population density and size, using a population grid approach.

    1. Urban Areas: ...
    2. Intermediate Areas: ...
    3. Rural Areas: ...
    """)

    # Extract the relevant columns for the pie charts
    vdl_counts = df['VDL'].value_counts()
    area_type_counts = df['AreaType'].value_counts()

    area_type_colors = {
        'Urban': '#1f77b4',
        'Intermediate': '#ff7f0e',
        'Rural': '#2ca02c',
    }

    vdl_colors = {
        'Very Low': '#1f77b4',
        'Low': '#aec7e8',
        'Moderate': '#ffbb78',
        'High': '#ff7f0e',
        'Very High': '#d62728',
    }

    st.write("**Vehicle Density Level (VDL) and Area Type Distribution**")

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].pie(area_type_counts, labels=area_type_counts.index, autopct='%1.1f%%', startangle=140, 
              colors=[area_type_colors[key] for key in area_type_counts.index])
    ax[0].set_title('Area Type Distribution')

    ax[1].pie(vdl_counts, labels=vdl_counts.index, autopct='%1.1f%%', startangle=140, 
              colors=[vdl_colors[key] for key in vdl_counts.index])
    ax[1].set_title('Vehicle Density Level (VDL) Distribution')

    plt.tight_layout()
    st.pyplot(fig)

    area_type_image = Image.open('AreaType_ger18-20.png')
    # Changed to use_container_width
    st.image(area_type_image, caption='Area Types in German Districts from 2018-2020 [geopandas, girtten, 08-24]', use_container_width=True)

    vdl_districts_image = Image.open('VDL_18-20.png')
    # Changed to use_container_width
    st.image(vdl_districts_image, caption='VDL in German Districts from 2018-2020 [geopandas, girtten, 08-24]', use_container_width=True)

    vdl_image = Image.open('VDL_DAX_Cat.png')
    # Changed to use_container_width
    st.image(vdl_image, caption='Vehicle Density Level with DAX', use_container_width=True)

    st.write("""
    The **Vehicle Density Level (VDL)** categorizes vehicle population data into five levels: "Very Low," "Low," "Moderate," "High," and "Very High." ...
    """)

    veh_per_1k_pop_image = Image.open('Veh_1kP.png')
    # Changed to use_container_width
    st.image(veh_per_1k_pop_image, caption='Vehicle per 1k Pop in German Districts from 2018-2020 [geopandas, girtten, 08-24]', use_container_width=True)

    st.write("**Comparison Population and Traffic in Germany with PowerBI**")

    def load_gif(file_path):
        with open(file_path, "rb") as f:
            contents = f.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        return data_url

    gif_path = "PBI_1.gif"
    data_url = load_gif(gif_path)
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True)

    st.write("## Cleaning and Preparation")
    st.write("""
    #### Data Format and Structure:
    - **CSV Files**: The primary dataset was stored in CSV format...
    """)

    flowchart_image = Image.open('Flowchart_project1.png')
    # Changed to use_container_width
    st.image(flowchart_image, caption='Flowchart Data Preparing', use_container_width=True)

if page == pages[2]:    # Data Exploration
    st.write("## Data Exploration")
    st.write("This section standardizes the selected features in the dataset before applying PCA and K-Means Clustering.")

    st.write("### Original Data Sample:")
    st.dataframe(df.head())

    features = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    scaled_df = pd.DataFrame(scaled_data, columns=features)
    st.write("### Standardized Data Sample:")
    st.dataframe(scaled_df.head())

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    pca_df['District_Name'] = df['District_Name']

    st.write("### PCA Results:")
    st.dataframe(pca_df.head())

    st.write("### PCA: 2D Visualization of Districts")
    pca_image = Image.open('PCA_2D_visualization.png')
    # Changed to use_container_width
    st.image(pca_image, caption='PCA 2D Plot', use_container_width=True)

    st.write("### Elbow and Silhouette Methods for K-Means Clustering")
    elbow_silhouette_image = Image.open('Ellbow_silhouette.png')
    # Changed to use_container_width
    st.image(elbow_silhouette_image, caption='Elbow and Silhouette Method', use_container_width=True)

    st.write("Choosing 4 Clusters ... but 2 clusters ...")

    st.write("## Model Selection")
    st.write("Comparison of Machine Learning Models ...")

    st.subheader("Model Comparison")
    st.markdown("""
    | **Model Type**    | **Description**                                       | **Pros** | **Cons** |
    |-------------------|-------------------------------------------------------|----------|----------|
    | **Clustering**    | Groups data based on similarities...                 | Captures complex...  | Requires careful tuning... |
    | **Regression**    | Predicts a continuous outcome...                     | Simple...            | May not capture complex... |
    | **Classification**| Categorizes data into predefined classes...          | Works well...        | May struggle with overlapping... |
    | **Decision Trees**| Uses tree-like models...                              | Easy to interpret... | Prone to overfitting...   |
    """)

    st.subheader("Why Clustering?")
    st.write("""
    The clustering method ... 
    """)

    st.subheader("Performance Characteristics of Clustering")
    st.write("""
    - **Silhouette Score**: ...
    """)

    st.subheader("Key Findings and Implications")
    st.write("""
    - **Urban Areas**: ...
    """)

    st.write("## Model Analysis")
    st.write("This section analyzes the clustering results...")

    features = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    optimal_clusters = 2
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    st.write("### Cluster Means of Numeric Features")
    numeric_columns = df.select_dtypes(include=[int, float]).columns
    cluster_means = df.groupby('Cluster')[numeric_columns].mean()
    st.dataframe(cluster_means)
    st.write("The values indicate that Cluster 0 represents areas with significantly larger area sizes...")

    st.write(f"K-Means Clustering with k={optimal_clusters}")
    df['Cluster'] = kmeans.labels_
    st.dataframe(df[['District_Name', 'Cluster']].head(6))

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 7))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis', marker='o')
    plt.title('K-Means Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    st.pyplot(plt)

    st.write("The K-means clustering result...")

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)

    st.write("### Cluster Centers")
    st.dataframe(cluster_centers_df)

    cluster_summary = df.groupby('Cluster')[features].mean()
    st.write("### Cluster Summary Statistics")
    st.dataframe(cluster_summary)

    st.write("### Comparison of Feature Distributions Across Clusters")
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='Cluster', y=feature, data=df)
        plt.title(f'Comparison of {feature} across Clusters')
    plt.tight_layout()
    st.pyplot(plt)

    st.write("Cluster 1 --Urban Areas--: ...")
    st.write("Cluster 0 --Rural Areas--: ...")

    spectral_image = Image.open('spectral.png')
    # Changed to use_container_width
    st.image(spectral_image, caption='Spectral', use_container_width=True)

    mean_image = Image.open('mean.png')
    # Changed to use_container_width
    st.image(mean_image, caption='mean', use_container_width=True)

    st.title("DBSCAN Clustering Analysis")
    st.write("This section analyzes the clustering results using the DBSCAN method.")

    df = pd.read_csv("PopTra_cluster_corx7.csv", delimiter=',', encoding='utf-8')
    features = ['Pop_Dens', 'Veh_Dens', 'Veh_1kpop', 'Pop_Sum', 'Veh_Sum']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    eps = 0.5
    min_samples = 5
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    df['Cluster'] = dbscan.fit_predict(scaled_data)
    st.write("### DBSCAN Cluster Assignments")
    st.dataframe(df.head())

    cluster_counts = df['Cluster'].value_counts()
    outliers_count = (df['Cluster'] == -1).sum()

    st.write("### Cluster Counts")
    st.write(cluster_counts)
    st.write(f"### Number of Outliers: {outliers_count}")

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    st.write("### DBSCAN Clustering Results")
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis', marker='o')
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    st.pyplot(plt)

    st.write("The DBSCAN clustering plot shows distinct groups...")

    st.write("### Hierarchical Clustering and Dendrogram")
    Z = linkage(scaled_data, method='ward')

    dendrogram_image = Image.open('dendrogram.png')
    # Changed to use_container_width
    st.image(dendrogram_image, caption='Dendrogram 1', use_container_width=True)

    dendrogram_image = Image.open('Dendrogram_1.png')
    # Changed to use_container_width
    st.image(dendrogram_image, caption='Dendrogram 2', use_container_width=True)

    max_d = 5
    clusters = fcluster(Z, max_d, criterion='distance')
    df['Cluster'] = clusters
    st.write("### Cluster Assignments Based on Dendrogram")
    st.dataframe(df.head())
    st.write("The dendrogram shows hierarchical clustering...")

if page == pages[3]:    # Conclusion and Perspectives
    st.write("## Conclusion and Perspectives")
    overview_image = Image.open('Overview_Predict_result.png')
    # Changed to use_container_width
    st.image(overview_image, caption='Overview Prediction and Results', use_container_width=True)
    st.write("The table reflects that the ML model's results...")

    st.write("**Together, the heatmap and K-means clustering confirm that rural areas...**")
    st.write("""
    #### Strengths:
    DBSCAN Success...
    #### Limitations:
    Data...
    #### Future Perspectives:
    More Data...
    Advanced Models...
    Temporal Analysis...
    """)

    st.write("## Conclusion and Future Work")
    st.write("""
    #### Key Findings and Implications:
    Urban Planning...
    Rural Planning...
    Policy Implications...
    """)

    st.write("## Some comments")
    st.write("1 - Own Topic/Project")
    st.write("2 - Teamwork")
    st.write("3 - Communication")
    st.write("4 - Time Management")

    if st.checkbox("It still applies ..."):
        st.write('*"No obligation of results, but an obligation of effort."*   DataScientest')
        st.write(' *"[...] there`s no an expected outcome of the project. We really want you to give your best effort to try as much things as you can [...] "*   Lucas Varela, 10.06.2024 ')
