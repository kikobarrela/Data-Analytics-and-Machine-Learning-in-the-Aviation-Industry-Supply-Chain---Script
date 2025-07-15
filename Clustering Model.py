import pandas as pd
import numpy as np
import warnings
import pickle
import hashlib
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from foundry.transforms import Dataset
from yellowbrick.cluster import KElbowVisualizer
warnings.filterwarnings('ignore')
sns.set(style='whitegrid', rc={'figure.figsize':(12,8)})

df_raw = Dataset.get("po_header_and_item_restricted_24_25").read_table(format="pandas")

df = df_raw.copy()
numeric_cols_to_check = ['item_net_price', 'item_purchase_order_quantity', 'net_order_value', 'item_planned_delivery_time_in_days']
for col in numeric_cols_to_check:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where essential features for aggregation are missing
df.dropna(subset=['vendor_name', 'item_purchase_order_quantity', 'net_order_value'], inplace=True)
print(f"  > Initial data cleaned. Shape is now: {df.shape}")


df_cleaned = df.copy()
cols_to_drop_identifiers = [col for col in df_cleaned.columns if '_id' in col or '_number' in col]
df_cleaned.drop(columns=cols_to_drop_identifiers, inplace=True, errors='ignore')

missing_threshold = 0.90
high_missing_cols = [col for col in df_cleaned.columns if df_cleaned[col].isnull().sum() / len(df_cleaned) > missing_threshold]
df_cleaned.drop(columns=high_missing_cols, inplace=True, errors='ignore')

low_variance_cols = [col for col in df_cleaned.columns if df_cleaned[col].nunique(dropna=False) <= 1]
df_cleaned.drop(columns=low_variance_cols, inplace=True, errors='ignore')
print(f"  > Pruning complete. Shape after pruning: {df_cleaned.shape}")

# Anonymize 'vendor_name' to create a stable numerical ID for each supplier.
df_cleaned['vendor_id'] = df_cleaned['vendor_name'].astype('category').cat.codes
df_cleaned.drop(columns=['vendor_name'], inplace=True, errors='ignore')

low_cardinality_threshold = 50
categorical_cols_to_agg = [col for col in df_cleaned.select_dtypes(include='object').columns if df_cleaned[col].nunique() < low_cardinality_threshold]
agg_numeric_cols = {
    'net_order_value': ['sum', 'mean', 'max', 'std'],
    'item_purchase_order_quantity': ['sum', 'mean', 'std'],
    'item_planned_delivery_time_in_days': ['mean', 'std', 'max'],
}
agg_categorical_cols = {col: (lambda x: x.mode().iloc[0] if not x.mode().empty else None) for col in categorical_cols_to_agg}

agg_numeric_cols = {k: v for k, v in agg_numeric_cols.items() if k in df_cleaned.columns}
agg_categorical_cols = {k: v for k, v in agg_categorical_cols.items() if k in df_cleaned.columns}

df_supplier_level = df_cleaned.groupby('vendor_id').agg({**agg_numeric_cols, **agg_categorical_cols})
if isinstance(df_supplier_level.columns, pd.MultiIndex):
    df_supplier_level.columns = ['_'.join(col).strip() for col in df_supplier_level.columns.values]

rename_map = {
    'net_order_value_sum': 'total_spend', 'net_order_value_mean': 'avg_spend_per_item', 'net_order_value_max': 'max_spend_per_item', 'net_order_value_std': 'std_spend_per_item',
    'item_purchase_order_quantity_sum': 'total_quantity', 'item_purchase_order_quantity_mean': 'avg_quantity_per_item', 'item_purchase_order_quantity_std': 'std_quantity_per_item',
    'item_planned_delivery_time_in_days_mean': 'avg_delivery_time', 'item_planned_delivery_time_in_days_std': 'std_delivery_time', 'item_planned_delivery_time_in_days_max': 'max_delivery_time',
}
df_supplier_level.rename(columns=rename_map, inplace=True, errors='ignore')
df_supplier_level.fillna(0, inplace=True)
print(f"  > Aggregated data to supplier level. Shape is now: {df_supplier_level.shape}")


df_final = df_supplier_level.copy()
numerical_features = df_final.select_dtypes(include=np.number).columns.tolist()
for col in numerical_features:
    # Add a small constant to avoid log(0)
    if (df_final[col] >= 0).all():
         df_final[col] = np.log1p(df_final[col])
    else:
         print(f"  > Skipping log transform for '{col}' as it contains negative values.")

categorical_features = df_final.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_features:
    df_final[col] = df_final[col].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), numerical_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ],
    remainder='drop'
)
X_processed = preprocessor.fit_transform(df_final)
# Get feature names after preprocessing for interpretation
feature_names_processed = numerical_features + preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()


def display_pca_component_loadings(pca, feature_names, n_components=2, top_n=5):
    print("PCA Component Interpretation")
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=feature_names)
    for i in range(n_components):
        print(f"\n Top Features for Principal Component {i+1}")
        sorted_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        print(loadings.loc[sorted_loadings.index[:top_n], f'PC{i+1}'])
    return loadings

def interpret_umap_components(X_reduced_umap, df_original_features, top_n=5):

    print("UMAP Component Interpretation (Approximation)")

   
    umap_df = pd.DataFrame(X_reduced_umap, columns=['UMAP1', 'UMAP2'], index=df_original_features.index)
    correlation_df = df_original_features.join(umap_df).corr()

    print("\n Top Correlated Features for UMAP Component 1 ")
    print(correlation_df['UMAP1'].drop(['UMAP1', 'UMAP2']).abs().sort_values(ascending=False).head(top_n))

    print("\n Top Correlated Features for UMAP Component 2 ")
    print(correlation_df['UMAP2'].drop(['UMAP1', 'UMAP2']).abs().sort_values(ascending=False).head(top_n))


# PATH A: PCA + K-Means
pca = PCA(n_components=2, random_state=42)
X_reduced_pca = pca.fit_transform(X_processed)

display_pca_component_loadings(pca, feature_names_processed)

if KElbowVisualizer:
    model_kmeans = KMeans(random_state=49, n_init=10)
    k_max = min(16, df_final.shape[0]-1)
    visualizer = KElbowVisualizer(model_kmeans, k=(2, k_max), timings=False)
    visualizer.fit(X_reduced_pca)
    visualizer.show()
    n_clusters_kmeans = visualizer.elbow_value_ if visualizer.elbow_value_ else 4
else:
    n_clusters_kmeans = 4 
print(f"  > Selected K={n_clusters_kmeans} for K-Means.")

kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
pca_kmeans_labels = kmeans.fit_predict(X_reduced_pca)
df_pca_results = df_final.copy()
df_pca_results['cluster'] = pca_kmeans_labels

#PATH B: UMAP + HDBSCAN
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42)
X_reduced_umap = umap_reducer.fit_transform(X_processed)

interpret_umap_components(X_reduced_umap, df_final[numerical_features])

hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
umap_hdbscan_labels = hdbscan_clusterer.fit_predict(X_reduced_umap)
df_umap_results = df_final.copy()
df_umap_results['cluster'] = umap_hdbscan_labels

# Patch C: Hierarchical Clustering
agglomerative = AgglomerativeClustering(n_clusters=n_clusters_kmeans)
agg_labels = agglomerative.fit_predict(X_reduced_pca)
df_agg_results = df_final.copy()
df_agg_results['cluster'] = agg_labels


def profile_clusters(df_results, method_name, X_reduced):
    print(f"PROFILING RESULTS FOR: {method_name} ")
   
    # --- Visualization ---
    df_plot = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
    df_plot['cluster'] = df_results['cluster']
    plt.figure(figsize=(15, 10))
    unique_clusters = sorted(df_plot['cluster'].unique())
    palette = sns.color_palette('deep', n_colors=len(unique_clusters))
    color_map = {cluster: color for cluster, color in zip(unique_clusters, palette)}
    if -1 in color_map:
        color_map[-1] = (0.5, 0.5, 0.5) # Assign grey to noise

    sns.scatterplot(data=df_plot, x='Component 1', y='Component 2', hue='cluster', palette=color_map, s=50, alpha=0.9, legend='full')
    plt.title(f'Supplier Clusters ({method_name})', fontsize=18)
    plt.show()

    linked_data = linkage(X_reduced, method='ward')
    plt.figure(figsize=(25, 10))
    plt.title(f'Dendrogram for {method_name}')
    dendrogram(linked_data, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
    plt.show()

    analysis_df = df_results.copy()
    num_features_profile = [col for col in analysis_df.columns if col != 'cluster' and pd.api.types.is_numeric_dtype(analysis_df[col])]
    cat_features_profile = analysis_df.select_dtypes(include=['object', 'category']).columns.tolist()
    global_means = analysis_df[num_features_profile].mean()
    global_stds = analysis_df[num_features_profile].std()
   
    for cluster_id in sorted(analysis_df['cluster'].unique()):
        header = f"CLUSTER {cluster_id}"
        if cluster_id == -1: header = "NOISE / OUTLIERS"
        print(f"\n {header} ")
       
        cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]
        print(f"  Number of Suppliers: {len(cluster_data)} ({len(cluster_data) / len(analysis_df) * 100:.2f}%)")
       
        if num_features_profile:
            cluster_means = cluster_data[num_features_profile].mean()
            z_scores = (cluster_means - global_means) / global_stds
            try:
                display(z_scores.abs().sort_values(ascending=False).head(5).to_frame(name='Z-Score Magnitude'))
            except NameError:
                print(z_scores.abs().sort_values(ascending=False).head(5).to_frame(name='Z-Score Magnitude'))


# Profile each model's results
profile_clusters(df_pca_results, "PCA + K-Means", X_reduced_pca)
if not df_umap_results.empty:
    profile_clusters(df_umap_results, "UMAP + HDBSCAN", X_reduced_umap)
profile_clusters(df_agg_results, "Agglomerative Clustering", X_reduced_pca)
