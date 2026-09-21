'''
Clustering Tabular Data: 

Agglomerative Hierarchiel Consensus Clustering

'''

import pandas as pd
from datetime import datetime
import os
import numpy as np
from scipy.io import loadmat
import glob
import json
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from umap import UMAP
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
    
from sklearn.model_selection import train_test_split
import xgboost as xgb
import scipy

if not hasattr(scipy, "interp"):
    scipy.interp = np.interp  # compatibility fix: adding a fallback scipy.interp reference to NumPy’s version

import scikitplot as skplt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from tc_consensus_clust import *
from collections import Counter
import shap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_keep = None

    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_keep = [column for column in X.columns if not any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        return X[self.to_keep]



def preprocessing_pipeline(X, 
                           standard_minmax=None,   # 'standard', 'minmax', or None
                           var_thresh=None,        # float or None
                           corr_thresh=None):      # float or None
    '''
    X: A pandas DataFrame / numpy array containing your input features (samples × features).
    standard_minmax: feature normalization in 'standard', 'minmax', or None (default: None)
    var_thresh: Threshold for removing low-variance features (default: None, typical: 0.025). Removes features with <var_thresh
    corr_thresh: Threshold for removing highly correlated features (default: None, typical: 0.9).Removes features with >corr_thresh
    '''
    
    is_dataframe = isinstance(X, pd.DataFrame)

    if not is_dataframe:
        X = pd.DataFrame(X, columns=np.arange(X.shape[1]).astype(str))

    X_processed = X.copy()

    # Step 0: Optional Scaling
    if standard_minmax == 'standard':
        scaler = StandardScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed),
                                   columns=X.columns,
                                   index=X.index)
    elif standard_minmax == 'minmax':
        scaler = MinMaxScaler()
        X_processed = pd.DataFrame(scaler.fit_transform(X_processed),
                                   columns=X.columns,
                                   index=X.index)

    # Step 1: Optional Variance Thresholding
    if var_thresh is not None:
        sel = VarianceThreshold(threshold=var_thresh)
        X_processed = pd.DataFrame(sel.fit_transform(X_processed),
                                   columns=X_processed.columns[sel.get_support()],
                                   index=X.index)

    # Step 2: Optional Correlation Filtering
    if corr_thresh is not None:
        corr_filter = CorrelationFilter(threshold=corr_thresh)
        corr_filter.fit(X_processed)
        X_processed = corr_filter.transform(X_processed)

    return X_processed



def tc_cluster_CC(df_data, feature_columns, standard_minmax, var_thresh, corr_thresh, verbose, dim_reduce_method, embedding_type, k_opt=4):
    '''
    df_data: input data dataframe
    feature_columns: list of feature names for analysis
    var_thresh: VIF variance threshold (var_thresh=0.01)
    verbose: display flag (default:0)
    dim_reduce_method: Dimensionality reduction method (default:None - no dim reduction): 'pca', 'umap', 'custom_embedding'
    embedding_type: name of the method created embedding. (dafault: None): 'None', 'rbm'
    
    '''
    df = df_data.copy()
    feat_data_all = df_data[feature_columns]   
    feat_data = preprocessing_pipeline(feat_data_all, standard_minmax=standard_minmax, var_thresh=var_thresh, corr_thresh=corr_thresh) 
    feature_list = list(feat_data.columns)
    umap_2d_standard = UMAP(n_components=2, init='spectral', random_state=10, n_neighbors=30)#min_dist=.25)
    proj_2d_standard = umap_2d_standard.fit_transform(feat_data)
    if dim_reduce_method == 'umap':
        print('UMAP Method')
        con_clust_labels = tc_consensus_clustering(proj_2d_standard, k_opt=k_opt, show=verbose)
    elif dim_reduce_method == 'pca':
        print('PCA Method')
        pca = PCA(n_components = 0.90)
        proj_pca90_standard = pca.fit_transform(feat_data)
        con_clust_labels = tc_consensus_clustering(proj_pca90_standard, k_opt=k_opt, show=verbose)
    elif dim_reduce_method == 'pca_umap':
        print('PCA Method')
        pca = PCA(n_components = 0.90)
        proj_pca90_standard = pca.fit_transform(feat_data)
        umap_2d_pca = UMAP(n_components=2, init='spectral', random_state=10, n_neighbors=30)#min_dist=.25)
        proj_2d_pca = umap_2d_pca.fit_transform(proj_pca90_standard)
        con_clust_labels = tc_consensus_clustering(proj_2d_pca, k_opt=k_opt, show=verbose)
    elif (dim_reduce_method == 'custom_embedding') and (embedding_type=='vae'):
        print('Custom Embedding')
        embeddings,_,_,_,_ = deep_VAE(feat_data)
        con_clust_labels = tc_consensus_clustering(embeddings, k_opt=k_opt, show=verbose)
    elif (dim_reduce_method == 'custom_embedding') and (embedding_type=='vae_umap'):
        print('Custom Embedding')
        embeddings,_,_,_,_,_ = deep_VAE_umap(feat_data)
        con_clust_labels = tc_consensus_clustering(embeddings, k_opt=k_opt, show=verbose)
    elif (dim_reduce_method == 'custom_embedding') and (embedding_type=='ftt'):
        print('Custom Embedding — TF FT-Transformer')
        # Train a TF TabTransformer-style autoencoder and extract embeddings
        embeddings, encoder, decoder, scaler, history = train_tf_tabtransformer(
            feat_data,
            d_model=128,
            num_heads=4,
            num_layers=3,
            mlp_dim=256,
            proj_out_dim=32,
            dropout=0.1,
            epochs=120,
            batch_size=32,
            lr=2e-3,
            validation_split=0.1,
            patience=20,
            standardize=True,
            random_state=42,
            verbose=verbose
        )
        con_clust_labels = tc_consensus_clustering(embeddings, k_opt=k_opt, show=verbose)
    elif (dim_reduce_method == 'custom_embedding') and (embedding_type=='ftt_umap'):
        print('Custom Embedding — TF FT-Transformer')
        # Train a TF TabTransformer-style autoencoder and extract embeddings
        embeddings, encoder, decoder, scaler, history = train_tf_tabtransformer(
            feat_data,
            d_model=128,
            num_heads=4,
            num_layers=3,
            mlp_dim=256,
            proj_out_dim=32,
            dropout=0.1,
            epochs=120,
            batch_size=32,
            lr=2e-3,
            validation_split=0.1,
            patience=20,
            standardize=True,
            random_state=42,
            verbose=verbose
        )
        umap_2d_ftt = UMAP(n_components=2, init='spectral', random_state=10, n_neighbors=30)#min_dist=.25)
        proj_2d_ftt = umap_2d_ftt.fit_transform(embeddings) 
        con_clust_labels = tc_consensus_clustering(proj_2d_ftt, k_opt=k_opt, show=verbose)
    else:
        print('No Dimensionality Reduction!')
        con_clust_labels = tc_consensus_clustering(np.array(feat_data), k_opt=k_opt, show=verbose)
    
    if verbose:
        print(f'Initial feature.shape: {feat_data_all.shape}')
        print(f'After VIF and dropping highly correlated features, feature.shape: {feat_data.shape}')
        corr = feat_data.corr()
        f, ax = plt.subplots(figsize=(22,20))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        heatmap = sns.heatmap(corr, cmap=cmap, center=0.0, vmax=1, linewidth=1, ax=ax)
        plt.savefig('Figures/pitts_corr_features_noABP.png', format="png", bbox_inches="tight", dpi=300)
        metric(proj_2d_standard, con_clust_labels)
        print(pd.Series(con_clust_labels).value_counts())
        color = dict(zip(range(0,7), plt.cm.tab10(range(0,7))))
        plt.figure(figsize=(5, 5))
        plt.scatter(proj_2d_standard[:,0], proj_2d_standard[:,1], c=pd.Series(con_clust_labels).astype('int').map(color), s=25)
        plt.title('Sepsis Phenotypes')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig("Figures/umap2D_con_clust_pitts.png", format="png", bbox_inches="tight", dpi=300)
        plt.show()
    return con_clust_labels, feature_list


''' Physiological phenotype predictor building -- XGBoost '''
def membership_predictor(data, sel_feature_list, cluster_labels, verbose=0):
    #data: dataframe of data
    #sel_feature_list: list of selected features
    X_df = data[sel_feature_list]
    X = data[sel_feature_list].values
    y = np.array(cluster_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb1 = xgb_model.fit(X_train,y_train)
    y_pred1 = xgb1.predict(X_test)
    y_probas = xgb1.predict_proba(X_test)
    precision, recall, fscore, support = score(y_test, y_pred1)
    f1 = (2*precision*recall)/(precision + recall)
    cm = confusion_matrix(y_test, y_pred1)
    acc_i = []
    for i in range(cm.shape[0]):
        acc  = 100*(cm[i,i])/np.sum(cm[i,:])
        acc_i = np.append(acc_i, acc)
    metrics = {'recall':recall,'precision':precision,'fscore':fscore, 'accuracy':acc_i}
    ## Feature Importance
    explainer = shap.TreeExplainer(xgb1)
    shap_values = explainer.shap_values(X) 
    
    # Create feature indices dictionary
    feature_indices = {feature: idx for idx, feature in enumerate(sel_feature_list)}
    
    # Debug: Print shapes
    if verbose:
        print("Shape of shap_values:", shap_values.shape if isinstance(shap_values, np.ndarray) else [sv.shape for sv in shap_values])

    # Handle SHAP values for multi-class or single-class
    if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3):
        # Multi-class: Aggregate SHAP values across classes for overall importance
        mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=(0, 2)) if isinstance(shap_values, np.ndarray) else np.abs(np.array(shap_values)).mean(axis=(0, 2))
        n_classes = len(shap_values) if isinstance(shap_values, list) else shap_values.shape[2]
    else:
        # Single-class or binary: Mean absolute SHAP values across samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        n_classes = 1 if shap_values.ndim == 2 else shap_values.shape[1]

    # Debug: Verify shapes
    if verbose:
        print("Shape of mean_abs_shap:", mean_abs_shap.shape)
        print("Length of sel_feature_list:", len(sel_feature_list))

    # Ensure matching lengths
    if len(mean_abs_shap) != len(sel_feature_list):
        raise ValueError(f"Mismatch between SHAP values length ({len(mean_abs_shap)}) and feature list length ({len(sel_feature_list)})")

    # Create overall feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': sel_feature_list,
        'importance': mean_abs_shap
    })
    top_5_features = feature_importance.sort_values(by='importance', ascending=False).head(5)

    # Create dictionary of top 5 features per class
    top_5_features_per_class = {}
    if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3):
        # Multi-class case
        for class_idx in range(n_classes):
            # Compute mean absolute SHAP values for this class
            class_shap_values = shap_values[class_idx] if isinstance(shap_values, list) else shap_values[:, :, class_idx]
            class_mean_abs_shap = np.abs(class_shap_values).mean(axis=0)
            
            # Create DataFrame for this class
            class_feature_importance = pd.DataFrame({
                'feature': sel_feature_list,
                'importance': class_mean_abs_shap
            })
            # Sort and get top 5 features
            top_5_features_per_class[class_idx] = class_feature_importance.sort_values(by='importance', ascending=False).head(5)
    else:
        # Single-class or binary case: Use overall feature importance
        top_5_features_per_class[0] = top_5_features
    
    if verbose:
        overall_accuracy = xgb1.score(X_test, y_test)
        print(f'Number of features input to the model: {len(sel_feature_list)}')
        print(f'Train-test split: {len(y_train)} train, {len(y_test)} test')
        print(f'Confusion matrix:\n{cm}')
        print(f'Overall accuracy: {overall_accuracy:.3f}')
        print(f'Precision: {precision.round(3)}')
        print(f'Recall: {recall.round(3)}')
        print(f'F1-score: {fscore.round(3)}')
        print(f'Accuracy per class: {(acc_i/100).round(3)}')
        print(f'Predicted clusters on test set: {Counter(y_test)}')
        print(f'selected feature list after filtering: {len(sel_feature_list)}')
        print(f'\nTop 5 features (overall):\n{top_5_features}')
        print(f'\nTop 5 features per class:')
        for class_idx, top_features in top_5_features_per_class.items():
            print(f'Class {class_idx}:\n{top_features}\n')
        
        # Plot ROC and Precision-Recall curves
        skplt.metrics.plot_roc(y_test, y_probas)
        plt.title('XGBoost: ROC Curves')
        plt.show()
        skplt.metrics.plot_precision_recall(y_test, y_probas)
        plt.title('XGBoost: Precision-Recall Curves')
        plt.show()
        skplt.metrics.plot_confusion_matrix(y_test, y_pred1, normalize=True)
        plt.title('XGBoost: Normalized Confusion Matrix')
        plt.show()
        
        
        # SHAP summary plots for each class
        X_train_df = pd.DataFrame(X_train, columns=sel_feature_list)
        if isinstance(shap_values, list) or shap_values.ndim == 3:
            for i in range(n_classes):
                print(f"SHAP summary plot for phenotype {i}")
                shap.summary_plot(shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i], X_df, show=False)#show=False
                plt.title(f"SHAP Summary Plot: Phenotype {i}")
                plt.show()
        else:
            print("SHAP summary plot for single/binary class")
            shap.summary_plot(shap_values, X_train_df, show=False)
            plt.title("SHAP Summary Plot")
            plt.show()
        
    return xgb1, metrics, top_5_features_per_class, shap_values, feature_indices


        
def clustering_and_prediction(df_data, 
                              feature_columns, 
                              standard_minmax='standard', var_thresh=0.01, corr_thresh=0.95, 
                              verbose=0, 
                              dim_reduce_method=None, embedding_type=None, 
                              k_opt=4):
    cluster_labels, sel_feature_list = tc_cluster_CC(df_data, feature_columns, standard_minmax, var_thresh, corr_thresh, verbose, dim_reduce_method, embedding_type, k_opt)
    model, metrics, top_5_features_per_class, shap_values, feature_indices  = membership_predictor(df_data, sel_feature_list, cluster_labels, verbose)
    return model, metrics, cluster_labels, sel_feature_list, top_5_features_per_class, shap_values, feature_indices

######################################################



import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def _sampling(args):
    """Sampling helper for VAE reparameterization trick."""
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def deep_VAE(X,
             latent_dim=32,
             hidden_dims=None,
             dropout_rate=0.1,
             activation='relu',
             epochs=200,
             batch_size=32,
             kl_weight=1.0,
             validation_split=0.1,
             patience=20,
             min_delta=1e-4,
             learning_rate=1e-3,
             standardize=True,
             random_state=42,
             use_sampling=False,
             verbose=1):
    """
    Train a configurable VAE on tabular data and return latent representations.

    Adjusted defaults for higher-dimensional tabular input (e.g. 350 features / ~2k samples):
      - latent_dim default increased to 32
      - epochs increased and patience increased for stable training
      - smaller batch_size to improve optimization on moderate dataset size
      - hidden_dims autogenerated when not provided (progressively smaller layers)

    Backwards compatible: if caller passes `hidden_dims` or other args in `vae_kwargs`, those are used.
    """
    # Reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Convert DataFrame to numpy if needed
    if hasattr(X, 'values'):
        X_in = X.values.astype('float32')
    else:
        X_in = np.asarray(X, dtype='float32')

    # Standardize if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_in).astype('float32')
    else:
        X_scaled = X_in

    input_dim = X_scaled.shape[1]

    # Auto-configure hidden dimensions for large input dimensionality when not provided
    if hidden_dims is None:
        # For very high-dimensional inputs create a moderately deep encoder/decoder
        if input_dim >= 300:
            hidden_dims = (512, 256, 128)
        elif input_dim >= 150:
            hidden_dims = (256, 128)
        else:
            hidden_dims = (128, 64)

    # Safety: ensure latent_dim < input_dim
    if latent_dim >= input_dim:
        latent_dim = max(1, min(int(input_dim/2), input_dim - 1))

    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = inputs
    for i, h_dim in enumerate(hidden_dims):
        x = Dense(h_dim, activation=activation, name=f'enc_dense_{i}')(x)
        x = BatchNormalization(name=f'enc_bn_{i}')(x)
        if dropout_rate and dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'enc_do_{i}')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder (mirror)
    decoder_input = Input(shape=(latent_dim,), name='z_sampling')
    y = decoder_input
    for i, h_dim in enumerate(reversed(hidden_dims)):
        y = Dense(h_dim, activation=activation, name=f'dec_dense_{i}')(y)
        y = BatchNormalization(name=f'dec_bn_{i}')(y)
        if dropout_rate and dropout_rate > 0:
            y = Dropout(dropout_rate, name=f'dec_do_{i}')(y)
    outputs = Dense(input_dim, activation='linear', name='decoder_output')(y)

    # Models
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, outputs, name='decoder') # Decoder output
    reconstructed = decoder(encoder(inputs)[2])
    ''' vae = Model(inputs, reconstructed, name='vae') '''

    # Loss: reconstruction (MSE) + KL
    from keras import ops as Kops
    class VAELossLayer(tf.keras.layers.Layer):
        def __init__(self, kl_weight=1.0, **kwargs):
            super().__init__(**kwargs)
            self.kl_weight = kl_weight
            self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        def call(self, inputs):
            x, reconstructed, z_mean, z_log_var = inputs

            reconstruction_loss = self.mse(x, reconstructed)
            if len(reconstruction_loss.shape) > 1:
                reconstruction_loss = Kops.sum(reconstruction_loss, axis=1)

            kl_loss = -0.5 * Kops.sum(
                1 + z_log_var - Kops.square(z_mean) - Kops.exp(z_log_var),
                axis=1
            )

            total_loss = Kops.mean(reconstruction_loss + self.kl_weight * kl_loss)
            self.add_loss(total_loss)
            return reconstructed
    # Wrap with VAE loss
    outputs = VAELossLayer(kl_weight=kl_weight)([inputs, reconstructed, z_mean, z_log_var])

    vae = Model(inputs, outputs, name='vae')
    # reconstruction_loss = tf.keras.losses.mse(inputs, reconstructed)  # per-sample mean squared error
    # reconstruction_loss = tf.reduce_mean(reconstruction_loss) * input_dim
    # kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    # kl_loss = tf.reduce_mean(kl_loss)
    # total_loss = reconstruction_loss + kl_weight * kl_loss
    # vae.add_loss(total_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, min_delta=min_delta, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, patience//3), min_lr=1e-6, verbose=0)
    ]

    # Fit
    history = vae.fit(X_scaled, X_scaled,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_split=validation_split,
                      callbacks=callbacks,
                      verbose=verbose)

    # Produce latent vectors: choose z_mean for stability unless sampling requested
    z_mean_pred, z_log_pred, z_sample_pred = encoder.predict(X_scaled, batch_size=batch_size, verbose=0)
    if use_sampling:
        latent_vectors = z_sample_pred
    else:
        latent_vectors = z_mean_pred

    if verbose:
        print(f"VAE trained: input_dim={input_dim}, hidden_dims={hidden_dims}, latent_dim={latent_vectors.shape[1]}, samples={X_scaled.shape[0]}")
    return latent_vectors, encoder, decoder, scaler, history


def deep_VAE_umap(X,
                  latent_dim=32,
                  n_components=2,
                  umap_kwargs=None,
                  vae_kwargs=None):
    """
    Train VAE and then apply UMAP to the latent vectors.

    Adjusted defaults tuned for ~350-dimensional input: default latent_dim=32 and conservative UMAP settings.

    Returns:
      umap_embeddings: n_samples x n_components
      latent_vectors: output of the VAE encoder (n_samples x latent_dim)
      encoder, decoder, scaler, history: from deep_VAE
    """
    if umap_kwargs is None:
        umap_kwargs = {'n_neighbors': 30, 'metric': 'euclidean', 'random_state': 42} #'min_dist': 0.1,

    if vae_kwargs is None:
        vae_kwargs = {}

    # Ensure the vae gets a reasonable latent_dim if not supplied in vae_kwargs
    if 'latent_dim' not in vae_kwargs:
        vae_kwargs['latent_dim'] = latent_dim

    latent_vectors, encoder, decoder, scaler, history = deep_VAE(X, **vae_kwargs)
    umap_model = UMAP(n_components=n_components, **umap_kwargs)
    umap_embeddings = umap_model.fit_transform(latent_vectors)
    return umap_embeddings, latent_vectors, encoder, decoder, scaler, history




# --- TensorFlow FT-Transformer / TabTransformer-style encoder for numeric tabular data ---
# Compatible with numeric-only datasets: projects each scalar feature to a token embedding,
# applies Transformer encoder blocks, and returns a class-token embedding for each row.
from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention, Dense as KDense

class _TFTransformerBlock(Layer):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            KDense(mlp_dim, activation='relu'),
            KDense(d_model)
        ])
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # x: (batch, seq_len, d_model)
        attn_out = self.mha(x, x, x, training=training)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out, training=training))
        return x


def build_tf_ft_transformer(input_dim,
                            d_model=128,
                            num_heads=4,
                            num_layers=3,
                            mlp_dim=256,
                            dropout=0.1,
                            proj_out_dim=64):
    """
    Build a TF FT-Transformer-like encoder for numeric tabular data.

    Returns: encoder_model, decoder_model, autoencoder_model
      - encoder_model: maps (batch, input_dim) -> (batch, proj_out_dim) embeddings
      - decoder_model: maps embedding -> reconstructed input (batch, input_dim)
      - autoencoder_model: full model for training (input -> reconstructed)
    """
    inputs = tf.keras.Input(shape=(input_dim,), name='tab_inputs')
    # project each scalar feature to a token embedding: (batch, input_dim, d_model)
    # x = tf.expand_dims(inputs, axis=-1)
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    token_proj = KDense(d_model, name='token_proj')
    x_tokens = token_proj(x)

    # learned class token and positional embeddings
    cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, d_model]), trainable=True, name='cls_token')
    pos_emb = tf.Variable(initial_value=tf.random.normal([1, input_dim + 1, d_model]), trainable=True, name='pos_emb')
    
#     # Constants for class token and positional embedding
#     cls_token_init = tf.random.normal([1, 1, d_model])
#     pos_emb_init = tf.random.normal([1, input_dim + 1, d_model])

#     # Lambda layer to tile the cls token for each batch
#     cls_token = tf.keras.layers.Lambda(
#         lambda x: tf.tile(tf.constant(cls_token_init), [tf.shape(x)[0], 1, 1]),
#         output_shape=lambda s: (s[0], 1, d_model),
#         name='cls_token_tiler'
#     )(x_tokens)

#     # Lambda layer to broadcast pos_emb (no tiling needed if static)
#     pos_emb = tf.keras.layers.Lambda(
#         lambda x: tf.constant(pos_emb_init),
#         output_shape=(input_dim + 1, d_model),
#         name='pos_emb_broadcast'
#     )(x_tokens)


    # concat class token
    # batch_size = tf.shape(x_tokens)[0]
    # cls_b = tf.tile(cls_token, [batch_size, 1, 1])
    
    cls_b = tf.keras.layers.Lambda(lambda x: tf.tile(cls_token, [tf.shape(x)[0], 1, 1]))(x_tokens)
    # tokens = tf.concat([cls_b, x_tokens], axis=1)
    tokens = tf.keras.layers.Concatenate(axis=1)([cls_b, x_tokens])
    # tokens = tokens + pos_emb
    tokens = tf.keras.layers.Add()([tokens, pos_emb])

    # Transformer encoder stack
    for i in range(num_layers):
        tokens = _TFTransformerBlock(d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, name=f'trans_block_{i}')(tokens)

    # class token output
    # cls_out = tokens[:, 0, :]
    cls_out = Lambda(lambda x: x[:, 0, :], name='cls_extraction')(tokens) # Extract class token representation: tokens[:, 0, :]
    
    # Normalize and project
    cls_out = LayerNormalization(epsilon=1e-6, name='cls_norm')(cls_out)
    embedding = KDense(proj_out_dim, activation=None, name='proj_head')(cls_out)

    encoder_model = tf.keras.Model(inputs=inputs, outputs=embedding, name='tf_ft_encoder')

    # decoder to reconstruct input from embedding (simple MLP)
    dec_in = tf.keras.Input(shape=(proj_out_dim,), name='dec_in')
    dec = KDense(mlp_dim, activation='relu')(dec_in)
    dec = tf.keras.layers.Dropout(dropout)(dec)
    dec_out = KDense(input_dim, activation='linear', name='reconstruction')(dec)
    decoder_model = tf.keras.Model(inputs=dec_in, outputs=dec_out, name='tf_ft_decoder')

    # autoencoder for training
    reconstructed = decoder_model(encoder_model(inputs))
    autoencoder = tf.keras.Model(inputs=inputs, outputs=reconstructed, name='tf_ft_autoencoder')

    return encoder_model, decoder_model, autoencoder


def train_tf_tabtransformer(X,
                            d_model=128,
                            num_heads=4,
                            num_layers=3,
                            mlp_dim=256,
                            proj_out_dim=64,
                            dropout=0.1,
                            epochs=200,
                            batch_size=32,
                            lr=1e-3,
                            validation_split=0.1,
                            patience=20,
                            standardize=True,
                            random_state=42,
                            verbose=1):
    """
    Train the TF TabTransformer-style autoencoder on numeric data X using MSE reconstruction.

    Returns: embeddings, encoder_model, decoder_model, scaler, history
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    if hasattr(X, 'values'):
        X_np = X.values.astype('float32')
    else:
        X_np = np.asarray(X, dtype='float32')

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype('float32')

    n_samples, input_dim = X_np.shape
    encoder, decoder, autoencoder = build_tf_ft_transformer(input_dim=input_dim,
                                                            d_model=d_model,
                                                            num_heads=num_heads,
                                                            num_layers=num_layers,
                                                            mlp_dim=mlp_dim,
                                                            dropout=dropout,
                                                            proj_out_dim=proj_out_dim)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, patience//3), min_lr=1e-6, verbose=0)
    ]

    history = autoencoder.fit(X_np, X_np,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=validation_split,
                              callbacks=callbacks,
                              verbose=verbose)

    # extract embeddings
    embeddings = encoder.predict(X_np, batch_size=batch_size, verbose=0)
    if verbose:
        print(f"TF TabTransformer trained: input_dim={input_dim}, d_model={d_model}, proj_out_dim={embeddings.shape[1]}, samples={n_samples}")

    return embeddings, encoder, decoder, scaler, history


def tf_tabtransformer_embeddings(encoder, X, scaler=None, batch_size=128):
    """Return embeddings from trained TensorFlow FT-Transformer encoder for X."""
    if hasattr(X, 'values'):
        X_np = X.values.astype('float32')
    else:
        X_np = np.asarray(X, dtype='float32')
    if scaler is not None:
        X_np = scaler.transform(X_np).astype('float32')
    embs = encoder.predict(X_np, batch_size=batch_size, verbose=0)
    return embs




















