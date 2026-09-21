import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import json
import pickle
from datetime import datetime

# Import your existing modules
from tc_tabular_consensus_clustering_pipeline_v4 import clustering_and_prediction
from sepsis_phenotype_interpreter import SepsisPhysiophenotypeInterpreter, run_sepsis_phenotype_interpretation

class IntegratedSepsisAnalysisPipeline:
    """
    Complete sepsis phenotyping pipeline: clustering + LLM interpretation
    """
    
    def __init__(self, 
                 api_provider='openai',
                 api_key: str = None,
                 model_name: str = None):
        """
        Initialize the integrated pipeline
        
        Args:
            api_provider: 'openai' or 'anthropic' for LLM interpretation
            api_key: API key for chosen provider
            model_name: Specific model name (e.g., 'gpt-4o-mini', 'claude-3-sonnet-20240229')
        """
        self.api_provider = api_provider
        self.api_key = api_key
        self.model_name = model_name
        
        # Results storage
        self.clustering_results = {}
        self.interpretation_results = {}
        self.phenotype_names = {}
    
    def make_json_serializable_consensus_clust(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, (np.generic,)):  # includes all numpy int/float types
            return obj.item()

        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='list')

        elif isinstance(obj, pd.Series):
            return obj.to_list()

        elif isinstance(obj, dict):
            return {k: self.make_json_serializable_consensus_clust(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self.make_json_serializable_consensus_clust(v) for v in obj]

        else:
            return obj
        
    def restore_consensus_clust_json_load(self, json_path, model_path=None):
        
        with open(json_path, "r") as f:
            d = json.load(f)

        restored = {}

        # metrics
        restored["metrics"] = d.get("metrics", {})

        # cluster_labels
        if "cluster_labels" in d:
            restored["cluster_labels"] = np.array(d["cluster_labels"])
        else:
            restored["cluster_labels"] = None

        # sel_feature_list
        restored["selected_features"] = d.get("selected_features", [])

        # top_5_features_per_class  (WIDE FORMAT: 2 rows, columns=clusters)
        if "top_5_features_per_class" in d:
            t5 = d["top_5_features_per_class"]

            # Recreate a DataFrame where keys = column names (cluster IDs)
            df = pd.DataFrame(t5)

            # Restore proper index: ["feature", "importance"]
            df.index = ["feature", "importance"]

            restored["top_5_features_per_class"] = df
        else:
            restored["top_5_features_per_class"] = None

        # shap_values
        if "shap_values" in d:
            sv = d["shap_values"]
            try:
                restored["shap_values"] = np.array(sv)
            except:
                restored["shap_values"] = [np.array(x) for x in sv]
        else:
            restored["shap_values"] = None

        # feature_indices
        restored["feature_indices"] = d.get("feature_indices", {})

        return restored
    
    
    def run_complete_analysis(self,
                            df_data: pd.DataFrame,
                            feature_columns: List[str],
                            standard_minmax: str = 'standard',
                            var_thresh: float = 0.01,
                            corr_thresh: float = 0.95,
                            dim_reduce_method: str = None,
                            embedding_type: str = None,
                            k_opt: int = 4,
                            verbose: int = 0,
                            output_dir: str = None,
                            save_results: bool = True) -> Dict:
        """
        Run complete analysis pipeline: clustering + interpretation
        
        Args:
            df_data: Patient data DataFrame
            feature_columns: List of physiological feature names
            standard_minmax: Feature scaling method ('standard', 'minmax', None)
            var_thresh: Variance threshold for feature selection
            corr_thresh: Correlation threshold for feature selection
            dim_reduce_method: Dimensionality reduction ('pca', 'umap', 'custom_embedding', None)
            embedding_type: Type of custom embedding ('vae', 'vae_umap', 'ft_transformer')
            k_opt: Optimal number of clusters
            verbose: Verbosity level for clustering
            output_dir: Directory to save results
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all analysis results
        """
        print("🚀 Starting Integrated Sepsis Phenotyping Analysis")
        print("="*60)
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./sepsis_analysis_{timestamp}/"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Consensus Clustering
        print("\n📊 Step 1: Performing Consensus Clustering...")
        print(f'run_complete_analysis - var_thresh: {var_thresh}, corr_thresh: {corr_thresh}, dim_reduce_method: {dim_reduce_method}, embedding_type: {embedding_type}')
        
        model, metrics, cluster_labels, selected_features, top_5_features_per_class, shap_values, feature_indices = clustering_and_prediction(
            df_data=df_data,
            feature_columns=feature_columns,
            standard_minmax=standard_minmax,
            var_thresh=var_thresh,
            corr_thresh=corr_thresh,
            verbose=verbose,
            dim_reduce_method=dim_reduce_method,
            embedding_type=embedding_type,
            k_opt=k_opt
        )

        self.consensus_clust = {'model': model, 
                                'metrics': metrics, 
                                'cluster_labels':cluster_labels, 
                                'selected_features':selected_features,
                                'top_5_features_per_class':top_5_features_per_class,
                                'shap_values': shap_values,
                                'feature_indices': feature_indices
                                }
        consensus_clust_no_model = {
            k: v for k, v in self.consensus_clust.items() if k != 'model'
        }
        consensus_clust_json = self.make_json_serializable_consensus_clust(consensus_clust_no_model)
        
        # Save consensus_clust_json
        consensus_clust_path = os.path.join(output_dir, "consensus_clust.json")
        with open(consensus_clust_path, "w") as json_file:
            json.dump(consensus_clust_json, json_file, indent=4)
        
        if verbose:
            self._debug_shap_structure(shap_values, feature_indices)
        
        
        # Convert cluster_labels to native Python int
        cluster_labels = cluster_labels.astype(int) if isinstance(cluster_labels, np.ndarray) else [int(x) for x in cluster_labels]
        
        top_5_features_per_class = {
            k: v.reset_index(drop=True)
            for k, v in top_5_features_per_class.items()
        }
        
        self.clustering_results = {
            'model': model,
            'metrics': metrics,
            'cluster_labels': cluster_labels,
            'selected_features': selected_features,
            'n_clusters': len(set(cluster_labels)),
            'n_patients': len(cluster_labels),
            'n_features': len(selected_features),
            'top_5_features_per_class': top_5_features_per_class,
            'shap_values': shap_values,
            'feature_indices': feature_indices
        }
        
        print(f"✅ Clustering complete: {self.clustering_results['n_clusters']} phenotypes identified")
        print(f"   - {self.clustering_results['n_patients']} patients analyzed")
        print(f"   - {self.clustering_results['n_features']} features selected")
        print(f"   - Top SHAP features with mean absolute values per class: {self.clustering_results['top_5_features_per_class']}")
        
        # Step 2: LLM-based Interpretation
        if self.api_key:
            print(f"\n🤖 Step 2: Generating LLM Interpretations ({self.api_provider.upper()})...")
            
            try:
                interpreter, report = run_sepsis_phenotype_interpretation(
                    data=df_data,
                    cluster_labels=cluster_labels,
                    feature_columns=selected_features,
                    top_5_features_per_class=self.clustering_results['top_5_features_per_class'],
                    shap_values=self.clustering_results['shap_values'],
                    feature_indices=self.clustering_results['feature_indices'],
                    api_provider=self.api_provider,
                    api_key=self.api_key,
                    model_name=self.model_name,
                    output_dir=output_dir
                )
                
                self.interpretation_results = interpreter.phenotype_interpretations
                self.phenotype_names = {int(k): v for k, v in interpreter.get_phenotype_names().items()}  # Ensure int keys
                
                print("✅ LLM interpretation complete")
                print("   - Phenotype names generated")
                print("   - Clinical interpretations created")
                
            except Exception as e:
                print(f"⚠️  LLM interpretation failed: {e}")
                print("   Continuing with cluster IDs only...")
                self.phenotype_names = {int(i): f"Phenotype_{i}" for i in set(cluster_labels)}
        else:
            print("\n⚠️  Step 2: Skipping LLM interpretation (no API key provided)")
            self.phenotype_names = {int(i): f"Phenotype_{i}" for i in set(cluster_labels)}
        
        # Step 3: Create Summary
        print(f"\n📋 Step 3: Creating Analysis Summary...")
        summary = self._create_analysis_summary()
        
        # Step 4: Save Results
        if save_results:
            print(f"\n💾 Step 4: Saving Results to {output_dir}")
            self._save_all_results(output_dir, summary, df_data)
        
        print("\n🎉 Analysis Complete!")
        print("="*60)
        
        # Final Results
        results = {
            'clustering': self.clustering_results,
            'interpretations': self.interpretation_results,
            'phenotype_names': self.phenotype_names,
            'summary': summary,
            'output_dir': output_dir if save_results else None
        }
        
        return results
    
    def _debug_shap_structure(self, shap_values, feature_indices):
        """Debug method to understand SHAP values structure"""
        print("🔍 SHAP Values Debug Info:")
        print(f"Type: {type(shap_values)}")

        if isinstance(shap_values, np.ndarray):
            print(f"Shape: {shap_values.shape}")
            print(f"Number of dimensions: {shap_values.ndim}")
        elif isinstance(shap_values, list):
            print(f"List length: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                print(f"  shap_values[{i}] shape: {sv.shape}")

        print(f"Feature indices: {len(feature_indices)} features")
        print(f"Sample feature indices: {list(feature_indices.items())[:5]}")


    
    
    def _create_analysis_summary(self) -> Dict:
        """Create comprehensive analysis summary"""
        cluster_labels = self.clustering_results['cluster_labels']
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        cluster_distribution = {}
        for cluster_id, count in zip(unique_clusters, counts):
            cluster_id_native = int(cluster_id)  # Convert to Python int
            count_native = int(count)
            phenotype_name = self.phenotype_names.get(cluster_id_native, f"Cluster_{cluster_id_native}")
            percentage = float(count_native / len(cluster_labels) * 100)
            
            cluster_distribution[str(cluster_id_native)] = {
                'cluster_id': cluster_id_native,
                'phenotype_name': phenotype_name,
                'n_patients': count_native,
                'percentage': percentage
            }
        
        selected_features = self.clustering_results['selected_features']
        feature_categories = self._categorize_features(selected_features)
        
        def make_json_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        summary = {
            'analysis_info': {
                'total_patients': int(self.clustering_results['n_patients']),
                'total_features_selected': int(self.clustering_results['n_features']),
                'n_phenotypes': int(self.clustering_results['n_clusters']),
                'has_llm_interpretation': bool(self.interpretation_results),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'phenotype_distribution': cluster_distribution,
            'feature_categories': feature_categories,
            'model_performance': {
                'precision': make_json_serializable(self.clustering_results['metrics']['precision']),
                'recall': make_json_serializable(self.clustering_results['metrics']['recall']),
                'f1_score': make_json_serializable(self.clustering_results['metrics']['fscore']),
                # 'accuracy_per_class': make_json_serializable(self.clustering_results['metrics']['accuracy_per_class']),
                'accuracy': make_json_serializable(self.clustering_results['metrics']['accuracy'])
            }
        }
        
        return summary
    
    def _categorize_features(self, features: List[str]) -> Dict:
        """Categorize physiological features"""
        categories = {
            'cardiac_rhythm': ['T_rr_ms', 'HR_bpm', 'rr_interval'],
            'heart_rate_variability': ['hrv', 'sdnn', 'rmssd', 'pnn50'],
            'respiratory': ['edr_rate', 'resp', 'breathing'],
            'cardiac_phases': ['atrialSys', 'ventricularSys', 'diastolic'],
            'frequency_domain': ['vlf', 'lf', 'hf', 'power'],
            'ppg_derived': ['ppg', 'pulse', 'spo2'],
            'statistical_measures': ['mean', 'p25', 'p75', 'skewness', 'entropy'],
            'morphology': ['morphology', 'shape', 'amplitude'],
            'other': []
        }
        
        feature_counts = {cat: 0 for cat in categories.keys()}
        
        for feature in features:
            categorized = False
            for category, keywords in categories.items():
                if category != 'other' and any(keyword.lower() in feature.lower() for keyword in keywords):
                    feature_counts[category] += 1
                    categorized = True
                    break
            if not categorized:
                feature_counts['other'] += 1
        
        return feature_counts
    
    def _save_all_results(self, output_dir: str, summary: Dict, df_data: pd.DataFrame):
        """Save all analysis results"""
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Save summary
        summary_path = os.path.join(output_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        # Save cluster labels
        labels_df = pd.DataFrame({
            'patient_id': df_data.index,
            'cluster_label': self.clustering_results['cluster_labels'],
            'phenotype_name': [self.phenotype_names.get(int(label), f"Phenotype_{label}") 
                             for label in self.clustering_results['cluster_labels']]
        })
        labels_df.to_csv(os.path.join(output_dir, "cluster_assignments.csv"), index=False)
        
        # Save selected features
        features_df = pd.DataFrame({
            'feature_name': self.clustering_results['selected_features'],
            'feature_category': [self._get_feature_category(f) for f in self.clustering_results['selected_features']]
        })
        features_df.to_csv(os.path.join(output_dir, "selected_features.csv"), index=False)
        
        # Save trained model
        model_path = os.path.join(output_dir, "trained_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.clustering_results['model'], f)
        
        # Save phenotype names
        names_path = os.path.join(output_dir, "phenotype_names.json")
        with open(names_path, 'w') as f:
            json.dump({int(k): v for k, v in self.phenotype_names.items()}, f, indent=2, cls=NumpyEncoder)
        
        print(f"   ✅ Analysis summary: analysis_summary.json")
        print(f"   ✅ Cluster assignments: cluster_assignments.csv")
        print(f"   ✅ Selected features: selected_features.csv")
        print(f"   ✅ Trained model: trained_model.pkl")
        print(f"   ✅ Phenotype names: phenotype_names.json")
    
    def _get_feature_category(self, feature: str) -> str:
        """Get category for a single feature"""
        categories = {
            'cardiac_rhythm': ['T_rr_ms', 'HR_bpm', 'rr_interval'],
            'heart_rate_variability': ['hrv', 'sdnn', 'rmssd', 'pnn50'],
            'respiratory': ['edr_rate', 'resp', 'breathing'],
            'cardiac_phases': ['atrialSys', 'ventricularSys', 'diastolic'],
            'frequency_domain': ['vlf', 'lf', 'hf', 'power'],
            'ppg_derived': ['ppg', 'pulse', 'spo2'],
            'statistical_measures': ['mean', 'p25', 'p75', 'skewness', 'entropy'],
            'morphology': ['morphology', 'shape', 'amplitude']
        }
        
        for category, keywords in categories.items():
            if any(keyword.lower() in feature.lower() for keyword in keywords):
                return category
        return 'other'
    
    def print_phenotype_summary(self):
        """Print a nice summary of identified phenotypes"""
        if not self.phenotype_names:
            print("No phenotypes identified yet. Run analysis first.")
            return
        
        print("\n" + "="*60)
        print("IDENTIFIED SEPSIS PHYSIOPHENOTYPES")
        print("="*60)
        
        cluster_labels = self.clustering_results['cluster_labels']
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        total_patients = len(cluster_labels)
        
        for cluster_id, count in zip(unique_clusters, counts):
            cluster_id = int(cluster_id)  # Ensure native Python int
            phenotype_name = self.phenotype_names.get(cluster_id, f"Phenotype_{cluster_id}")
            percentage = (count / total_patients) * 100
            
            print(f"\n🔬 PHENOTYPE {cluster_id}: {phenotype_name}")
            print(f"   📊 Patients: {count:,} ({percentage:.1f}%)")
            
            if self.interpretation_results:
                cluster_key = f"Cluster_{cluster_id}"
                if cluster_key in self.interpretation_results:
                    key_features = self.interpretation_results[cluster_key].get('key_features', [])
                    print(f"   🔑 Key Features: {', '.join(key_features[:3])}...")
        
        print("\n" + "="*60)
        print(f"Total: {total_patients:,} patients across {len(unique_clusters)} phenotypes")
        print("="*60)

# Example usage functions
def run_basic_sepsis_analysis(df_data: pd.DataFrame,
                            feature_columns: List[str],
                            standard_minmax: str = 'standard',
                            var_thresh: float = 0.01,
                            corr_thresh: float = 0.95,
                            dim_reduce_method: str = None,
                            embedding_type: str = None,
                            k_opt: int = 4,
                            verbose: int = 0,
                            output_dir: str = None) -> Dict:
    """Run basic analysis without LLM interpretation"""
    pipeline = IntegratedSepsisAnalysisPipeline()
    results = pipeline.run_complete_analysis(
        df_data=df_data,
        feature_columns=feature_columns,
        standard_minmax=standard_minmax,
        var_thresh=var_thresh,
        corr_thresh=corr_thresh,
        dim_reduce_method=dim_reduce_method,
        embedding_type=embedding_type,
        k_opt=k_opt,
        verbose=verbose,
        output_dir=output_dir,
    )
    pipeline.print_phenotype_summary()
    return results

def run_full_sepsis_analysis_with_llm(df_data: pd.DataFrame,
                                     feature_columns: List[str],
                                     standard_minmax: str = 'standard',
                                     var_thresh: float = 0.01,
                                     corr_thresh: float = 0.95,
                                     dim_reduce_method: str = None,
                                     embedding_type: str = None,
                                     k_opt: int = 4,
                                     verbose: int = 1,
                                     api_provider: str = 'openai',
                                     api_key: str = None,
                                     model_name: str = None,
                                     output_dir: str = None) -> Dict:
    """Run complete analysis with LLM interpretation"""
    if model_name is None:
        print("Model name not supplied. Attempting default selection based on api_provider...")
        if api_provider == 'openai':
            model_name = 'gpt-4o-mini'
            print("Defaulting to 'gpt-4o-mini' for OpenAI.")
        else:
            raise ValueError(f"No model_name provided and no default available for api_provider: '{api_provider}'")

    pipeline = IntegratedSepsisAnalysisPipeline(
        api_provider=api_provider,
        api_key=api_key,
        model_name=model_name
    )

    results = pipeline.run_complete_analysis(
        df_data=df_data,
        feature_columns=feature_columns,
        standard_minmax=standard_minmax,
        var_thresh=var_thresh,
        corr_thresh=corr_thresh,
        dim_reduce_method=dim_reduce_method,
        embedding_type=embedding_type,
        k_opt=k_opt,
        verbose=verbose,
        output_dir=output_dir,
    )
    
    pipeline.print_phenotype_summary()
    return results