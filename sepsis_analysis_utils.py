"""
Utility functions for sepsis phenotype analysis and clinical validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import os

import random
import tensorflow as tf

#For Reproducibility (also dropout, but currently ignored)
seed = 25  # Or any integer of your choice

# 1. Set Python random seed
random.seed(seed)

# 2. Set NumPy random seed
np.random.seed(seed)

# 3. Set TensorFlow random seed
tf.random.set_seed(seed)

# 4. Ensure deterministic ops (where possible)
os.environ["PYTHONHASHSEED"] = str(seed)

def load_analysis_results(results_dir: str) -> Dict:
    """
    Load previously saved analysis results
    
    Args:
        results_dir: Directory containing saved results
        
    Returns:
        Dictionary with loaded results
    """
    results_path = Path(results_dir)
    
    # Load summary
    summary_file = results_path / "analysis_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load cluster assignments
    assignments_file = results_path / "cluster_assignments.csv"
    assignments = pd.read_csv(assignments_file)
    
    # Load phenotype names
    names_file = results_path / "phenotype_names.json"
    with open(names_file, 'r') as f:
        phenotype_names = json.load(f)
    
    # Load interpretations if available
    interp_file = results_path / "phenotype_interpretations.json"
    interpretations = {}
    if interp_file.exists():
        with open(interp_file, 'r') as f:
            interpretations = json.load(f)
    
    return {
        'summary': summary,
        'assignments': assignments,
        'phenotype_names': phenotype_names,
        'interpretations': interpretations,
        'results_dir': str(results_dir)
    }


def create_clinical_validation_report(results: Dict, 
                                    clinical_outcomes: pd.DataFrame = None,
                                    output_file: str = None) -> str:
    """
    Create a clinical validation report for the identified phenotypes
    
    Args:
        results: Results from load_analysis_results or analysis pipeline
        clinical_outcomes: DataFrame with clinical outcomes (mortality, LOS, etc.)
        output_file: Optional file to save the report
        
    Returns:
        Formatted clinical validation report
    """
    
    report_lines = [
        "=" * 80,
        "SEPSIS PHYSIOPHENOTYPE CLINICAL VALIDATION REPORT",
        "=" * 80,
        "",
        "PHENOTYPE OVERVIEW:",
        "-" * 40
    ]
    
    # Get phenotype distribution
    if 'summary' in results:
        summary = results['summary']
        phenotype_dist = summary['phenotype_distribution']
        
        for cluster_id, info in phenotype_dist.items():
            report_lines.extend([
                f"• {info['phenotype_name']}:",
                f"  - Patients: {info['n_patients']} ({info['percentage']:.1f}%)",
                f"  - Cluster ID: {cluster_id}",
                ""
            ])
    
    # Clinical outcomes analysis
    if clinical_outcomes is not None:
        report_lines.extend([
            "CLINICAL OUTCOMES ANALYSIS:",
            "-" * 40,
            ""
        ])
        
        assignments = results.get('assignments', pd.DataFrame())
        if not assignments.empty:
            # Merge with clinical outcomes
            merged_data = assignments.merge(clinical_outcomes, 
                                          left_on='patient_id', 
                                          right_index=True, 
                                          how='inner')
            
            # Analyze outcomes by phenotype
            outcome_analysis = analyze_outcomes_by_phenotype(merged_data)
            
            for phenotype, outcomes in outcome_analysis.items():
                report_lines.extend([
                    f"📊 {phenotype}:",
                    f"   Mortality: {outcomes.get('mortality', 'N/A')}",
                    f"   Length of Stay: {outcomes.get('los_median', 'N/A')} days (median)",
                    f"   ICU Duration: {outcomes.get('icu_days_median', 'N/A')} days (median)",
                    ""
                ])
    
    # Clinical interpretation summary
    if 'interpretations' in results and results['interpretations']:
        report_lines.extend([
            "CLINICAL INTERPRETATION SUMMARY:",
            "-" * 40,
            ""
        ])
        
        for cluster_key, interp in results['interpretations'].items():
            phenotype_name = interp.get('phenotype_name', cluster_key)
            key_features = interp.get('key_features', [])
            
            report_lines.extend([
                f"🔬 {phenotype_name}:",
                f"   Key discriminative features: {', '.join(key_features[:3])}",
                f"   Patient count: {interp['cluster_stats']['n_patients']}",
                ""
            ])
    
    # Clinical recommendations
    report_lines.extend([
        "CLINICAL RECOMMENDATIONS:",
        "-" * 40,
        "",
        "1. VALIDATION STEPS:",
        "   • Validate phenotypes in independent cohort",
        "   • Correlate with clinical outcomes (mortality, LOS, organ failure)",
        "   • Assess biomarker associations",
        "",
        "2. THERAPEUTIC IMPLICATIONS:",
        "   • Consider phenotype-specific treatment protocols",
        "   • Evaluate differential drug responses",
        "   • Assess timing of interventions by phenotype",
        "",
        "3. MONITORING RECOMMENDATIONS:",
        "   • Implement real-time phenotype identification",
        "   • Track phenotype transitions over time",
        "   • Monitor treatment response by phenotype",
        "",
        "=" * 80,
        f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Clinical validation report saved: {output_file}")
    
    return report


def analyze_outcomes_by_phenotype(merged_data: pd.DataFrame) -> Dict:
    """
    Analyze clinical outcomes by phenotype
    
    Args:
        merged_data: DataFrame with phenotype assignments and clinical outcomes
        
    Returns:
        Dictionary with outcome analysis by phenotype
    """
    outcome_analysis = {}
    
    for phenotype in merged_data['phenotype_name'].unique():
        phenotype_data = merged_data[merged_data['phenotype_name'] == phenotype]
        
        outcomes = {}
        
        # Mortality analysis
        if 'mortality' in phenotype_data.columns:
            mortality_rate = phenotype_data['mortality'].mean() * 100
            outcomes['mortality'] = f"{mortality_rate:.1f}%"
        
        # Length of stay
        if 'length_of_stay' in phenotype_data.columns:
            los_median = phenotype_data['length_of_stay'].median()
            outcomes['los_median'] = f"{los_median:.1f}"
        
        # ICU days
        if 'icu_days' in phenotype_data.columns:
            icu_median = phenotype_data['icu_days'].median()
            outcomes['icu_days_median'] = f"{icu_median:.1f}"
        
        # Organ failure scores
        if 'sofa_score' in phenotype_data.columns:
            sofa_mean = phenotype_data['sofa_score'].mean()
            outcomes['sofa_mean'] = f"{sofa_mean:.1f}"
        
        outcome_analysis[phenotype] = outcomes
    
    return outcome_analysis


def create_feature_importance_plot(results: Dict, 
                                 top_n: int = 15,
                                 save_path: str = None) -> plt.Figure:
    """
    Create feature importance visualization across all phenotypes
    
    Args:
        results: Analysis results
        top_n: Number of top features to show
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    
    if 'interpretations' not in results or not results['interpretations']:
        print("No interpretation data available for feature importance plot")
        return None
    
    # Collect all feature importance data
    all_features = {}
    
    for cluster_key, interp in results['interpretations'].items():
        phenotype_name = interp.get('phenotype_name', cluster_key)
        key_features = interp.get('key_features', [])
        
        for i, feature in enumerate(key_features):
            if feature not in all_features:
                all_features[feature] = {}
            all_features[feature][phenotype_name] = len(key_features) - i  # Reverse ranking
    
    # Create feature importance matrix
    phenotype_names = list(set().union(*[features.keys() for features in all_features.values()]))
    feature_matrix = []
    feature_labels = []
    
    for feature, phenotype_scores in all_features.items():
        if len(phenotype_scores) > 1:  # Only features important in multiple phenotypes
            row = [phenotype_scores.get(phenotype, 0) for phenotype in phenotype_names]
            if sum(row) > 0:
                feature_matrix.append(row)
                feature_labels.append(feature)
    
    if not feature_matrix:
        print("No shared important features found")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, min(8, len(feature_labels) * 0.4 + 2)))
    
    # Convert to DataFrame for easier plotting
    importance_df = pd.DataFrame(feature_matrix, 
                               index=feature_labels, 
                               columns=phenotype_names)
    
    # Sort by total importance
    importance_df['total'] = importance_df.sum(axis=1)
    importance_df = importance_df.sort_values('total', ascending=True).drop('total', axis=1)
    importance_df = importance_df.tail(top_n)
    
    # Create heatmap
    sns.heatmap(importance_df, 
                annot=True, 
                cmap='YlOrRd', 
                fmt='.0f',
                ax=ax,
                cbar_kws={'label': 'Feature Importance Rank'})
    
    ax.set_title('Feature Importance Across Sepsis Physiophenotypes', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Phenotypes', fontsize=12)
    ax.set_ylabel('Physiological Features', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {save_path}")
    
    return fig


def create_phenotype_comparison_table(results: Dict, 
                                     output_file: str = None) -> pd.DataFrame:
    """
    Create a comparison table of all identified phenotypes
    
    Args:
        results: Analysis results dictionary
        output_file: Optional CSV file to save the table
        
    Returns:
        DataFrame with phenotype comparison
    """
    
    if 'summary' not in results:
        print("No summary data available")
        return pd.DataFrame()
    
    comparison_data = []
    
    # Get phenotype distribution data
    phenotype_dist = results['summary']['phenotype_distribution']
    interpretations = results.get('interpretations', {})
    
    for cluster_id, info in phenotype_dist.items():
        cluster_key = f"Cluster_{cluster_id}"
        
        row_data = {
            'Cluster_ID': cluster_id,
            'Phenotype_Name': info['phenotype_name'],
            'N_Patients': info['n_patients'],
            'Percentage': f"{info['percentage']:.1f}%",
        }
        
        # Add key features if available
        if cluster_key in interpretations:
            interp = interpretations[cluster_key]
            key_features = interp.get('key_features', [])
            row_data['Top_3_Features'] = ', '.join(key_features[:3])
            row_data['All_Key_Features'] = ', '.join(key_features)
            
            # Extract clinical characteristics from interpretation
            full_interp = interp.get('full_interpretation', '')
            row_data['Has_Clinical_Interpretation'] = 'Yes' if full_interp else 'No'
        else:
            row_data['Top_3_Features'] = 'N/A'
            row_data['All_Key_Features'] = 'N/A'
            row_data['Has_Clinical_Interpretation'] = 'No'
        
        comparison_data.append(row_data)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('N_Patients', ascending=False)
    
    if output_file:
        comparison_df.to_csv(output_file, index=False)
        print(f"Phenotype comparison table saved: {output_file}")
    
    return comparison_df


def visualize_phenotype_timeline(results: Dict, 
                               patient_data: pd.DataFrame = None,
                               time_column: str = 'sepsis_onset_time',
                               save_path: str = None) -> plt.Figure:
    """
    Create timeline visualization of phenotype progression
    
    Args:
        results: Analysis results
        patient_data: DataFrame with patient data including time information
        time_column: Column name for time information
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    
    if patient_data is None or 'assignments' not in results:
        print("Patient data and assignments needed for timeline visualization")
        return None
    
    # Merge assignments with patient data
    assignments = results['assignments']
    merged = assignments.merge(patient_data[[time_column]], 
                              left_on='patient_id', 
                              right_index=True, 
                              how='inner')
    
    if merged.empty:
        print("No matching patient data found")
        return None
    
    # Create the timeline plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Phenotype distribution over time
    merged[time_column] = pd.to_datetime(merged[time_column])
    merged['month'] = merged[time_column].dt.to_period('M')
    
    phenotype_timeline = merged.groupby(['month', 'phenotype_name']).size().unstack(fill_value=0)
    
    phenotype_timeline.plot(kind='area', stacked=True, ax=ax1, alpha=0.7)
    ax1.set_title('Phenotype Distribution Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Number of Patients')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Phenotype proportions over time
    phenotype_props = phenotype_timeline.div(phenotype_timeline.sum(axis=1), axis=0) * 100
    phenotype_props.plot(kind='line', ax=ax2, marker='o')
    ax2.set_title('Phenotype Proportions Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Percentage of Patients (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline visualization saved: {save_path}")
    
    return fig


def generate_clinical_summary_cards(results: Dict, 
                                  output_dir: str = None) -> List[str]:
    """
    Generate clinical summary cards for each phenotype
    
    Args:
        results: Analysis results with interpretations
        output_dir: Optional directory to save individual card files
        
    Returns:
        List of formatted summary cards
    """
    
    if 'interpretations' not in results or not results['interpretations']:
        print("No interpretations available for summary cards")
        return []
    
    summary_cards = []
    
    for cluster_key, interp in results['interpretations'].items():
        phenotype_name = interp.get('phenotype_name', cluster_key)
        stats = interp.get('cluster_stats', {})
        key_features = interp.get('key_features', [])
        full_interp = interp.get('full_interpretation', '')
        
        # Extract key points from interpretation
        clinical_chars = extract_section(full_interp, 'CLINICAL CHARACTERISTICS')
        pathophysiology = extract_section(full_interp, 'PATHOPHYSIOLOGICAL INTERPRETATION')
        therapeutic = extract_section(full_interp, 'THERAPEUTIC IMPLICATIONS')
        
        card = f"""
┌──────────────────────────────────────────────────────────────────────┐
│  {phenotype_name.center(68)}  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📊 DEMOGRAPHICS                                                     │
│  • Patients: {stats.get('n_patients', 'N/A')} ({stats.get('percentage', 'N/A'):.1f}%)                                       │
│                                                                      │
│  🔬 KEY FEATURES                                                     │
│  • {key_features[0] if len(key_features) > 0 else 'N/A'}                                                │
│  • {key_features[1] if len(key_features) > 1 else 'N/A'}                                                │
│  • {key_features[2] if len(key_features) > 2 else 'N/A'}                                                │
│                                                                      │
│  🧬 PATHOPHYSIOLOGY                                                  │
│  {wrap_text(pathophysiology, 66)}                                    │
│                                                                      │
│  🏥 CLINICAL PRESENTATION                                            │
│  {wrap_text(clinical_chars, 66)}                                     │
│                                                                      │
│  💊 THERAPEUTIC CONSIDERATIONS                                       │
│  {wrap_text(therapeutic, 66)}                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
        """
        
        summary_cards.append(card.strip())
        
        # Save individual cards if output directory specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            card_file = Path(output_dir) / f"{phenotype_name.replace(' ', '_')}_summary_card.txt"
            with open(card_file, 'w', encoding='utf-8') as f:
                f.write(card)
    
    return summary_cards


def extract_section(text: str, section_name: str) -> str:
    """Extract specific section from LLM interpretation"""
    lines = text.split('\n')
    section_lines = []
    in_section = False
    
    for line in lines:
        if section_name.upper() in line.upper():
            in_section = True
            continue
        elif in_section:
            if line.strip() and any(header in line.upper() for header in 
                                  ['NAME', 'PATHOPHYSIOLOGICAL', 'CLINICAL', 'THERAPEUTIC', 'PROGNOSIS']):
                break
            elif line.strip():
                section_lines.append(line.strip())
    
    return ' '.join(section_lines)[:200] + ('...' if len(' '.join(section_lines)) > 200 else '')


def wrap_text(text: str, width: int) -> str:
    """Simple text wrapping for cards"""
    if not text or len(text) <= width:
        return f"│  {text:<{width-4}}  │"
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + word) <= width - 4:
            current_line += word + " "
        else:
            lines.append(f"│  {current_line.strip():<{width-4}}  │")
            current_line = word + " "
    
    if current_line:
        lines.append(f"│  {current_line.strip():<{width-4}}  │")
    
    return '\n'.join(lines)


def export_for_publication(results: Dict, 
                         output_dir: str,
                         include_figures: bool = True) -> None:
    """
    Export results in publication-ready format
    
    Args:
        results: Complete analysis results
        output_dir: Directory to save publication materials
        include_figures: Whether to generate publication figures
    """
    
    pub_dir = Path(output_dir)
    pub_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Exporting publication materials to: {pub_dir}")
    
    # 1. Create methods section
    methods_text = generate_methods_section(results)
    with open(pub_dir / "methods_section.txt", 'w') as f:
        f.write(methods_text)
    
    # 2. Create results summary table
    summary_table = create_phenotype_comparison_table(results, 
                                                     str(pub_dir / "phenotype_summary_table.csv"))
    
    # 3. Create supplementary materials
    create_supplementary_materials(results, pub_dir)
    
    # 4. Generate figures if requested
    if include_figures:
        fig_dir = pub_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # Feature importance figure
        create_feature_importance_plot(results, save_path=str(fig_dir / "feature_importance.png"))
        
        print(f"   ✅ Figures saved to: {fig_dir}")
    
    # 5. Create clinical summary cards
    cards_dir = pub_dir / "clinical_cards"
    generate_clinical_summary_cards(results, str(cards_dir))
    
    print("   ✅ Methods section: methods_section.txt")
    print("   ✅ Summary table: phenotype_summary_table.csv")
    print("   ✅ Clinical cards: clinical_cards/")
    print("   ✅ Supplementary materials: supplementary/")
    print("\n🎉 Publication materials exported successfully!")


def generate_methods_section(results: Dict) -> str:
    """Generate methods section for publication"""
    
    summary = results.get('summary', {})
    analysis_info = summary.get('analysis_info', {})
    
    methods = f"""
METHODS - Sepsis Physiophenotype Identification

Data Processing and Feature Selection:
A total of {analysis_info.get('total_patients', 'N/A')} sepsis patients were analyzed using {analysis_info.get('total_features_selected', 'N/A')} physiological features derived from ECG, PPG, and respiratory signals collected in the 5-minute window prior to sepsis-3 onset. Features included statistical measures (mean, 25th/75th percentiles, skewness, entropy) of cardiac rhythm parameters (RR intervals, heart rate), heart rate variability metrics, respiratory patterns, and signal morphology characteristics.

Preprocessing included standardization, variance-based feature selection (threshold < 0.01), and correlation filtering (threshold > 0.95) to remove redundant variables. 

Consensus Clustering:
Physiophenotypes were identified using consensus clustering with agglomerative hierarchical clustering (average linkage, euclidean distance). The algorithm performed 100 resampling iterations with 80% sample fraction to ensure stability. The optimal number of clusters (k = {analysis_info.get('n_phenotypes', 'N/A')}) was determined using area under the cumulative distribution function and knee point analysis.

Phenotype Interpretation:
Large language models were employed to generate clinically meaningful interpretations of identified phenotypes based on discriminative features shown by SHAP importance, and statistical significance (p < 0.01).

Validation:
An XGBoost classifier was trained to predict phenotype membership, achieving cross-validated performance metrics for clinical applicability assessment.
    """
    
    return methods.strip()


def create_supplementary_materials(results: Dict, output_dir: Path) -> None:
    """Create supplementary materials"""
    
    supp_dir = output_dir / "supplementary"
    supp_dir.mkdir(exist_ok=True)
    
    # Supplementary Table 1: Detailed feature list
    if 'assignments' in results:
        # This would need the original feature data
        pass
    
    # Supplementary Material: Full LLM interpretations
    if 'interpretations' in results:
        interp_file = supp_dir / "full_interpretations.txt"
        with open(interp_file, 'w', encoding='utf-8') as f:
            f.write("SUPPLEMENTARY MATERIAL: Full Clinical Interpretations\n")
            f.write("=" * 60 + "\n\n")
            
            for cluster_key, interp in results['interpretations'].items():
                f.write(f"PHENOTYPE: {interp.get('phenotype_name', cluster_key)}\n")
                f.write("-" * 40 + "\n")
                f.write(interp.get('full_interpretation', 'No interpretation available'))
                f.write("\n\n" + "=" * 60 + "\n\n")


# Complete usage example:
def run_complete_sepsis_analysis_workflow(df_data: pd.DataFrame,
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
                                    clinical_outcomes: pd.DataFrame = None,
                                    output_base_dir: str = "./sepsis_study_results"):
    """
    Complete workflow from data to publication-ready results
    
    Args:
        df_data: Patient physiological data
        feature_columns: List of feature names
        api_key: OpenAI or Claude API key
        clinical_outcomes: Optional clinical outcomes data
        output_base_dir: Base directory for all outputs
    """
    
    print("🚀 Starting Complete Sepsis Analysis Workflow")
    print("=" * 60)
    
    # Step 1: Run full analysis
    from integrated_sepsis_pipeline import run_full_sepsis_analysis_with_llm
    
    results = run_full_sepsis_analysis_with_llm(df_data=df_data, 
         feature_columns=feature_columns, 
         standard_minmax=standard_minmax, 
         var_thresh=var_thresh, 
         corr_thresh=corr_thresh, 
         dim_reduce_method=dim_reduce_method, 
         embedding_type=embedding_type, 
         k_opt=k_opt, 
         verbose=verbose, 
         api_provider=api_provider, 
         api_key=api_key, 
         model_name=model_name,
         output_dir=f"{output_base_dir}/analysis_results/")
    
    # Step 2: Create clinical validation report
    validation_report = create_clinical_validation_report(
        results=results,
        clinical_outcomes=clinical_outcomes,
        output_file=f"{output_base_dir}/clinical_validation_report.txt"
    )
    
    # Step 3: Create comparison table
    comparison_table = create_phenotype_comparison_table(
        results=results,
        output_file=f"{output_base_dir}/phenotype_comparison.csv"
    )
    
    # Step 4: Generate clinical summary cards
    summary_cards = generate_clinical_summary_cards(
        results=results,
        output_dir=f"{output_base_dir}/clinical_cards/"
    )
    
    # Step 5: Export publication materials
    export_for_publication(
        results=results,
        output_dir=f"{output_base_dir}/publication_materials/"
    )
    
    print("\n🎉 Complete workflow finished!")
    print(f"📁 All results saved to: {output_base_dir}")
    
    return results, validation_report, comparison_table, summary_cards