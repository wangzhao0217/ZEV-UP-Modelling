"""
Adoption Propensity Group Analysis Module

Analyzes socio-demographic characteristics across three adoption propensity groups
defined by feasibility thresholds (τ_AP_min and τ_AP_max).

Groups:
1. Below Minimum (AP < τ_AP_min): Too low for feasibility
2. Feasible Range (τ_AP_min ≤ AP ≤ τ_AP_max): Suitable for deployment
3. Above Maximum (AP > τ_AP_max): Potentially unrealistic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PropensityGroupAnalyzer:
    """
    Analyzes socio-demographic characteristics across adoption propensity groups.
    """
    
    def __init__(self, tau_AP_min: float = 0.4, tau_AP_max: float = 0.8):
        """
        Initialize analyzer with feasibility thresholds.
        
        Parameters
        ----------
        tau_AP_min : float
            Minimum adoption propensity threshold (default: 0.4)
        tau_AP_max : float
            Maximum adoption propensity threshold (default: 0.8)
        """
        self.tau_AP_min = tau_AP_min
        self.tau_AP_max = tau_AP_max
        
        # Define socio-demographic feature columns
        self.feature_columns = [
            'social_grade_score',
            'education_score',
            'car_ownership_score',
            'housing_score',
            'age_score',
            'economic_activity_score',
            'household_composition_score',
            'population_density_score'
        ]
        
        # Feature display names for plots
        self.feature_labels = {
            'social_grade_score': 'Social Grade',
            'education_score': 'Education',
            'car_ownership_score': 'Car Ownership',
            'housing_score': 'Housing',
            'age_score': 'Age',
            'economic_activity_score': 'Economic Activity',
            'household_composition_score': 'Household Composition',
            'population_density_score': 'Population Density'
        }
    
    def categorize_areas(self, df: pd.DataFrame, 
                        ap_column: str = 'final_adoption_propensity') -> pd.DataFrame:
        """
        Categorize areas into three adoption propensity groups.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with adoption propensity scores
        ap_column : str
            Name of adoption propensity column
            
        Returns
        -------
        pd.DataFrame
            Dataframe with added 'ap_group' column
        """
        df = df.copy()
        
        # Find adoption propensity column if not specified
        if ap_column not in df.columns:
            for col in ['final_adoption_propensity', 'adoption_propensity', 'AP']:
                if col in df.columns:
                    ap_column = col
                    break
        
        if ap_column not in df.columns:
            raise ValueError(f"Adoption propensity column '{ap_column}' not found")
        
        # Categorize into three groups
        conditions = [
            df[ap_column] < self.tau_AP_min,
            (df[ap_column] >= self.tau_AP_min) & (df[ap_column] <= self.tau_AP_max),
            df[ap_column] > self.tau_AP_max
        ]
        
        choices = ['Below_Min', 'Feasible_Range', 'Above_Max']
        
        df['ap_group'] = np.select(conditions, choices, default='Unknown')
        
        return df
    
    def perform_pca_analysis(self, df: pd.DataFrame) -> Tuple[np.ndarray, PCA, StandardScaler]:
        """
        Perform PCA on socio-demographic features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Categorized dataframe with socio-demographic scores
            
        Returns
        -------
        pca_features : np.ndarray
            Transformed features in PCA space
        pca : PCA
            Fitted PCA object
        scaler : StandardScaler
            Fitted scaler object
        """
        # Check available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No socio-demographic score columns found")
        
        # Extract and standardize features
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=min(len(available_features), 3))
        pca_features = pca.fit_transform(X_scaled)
        
        return pca_features, pca, scaler
    
    def compute_group_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics for each group.
        
        Parameters
        ----------
        df : pd.DataFrame
            Categorized dataframe with 'ap_group' column
            
        Returns
        -------
        pd.DataFrame
            Summary statistics by group
        """
        # Check available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No socio-demographic score columns found")
        
        # Compute statistics for each group
        stats_list = []
        
        for group in ['Below_Min', 'Feasible_Range', 'Above_Max']:
            group_data = df[df['ap_group'] == group]
            
            if len(group_data) == 0:
                continue
            
            stats = {
                'group': group,
                'n_areas': len(group_data),
                'percentage': len(group_data) / len(df) * 100
            }
            
            # Calculate mean and std for each feature
            for feature in available_features:
                stats[f'{feature}_mean'] = group_data[feature].mean()
                stats[f'{feature}_std'] = group_data[feature].std()
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def visualize_group_profiles(self, df: pd.DataFrame, 
                                 output_path: Optional[str] = None,
                                 region_name: str = "Region") -> plt.Figure:
        """
        Create comprehensive PCA-based visualization of group profiles.
        
        Parameters
        ----------
        df : pd.DataFrame
            Categorized dataframe with 'ap_group' column
        output_path : str, optional
            Path to save figure
        region_name : str
            Name of region for plot title
            
        Returns
        -------
        plt.Figure
            Generated matplotlib figure
        """
        # Check available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No socio-demographic score columns found")
        
        # Perform PCA
        pca_features, pca, scaler = self.perform_pca_analysis(df)
        
        # Add PCA coordinates to dataframe
        df_with_pca = df.copy()
        df_with_pca['PC1'] = pca_features[:, 0]
        df_with_pca['PC2'] = pca_features[:, 1]
        if pca_features.shape[1] > 2:
            df_with_pca['PC3'] = pca_features[:, 2]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # Define group colors
        group_colors = {
            'Below_Min': '#e74c3c',
            'Feasible_Range': '#2ecc71',
            'Above_Max': '#f39c12'
        }
        
        group_labels = {
            'Below_Min': f'Below τ_AP_min (<{self.tau_AP_min})',
            'Feasible_Range': f'Feasible Range ({self.tau_AP_min}-{self.tau_AP_max})',
            'Above_Max': f'Above τ_AP_max (>{self.tau_AP_max})'
        }
        
        # 1. PCA Scatter Plot (PC1 vs PC2) - Main visualization
        ax1 = fig.add_subplot(gs[0, :2])
        
        for group in ['Below_Min', 'Feasible_Range', 'Above_Max']:
            group_data = df_with_pca[df_with_pca['ap_group'] == group]
            if len(group_data) > 0:
                ax1.scatter(group_data['PC1'], group_data['PC2'],
                          c=group_colors[group], label=group_labels[group],
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        var_explained_1 = pca.explained_variance_ratio_[0] * 100
        var_explained_2 = pca.explained_variance_ratio_[1] * 100
        
        ax1.set_xlabel(f'PC1 ({var_explained_1:.1f}% variance)', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'PC2 ({var_explained_2:.1f}% variance)', fontsize=14, fontweight='bold')
        ax1.set_title(f'PCA Cluster Visualization: Adoption Propensity Groups - {region_name}',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 2. Explained Variance
        ax2 = fig.add_subplot(gs[0, 2])
        
        var_explained = pca.explained_variance_ratio_ * 100
        cumsum_var = np.cumsum(var_explained)
        n_components = len(var_explained)
        
        bars = ax2.bar(range(1, n_components + 1), var_explained,
                      color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax2.plot(range(1, n_components + 1), cumsum_var, 'ro-',
                linewidth=2, markersize=6, label='Cumulative')
        
        ax2.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
        ax2.set_title('PCA Variance Explained', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, var_explained)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 3. PCA Loadings/Contributions (PC1 and PC2)
        ax3 = fig.add_subplot(gs[1, :])
        
        loadings = pca.components_[:2, :].T
        x = np.arange(len(available_features))
        width = 0.35
        
        ax3.bar(x - width/2, loadings[:, 0], width, label='PC1', 
               color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax3.bar(x + width/2, loadings[:, 1], width, label='PC2',
               color='#e67e22', edgecolor='black', linewidth=0.5, alpha=0.7)
        
        ax3.set_xlabel('Socio-Demographic Dimension', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Feature Loading', fontsize=12, fontweight='bold')
        ax3.set_title('PCA Feature Loadings (Contributions to Principal Components)',
                     fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([self.feature_labels.get(f, f) for f in available_features],
                           rotation=45, ha='right')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
        
        # 4. Group Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        
        group_counts = df_with_pca['ap_group'].value_counts()
        colors_list = [group_colors.get(g, 'gray') for g in group_counts.index]
        labels_list = [group_labels.get(g, g) for g in group_counts.index]
        
        ax4.bar(range(len(group_counts)), group_counts.values,
               color=colors_list, edgecolor='black', linewidth=0.5)
        ax4.set_xticks(range(len(group_counts)))
        ax4.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Number of Areas', fontsize=11, fontweight='bold')
        ax4.set_title('Group Distribution', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (count, pct) in enumerate(zip(group_counts.values,
                                             group_counts.values / len(df) * 100)):
            ax4.text(i, count, f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 5. Feature Contribution Magnitudes
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Calculate loading magnitudes for PC1 and PC2
        loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
        sorted_idx = np.argsort(loading_magnitudes)[::-1]
        
        # Plot top features
        y_pos = np.arange(len(available_features))
        ax5.barh(y_pos, loading_magnitudes[sorted_idx], 
                color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([self.feature_labels.get(available_features[i], available_features[i]) 
                             for i in sorted_idx], fontsize=9)
        ax5.set_xlabel('Loading Magnitude', fontsize=10, fontweight='bold')
        ax5.set_title('Feature Importance in PC1-PC2 Space', fontsize=11, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        ax5.invert_yaxis()
        
        # 6. Group Centroids in PCA Space
        ax6 = fig.add_subplot(gs[2, 2])
        
        for group in ['Below_Min', 'Feasible_Range', 'Above_Max']:
            group_data = df_with_pca[df_with_pca['ap_group'] == group]
            if len(group_data) > 0:
                centroid_pc1 = group_data['PC1'].mean()
                centroid_pc2 = group_data['PC2'].mean()
                ax6.scatter(centroid_pc1, centroid_pc2,
                          c=group_colors[group], s=300, marker='D',
                          edgecolors='black', linewidth=2,
                          label=group_labels[group], alpha=0.8)
                ax6.text(centroid_pc1, centroid_pc2, group[:3],
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
        
        ax6.set_xlabel(f'PC1 ({var_explained_1:.1f}%)', fontsize=10, fontweight='bold')
        ax6.set_ylabel(f'PC2 ({var_explained_2:.1f}%)', fontsize=10, fontweight='bold')
        ax6.set_title('Group Centroids', fontsize=11, fontweight='bold')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(alpha=0.3, linestyle='--')
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax6.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.suptitle(f'PCA-Based Adoption Propensity Group Analysis: {region_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {output_path}")
        
        return fig
    
    def create_biplot(self, df: pd.DataFrame, 
                     output_path: Optional[str] = None,
                     region_name: str = "Region") -> plt.Figure:
        """
        Create a separate, detailed biplot figure with confidence ellipses.
        
        Parameters
        ----------
        df : pd.DataFrame
            Categorized dataframe with 'ap_group' column
        output_path : str, optional
            Path to save figure
        region_name : str
            Name of region for plot title
            
        Returns
        -------
        plt.Figure
            Generated biplot figure
        """
        from matplotlib.patches import Ellipse
        from scipy import stats
        
        # Perform PCA
        pca_features, pca, scaler = self.perform_pca_analysis(df)
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        df_with_pca = df.copy()
        df_with_pca['PC1'] = pca_features[:, 0]
        df_with_pca['PC2'] = pca_features[:, 1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Define colors matching the example (wine colors)
        group_colors = {
            'Below_Min': '#d9534f',      # Red/coral like barolo
            'Feasible_Range': '#5cb85c',  # Green like grignolino
            'Above_Max': '#5bc0de'        # Blue like barbera
        }
        
        group_labels = {
            'Below_Min': f'Below AP_min (<{self.tau_AP_min})',
            'Feasible_Range': f'Feasible Range ({self.tau_AP_min}-{self.tau_AP_max})',
            'Above_Max': f'Above AP_max (>{self.tau_AP_max})'
        }
        
        # Function to draw confidence ellipse
        def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
            """Draw confidence ellipse for data points"""
            if len(x) < 3:
                return None
            
            cov = np.cov(x, y)
            mean_x, mean_y = np.mean(x), np.mean(y)
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            
            # Calculate angle and dimensions
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)
            
            ellipse = Ellipse((mean_x, mean_y), width, height, 
                            angle=angle, facecolor=facecolor, **kwargs)
            return ax.add_patch(ellipse)
        
        # Plot points and confidence ellipses
        for group in ['Below_Min', 'Feasible_Range', 'Above_Max']:
            group_data = df_with_pca[df_with_pca['ap_group'] == group]
            if len(group_data) > 0:
                # Plot scatter points
                ax.scatter(group_data['PC1'], group_data['PC2'],
                          c=group_colors[group], label=group_labels[group],
                          alpha=0.5, s=80, edgecolors='white', linewidth=0.5, zorder=5)
                
                # Add confidence ellipse
                confidence_ellipse(group_data['PC1'].values, group_data['PC2'].values,
                                 ax, n_std=2.0, 
                                 edgecolor=group_colors[group], 
                                 facecolor='none',
                                 linewidth=2.5, alpha=0.8, zorder=4)
        
        # Calculate loadings and scaling
        loadings = pca.components_[:2, :].T
        pc1_range = df_with_pca['PC1'].max() - df_with_pca['PC1'].min()
        pc2_range = df_with_pca['PC2'].max() - df_with_pca['PC2'].min()
        data_scale = max(pc1_range, pc2_range)
        scale_factor = data_scale * 0.35
        
        # Calculate label positions: exactly at arrow tips along arrow direction
        label_info = []
        
        for i, feature in enumerate(available_features):
            arrow_x = loadings[i, 0] * scale_factor
            arrow_y = loadings[i, 1] * scale_factor
            
            angle = np.arctan2(arrow_y, arrow_x)
            angle_deg = np.degrees(angle)
            
            # Position further from arrow tip along arrow direction
            # Text will extend from this point along the arrow angle
            label_x = arrow_x * 1.08
            label_y = arrow_y * 1.08
            
            # Calculate text rotation to match arrow angle
            text_angle = angle_deg
            
            # Determine alignment based on arrow direction
            # For arrows pointing right (angle between -90 and 90): left-align
            # For arrows pointing left (angle < -90 or > 90): right-align
            if -90 <= angle_deg <= 90:
                # Arrow points right: text extends to the right from arrow tip
                ha = 'left'
            else:
                # Arrow points left: text extends to the left from arrow tip
                ha = 'right'
            
            # Normalize rotation to avoid upside-down text (keep readable)
            if text_angle > 90:
                text_angle = text_angle - 180
            elif text_angle < -90:
                text_angle = text_angle + 180
            
            va = 'center'
            
            label_info.append({
                'feature': feature,
                'arrow_x': arrow_x,
                'arrow_y': arrow_y,
                'label_x': label_x,
                'label_y': label_y,
                'ha': ha,
                'va': va,
                'angle': angle,
                'text_angle': text_angle
            })
        
        # Second pass: draw arrows and labels
        for info in label_info:
            # Draw arrow
            ax.arrow(0, 0, info['arrow_x'], info['arrow_y'],
                    head_width=data_scale*0.025, 
                    head_length=data_scale*0.025,
                    fc='#8B4513',
                    ec='#8B4513',
                    alpha=0.9,
                    linewidth=2.0,
                    zorder=10,
                    length_includes_head=True)
            
            # Draw label at arrow tip, rotated to match arrow angle
            label_text = self.feature_labels.get(info['feature'], info['feature'])
            ax.text(info['label_x'], info['label_y'], label_text,
                   fontsize=11, 
                   ha=info['ha'], va=info['va'],
                   fontweight='bold',
                   color='#8B4513',
                   rotation=info['text_angle'],  # Rotate to align with arrow
                   rotation_mode='anchor',
                   zorder=11,
                   bbox=dict(boxstyle='round,pad=0.25',
                            facecolor='white',
                            edgecolor='#8B4513',
                            alpha=0.9,
                            linewidth=1.0))
        
        var_explained_1 = pca.explained_variance_ratio_[0] * 100
        var_explained_2 = pca.explained_variance_ratio_[1] * 100
        
        # Styling to match the wine example
        ax.set_xlabel(f'PC1 ({var_explained_1:.1f}% explained var.)', 
                     fontsize=18, fontweight='normal')
        ax.set_ylabel(f'PC2 ({var_explained_2:.1f}% explained var.)', 
                     fontsize=18, fontweight='normal')
        
        # Title at the top
        ax.set_title(f'Adoption Propensity (AP) Group Analysis: PCA Biplot\n{region_name}',
                    fontsize=15, fontweight='bold', pad=10)
        
        # Legend positioned at BOTTOM of figure, outside plot area
        # Labels already use AP_min/AP_max from group_labels dictionary  
        legend = ax.legend(title='Adoption Propensity Groups', 
                          loc='upper center', bbox_to_anchor=(0.5, -0.12),
                          ncol=3, fontsize=14, frameon=False, 
                          markerscale=1.3, title_fontsize=11)
        
        # Reserve space at bottom for legend
        plt.subplots_adjust(bottom=0.15, top=0.95)
        
        # Clean grid like the example
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        
        # White background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Equal aspect ratio for proper interpretation
        ax.set_aspect('equal', adjustable='datalim')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Biplot saved to: {output_path}")
        
        return fig
    
    def generate_summary_report(self, df: pd.DataFrame, 
                               region_name: str = "Region") -> str:
        """
        Generate text summary report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Categorized dataframe with 'ap_group' column
        region_name : str
            Name of region
            
        Returns
        -------
        str
            Formatted summary report
        """
        report = []
        report.append("="*80)
        report.append(f"ADOPTION PROPENSITY GROUP ANALYSIS: {region_name}")
        report.append("="*80)
        report.append(f"\nThresholds: τ_AP_min = {self.tau_AP_min}, τ_AP_max = {self.tau_AP_max}")
        report.append(f"Total Areas: {len(df)}\n")
        
        # Check available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        # Overall distribution
        report.append("GROUP DISTRIBUTION:")
        report.append("-"*80)
        
        for group in ['Below_Min', 'Feasible_Range', 'Above_Max']:
            group_data = df[df['ap_group'] == group]
            count = len(group_data)
            pct = count / len(df) * 100
            
            group_label = {
                'Below_Min': f'Below Minimum (AP < {self.tau_AP_min})',
                'Feasible_Range': f'Feasible Range ({self.tau_AP_min} ≤ AP ≤ {self.tau_AP_max})',
                'Above_Max': f'Above Maximum (AP > {self.tau_AP_max})'
            }[group]
            
            report.append(f"  • {group_label:50s}: {count:>5} ({pct:>5.1f}%)")
        
        # Feature profiles by group
        report.append("\n" + "="*80)
        report.append("SOCIO-DEMOGRAPHIC PROFILES BY GROUP")
        report.append("="*80)
        
        for group in ['Below_Min', 'Feasible_Range', 'Above_Max']:
            group_data = df[df['ap_group'] == group]
            
            if len(group_data) == 0:
                continue
            
            report.append(f"\n{'-'*80}")
            report.append(f"{group.replace('_', ' ').upper()} (n={len(group_data)})")
            report.append(f"{'-'*80}")
            
            for feature in available_features:
                mean_val = group_data[feature].mean()
                std_val = group_data[feature].std()
                label = self.feature_labels.get(feature, feature)
                report.append(f"  {label:30s}: {mean_val:.3f} ± {std_val:.3f}")
        
        return "\n".join(report)


def analyze_propensity_groups(df: pd.DataFrame, 
                              tau_AP_min: float = 0.4,
                              tau_AP_max: float = 0.8,
                              ap_column: str = 'final_adoption_propensity',
                              output_path: Optional[str] = None,
                              region_name: str = "Region") -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Convenience function to perform complete propensity group analysis with PCA.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with adoption propensity and socio-demographic scores
    tau_AP_min : float
        Minimum adoption propensity threshold
    tau_AP_max : float
        Maximum adoption propensity threshold
    ap_column : str
        Name of adoption propensity column
    output_path : str, optional
        Path to save visualization
    region_name : str
        Name of region for labeling
        
    Returns
    -------
    df_categorized : pd.DataFrame
        Input dataframe with 'ap_group', 'PC1', 'PC2', (and 'PC3' if applicable) columns added
    summary_stats : pd.DataFrame
        Summary statistics by group
    fig : plt.Figure
        Generated visualization figure
    """
    # Initialize analyzer
    analyzer = PropensityGroupAnalyzer(tau_AP_min=tau_AP_min, tau_AP_max=tau_AP_max)
    
    # Categorize areas
    df_categorized = analyzer.categorize_areas(df, ap_column=ap_column)
    
    # Perform PCA and add coordinates to dataframe
    pca_features, pca, scaler = analyzer.perform_pca_analysis(df_categorized)
    df_categorized['PC1'] = pca_features[:, 0]
    df_categorized['PC2'] = pca_features[:, 1]
    if pca_features.shape[1] > 2:
        df_categorized['PC3'] = pca_features[:, 2]
    
    # Add PCA variance explained as attributes (for reference)
    df_categorized.attrs['pca_variance_explained'] = pca.explained_variance_ratio_
    df_categorized.attrs['pca_n_components'] = pca.n_components_
    
    # Compute statistics
    summary_stats = analyzer.compute_group_statistics(df_categorized)
    
    # Create visualization
    fig = analyzer.visualize_group_profiles(df_categorized, 
                                           output_path=output_path,
                                           region_name=region_name)
    
    # Generate text report
    report = analyzer.generate_summary_report(df_categorized, region_name=region_name)
    print(report)
    
    # Print PCA summary
    print(f"\n{'='*80}")
    print("PCA SUMMARY")
    print(f"{'='*80}")
    print(f"Number of components: {pca.n_components_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {var*100:.1f}%")
    
    # Create separate biplot if output path provided
    if output_path:
        biplot_path = output_path.replace('.png', '_biplot.png')
        biplot_fig = analyzer.create_biplot(df_categorized, 
                                           output_path=biplot_path,
                                           region_name=region_name)
        import matplotlib.pyplot as plt
        plt.close(biplot_fig)
    
    return df_categorized, summary_stats, fig

