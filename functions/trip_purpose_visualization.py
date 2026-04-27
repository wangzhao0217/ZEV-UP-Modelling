"""
Trip Purpose Analysis Visualization Module

This module provides comprehensive visualization functions for verifying Stage 3:
Trip Purpose Analysis results, including purpose weight distributions, range
feasibility, charging strategies, and weighted suitability metrics.

Author: EV Modeling Project
Date: 2025-11-02
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from pathlib import Path


# Purpose name mapping: data names -> standardized config names
PURPOSE_NAME_MAPPING = {
    # Exact matches (no change needed)
    'Commuting': 'Commuting',
    'Education': 'Education',
    'Escort': 'Escort',
    'Other personal business': 'Other personal business',
    'Business': 'Business',
    'Visiting friends or relatives': 'Visiting friends or relatives',
    
    # Case/spelling variations
    'shopping': 'Shopping',
    'commuting': 'Commuting',
    'education': 'Education',
    'escort': 'Escort',
    'business': 'Business',
    
    # Capitalization variations
    'Other Journey': 'Other journey',
    'other journey': 'Other journey',
    'Sport/Entertainment': 'Sport/entertainment',
    'sport/entertainment': 'Sport/entertainment',
    'Eating/Drinking': 'Eating/drinking',
    'eating/drinking': 'Eating/drinking',
    
    # Different text variations
    'Visit Hospital or other health': 'Health visits',
    'Health Visits': 'Health visits',
    'health visits': 'Health visits',
    'Hospital visits': 'Health visits',
    
    # Hyphenation variations
    'Holiday/daytrip': 'Holiday/day-trip',
    'Holiday/day trip': 'Holiday/day-trip',
    'holiday/daytrip': 'Holiday/day-trip',
    
    # Social visits variations
    'Social Visits': 'Social visits',
    'social visits': 'Social visits',
    
    # Recreation variations
    'Recreation': 'Recreation',
    'recreation': 'Recreation',
}


def standardize_purpose_names(df: pd.DataFrame, 
                              purpose_column: str = 'purpose',
                              inplace: bool = False) -> pd.DataFrame:
    """
    Standardize trip purpose names to match configuration naming.
    
    Maps various naming conventions in the data to the standardized
    purpose names used in scoring_weights.json configuration.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trip purpose data
    purpose_column : str
        Name of column containing purpose names
    inplace : bool
        If True, modify DataFrame in place
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized purpose names
        
    Examples
    --------
    >>> df = pd.DataFrame({'purpose': ['shopping', 'Commuting', 'Visit Hospital or other health']})
    >>> standardized_df = standardize_purpose_names(df)
    >>> print(standardized_df['purpose'].tolist())
    ['Shopping', 'Commuting', 'Health visits']
    """
    if not inplace:
        df = df.copy()
    
    if purpose_column not in df.columns:
        raise ValueError(f"Column '{purpose_column}' not found in DataFrame")
    
    # Apply mapping
    df[purpose_column] = df[purpose_column].map(
        lambda x: PURPOSE_NAME_MAPPING.get(x, x)
    )
    
    # Log any unmapped purposes
    unique_purposes = df[purpose_column].unique()
    unmapped = [p for p in unique_purposes if p not in PURPOSE_NAME_MAPPING.values()]
    if unmapped:
        print(f"Warning: {len(unmapped)} purpose name(s) not standardized: {unmapped}")
    
    return df


class TripPurposeVisualizer:
    """
    Visualizer for trip purpose analysis verification.
    
    Generates comprehensive figures to validate trip purpose weighting,
    range feasibility outcomes, and weighted suitability scores.
    """
    
    def __init__(self, purpose_weights_config: Dict):
        """
        Initialize visualizer with purpose weight configuration.
        
        Parameters
        ----------
        purpose_weights_config : dict
            Configuration dictionary containing R_p, P_p, D_p, S_p weights
            for each trip purpose
        """
        self.purpose_weights_config = purpose_weights_config
        
        # Define purpose display order (by importance/frequency)
        self.purpose_order = [
            'Commuting',
            'Education', 
            'Shopping',
            'Health visits',
            'Escort',
            'Other personal business',
            'Business',
            'Eating/drinking',
            'Sport/entertainment',
            'Visiting friends/relatives',
            'Social visits',
            'Other journey',
            'Recreation',
            'Holiday/day-trip'
        ]
        
        # Color palette for purposes - using high-contrast distinctive colors
        # Designed for maximum visual distinction across 12+ categories
        self.purpose_colors = {
            'Commuting': '#E41A1C',          # Bright red (essential)
            'Education': '#377EB8',          # Blue
            'Shopping': '#4DAF4A',           # Green
            'Health visits': '#984EA3',      # Purple
            'Escort': '#FF7F00',             # Orange
            'Other personal business': '#FFFF33',  # Yellow
            'Business': '#A65628',           # Brown
            'Eating/drinking': '#F781BF',    # Pink
            'Sport/entertainment': '#999999', # Gray
            'Visiting friends/relatives': '#00CED1',  # Dark turquoise
            'Social visits': '#FFD700',      # Gold
            'Other journey': '#8B4513',      # Saddle brown
            'Recreation': '#00FA9A',         # Medium spring green
            'Holiday/day-trip': '#9370DB'    # Medium purple (discretionary)
        }
    
    def create_purpose_weight_overview(self, 
                                      df: pd.DataFrame,
                                      output_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive overview of purpose weights and components.
        
        Generates a 3-panel figure showing:
        - Panel A: Purpose weight distribution by purpose
        - Panel B: Component weights (R_p, P_p, D_p, S_p) stacked bars
        - Panel C: Component weight heatmap
        
        Parameters
        ----------
        df : pd.DataFrame
            Trip data with 'purpose' and 'purpose_weight' columns
        output_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Prepare data
        df_sorted = df.copy()
        if 'purpose' not in df_sorted.columns or 'purpose_weight' not in df_sorted.columns:
            print("Warning: Required columns missing for purpose weight overview")
            return fig
        
        # Panel A: Violin plot of purpose weights
        ax = axes[0]
        purposes_in_data = [p for p in self.purpose_order if p in df_sorted['purpose'].unique()]
        
        sns.violinplot(data=df_sorted[df_sorted['purpose'].isin(purposes_in_data)],
                      y='purpose', x='purpose_weight', 
                      order=purposes_in_data,
                      palette=self.purpose_colors,
                      ax=ax, inner='box')
        
        ax.set_xlabel('Purpose Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trip Purpose', fontsize=12, fontweight='bold')
        ax.set_title('A) Purpose Weight Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        # Panel B: Stacked bar chart of component weights
        ax = axes[1]
        
        # Extract component weights from config
        component_data = []
        for purpose in purposes_in_data:
            if purpose in self.purpose_weights_config:
                weights = self.purpose_weights_config[purpose]
                component_data.append({
                    'purpose': purpose,
                    'Regularity': weights.get('R_p', 0),
                    'Predictability': weights.get('P_p', 0),
                    'Distance': weights.get('D_p', 0),
                    'Scheduling': weights.get('S_p', 0)
                })
        
        if component_data:
            comp_df = pd.DataFrame(component_data)
            comp_df.set_index('purpose', inplace=True)
            
            comp_df.plot(kind='barh', stacked=True, ax=ax,
                        color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                        edgecolor='white', linewidth=0.5)
            
            ax.set_xlabel('Component Weight Sum', fontsize=12, fontweight='bold')
            ax.set_ylabel('Trip Purpose', fontsize=12, fontweight='bold')
            ax.set_title('B) Component Weights (R, P, D, S)', fontsize=14, fontweight='bold', pad=15)
            ax.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='x', alpha=0.3)
        
        # Panel C: Heatmap of component weights
        ax = axes[2]
        
        if component_data:
            heatmap_data = comp_df.T
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                       cbar_kws={'label': 'Weight Value'},
                       linewidths=0.5, ax=ax)
            
            ax.set_xlabel('Trip Purpose', fontsize=12, fontweight='bold')
            ax.set_ylabel('Component', fontsize=12, fontweight='bold')
            ax.set_title('C) Component Weight Heatmap', fontsize=14, fontweight='bold', pad=15)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Purpose weight overview saved to: {output_path}")
        
        return fig
    
    def create_range_feasibility_by_purpose(self,
                                           df: pd.DataFrame,
                                           output_path: Optional[str] = None) -> plt.Figure:
        """
        Create stacked bar chart of range feasibility by purpose.
        
        Shows proportion of trips that are range_sufficient, range_marginal,
        and range_exceeded for each trip purpose.
        
        Parameters
        ----------
        df : pd.DataFrame
            Trip data with purpose and range feasibility columns
        output_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate proportions
        purposes_in_data = [p for p in self.purpose_order if p in df['purpose'].unique()]
        
        range_data = []
        for purpose in purposes_in_data:
            purpose_df = df[df['purpose'] == purpose]
            total = len(purpose_df)
            
            if total > 0:
                range_data.append({
                    'purpose': purpose,
                    'Sufficient': (purpose_df['range_sufficient'].sum() / total) * 100,
                    'Marginal': (purpose_df['range_marginal'].sum() / total) * 100,
                    'Exceeded': (purpose_df['range_exceeded'].sum() / total) * 100,
                    'total_trips': total
                })
        
        if not range_data:
            print("Warning: No range feasibility data available")
            return fig
        
        range_df = pd.DataFrame(range_data)
        range_df.set_index('purpose', inplace=True)
        
        # Create stacked bar chart
        range_df[['Sufficient', 'Marginal', 'Exceeded']].plot(
            kind='barh', stacked=True, ax=ax,
            color=['#27ae60', '#f39c12', '#e74c3c'],
            edgecolor='white', linewidth=0.5
        )
        
        # Add trip counts as text
        for i, purpose in enumerate(range_df.index):
            total = range_df.loc[purpose, 'total_trips']
            ax.text(102, i, f'n={total:,}', va='center', fontsize=9, color='gray')
        
        ax.set_xlabel('Percentage of Trips (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trip Purpose', fontsize=12, fontweight='bold')
        ax.set_title('Range Feasibility by Trip Purpose\n(100km Range Constraint)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 120)
        ax.legend(title='Range Status', bbox_to_anchor=(1.15, 1), loc='upper left')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Range feasibility by purpose saved to: {output_path}")
        
        return fig
    
    def create_purpose_distance_scatter(self,
                                       df: pd.DataFrame,
                                       output_path: Optional[str] = None) -> plt.Figure:
        """
        Create scatter plot of purpose weight vs distance.
        
        Shows relationship between trip purpose weights and distance,
        with marginal distributions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Trip data with purpose, purpose_weight, and distance_km columns
        output_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.05)
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[1:, :-1])
        
        # Marginal histograms
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
        
        # Sample data if too large
        plot_df = df.sample(n=min(50000, len(df)), random_state=42)
        
        # Main scatter with purpose colors (enhanced visibility)
        # Store label positions for direct labeling
        label_positions = []
        
        for purpose in self.purpose_order:
            if purpose in plot_df['purpose'].unique():
                purpose_data = plot_df[plot_df['purpose'] == purpose]
                ax_main.scatter(purpose_data['distance_km'], 
                              purpose_data['purpose_weight'],
                              alpha=0.6, s=15, 
                              color=self.purpose_colors.get(purpose, 'gray'),
                              edgecolors='white', linewidths=0.2)
                
                # Calculate label position (rightmost point with median y)
                # Use 95th percentile of distance to avoid outliers
                x_pos = purpose_data['distance_km'].quantile(0.95)
                y_pos = purpose_data['purpose_weight'].median()
                
                label_positions.append({
                    'purpose': purpose,
                    'x': x_pos,
                    'y': y_pos,
                    'color': self.purpose_colors.get(purpose, 'gray')
                })
        
        # Add direct labels on the right side of points with collision avoidance
        texts = []
        for label_info in label_positions:
            text = ax_main.text(label_info['x'] + 1, label_info['y'], 
                        label_info['purpose'],
                        fontsize=8, 
                        va='center',
                        ha='left',
                        color=label_info['color'],
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor=label_info['color'],
                                alpha=0.9,
                                linewidth=1.5))
            texts.append(text)
        
        # Use adjustText if available for automatic label positioning
        try:
            from adjustText import adjust_text
            adjust_text(texts,
                       ax=ax_main,
                       only_move={'text': 'xy'},
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                       expand_text=(1.2, 1.3),
                       expand_points=(1.2, 1.3),
                       force_text=(0.5, 0.5),
                       force_points=(0.3, 0.3),
                       lim=500)
        except ImportError:
            # Manual collision avoidance if adjustText not available
            pass
        
        ax_main.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Purpose Weight', fontsize=12, fontweight='bold')
        ax_main.grid(alpha=0.3)
        
        # Top marginal: distance distribution
        ax_top.hist(plot_df['distance_km'], bins=50, color='steelblue', 
                   alpha=0.6, edgecolor='white')
        ax_top.set_ylabel('Count', fontsize=10)
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(axis='y', alpha=0.3)
        
        # Right marginal: purpose weight distribution
        ax_right.hist(plot_df['purpose_weight'], bins=50, 
                     orientation='horizontal', color='coral',
                     alpha=0.6, edgecolor='white')
        ax_right.set_xlabel('Count', fontsize=10)
        ax_right.tick_params(labelleft=False)
        ax_right.grid(axis='x', alpha=0.3)
        
        fig.suptitle('Purpose Weight vs Distance Relationship', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Purpose-distance scatter saved to: {output_path}")
        
        return fig
    
    def create_weighted_suitability_ridges(self,
                                          df: pd.DataFrame,
                                          output_path: Optional[str] = None) -> plt.Figure:
        """
        Create ridge plot of weighted suitability by purpose.
        
        Shows distribution of weighted_suitability scores across
        different trip purposes, ordered by median suitability.
        
        Parameters
        ----------
        df : pd.DataFrame
            Trip data with purpose and weighted_suitability columns
        output_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate median suitability for ordering
        purposes_in_data = [p for p in self.purpose_order if p in df['purpose'].unique()]
        median_suitability = df.groupby('purpose')['weighted_suitability'].median()
        purposes_sorted = median_suitability.sort_values(ascending=False).index.tolist()
        purposes_sorted = [p for p in purposes_sorted if p in purposes_in_data]
        
        # Create violin plots (ridge-like)
        positions = list(range(len(purposes_sorted)))
        
        for i, purpose in enumerate(purposes_sorted):
            purpose_data = df[df['purpose'] == purpose]['weighted_suitability'].dropna()
            
            if len(purpose_data) > 0:
                parts = ax.violinplot([purpose_data], positions=[i],
                                     vert=False, widths=0.7,
                                     showmeans=True, showmedians=True)
                
                # Color the violin
                for pc in parts['bodies']:
                    pc.set_facecolor(self.purpose_colors.get(purpose, 'gray'))
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(purposes_sorted)
        ax.set_xlabel('Weighted Suitability Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trip Purpose', fontsize=12, fontweight='bold')
        ax.set_title('Weighted Suitability Distribution by Purpose\n(Ordered by Median)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=df['weighted_suitability'].median(), color='red', 
                  linestyle='--', linewidth=2, alpha=0.5, label='Overall Median')
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Weighted suitability ridges saved to: {output_path}")
        
        return fig
    
    def create_purpose_clustering_analysis(self,
                                          df: pd.DataFrame,
                                          output_path: Optional[str] = None) -> plt.Figure:
        """
        Create PCA biplot visualization for purpose clustering.
        
        Shows PCA biplot with all component relationships, colored by
        suitability category.
        
        Parameters
        ----------
        df : pd.DataFrame
            Trip data with purpose and weighted_suitability columns
        output_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Generated figure
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Aggregate data by purpose (use all purposes in data)
        purpose_summary = []
        purposes_in_data = sorted(df['purpose'].unique())
        
        for purpose in purposes_in_data:
            purpose_df = df[df['purpose'] == purpose]
            
            # Try to find weights in config (with fallback for naming variations)
            weights = None
            config_key = None
            
            if purpose in self.purpose_weights_config:
                weights = self.purpose_weights_config[purpose]
                config_key = purpose
            else:
                # Try alternative names (handle hyphen variations, case, etc.)
                alt_names = [
                    purpose.replace('/day-trip', '/daytrip'),  # Holiday/day-trip -> Holiday/daytrip
                    purpose.replace('/daytrip', '/day-trip'),
                    purpose.replace(' or ', '/'),
                    purpose.replace('/', ' or ')
                ]
                for alt_name in alt_names:
                    if alt_name in self.purpose_weights_config:
                        weights = self.purpose_weights_config[alt_name]
                        config_key = alt_name
                        break
            
            if not weights:
                print(f"    ⚠ No config found for: {purpose}")
                print(f"       Available config keys: {list(self.purpose_weights_config.keys())}")
                continue
            
            if weights:
                
                # Calculate charging flexibility
                charging_flex = (
                    weights.get('D_p', 0) * 0.4 +
                    (1 - weights.get('S_p', 0)) * 0.3 +
                    0.5 * 0.2 +  # Placeholder for destination
                    0.5 * 0.1    # Placeholder for home
                )
                
                # Calculate suitability
                suitability = (
                    weights.get('R_p', 0) * 0.35 +
                    weights.get('P_p', 0) * 0.35 +
                    charging_flex * 0.30
                )
                
                purpose_summary.append({
                    'purpose': purpose,
                    'regularity': weights.get('R_p', 0),
                    'predictability': weights.get('P_p', 0),
                    'dwell_flex': weights.get('D_p', 0),
                    'temporal_sens': weights.get('S_p', 0),
                    'charging_flex': charging_flex,
                    'suitability': suitability,
                    'trip_count': len(purpose_df)
                })
        
        if not purpose_summary:
            print("Warning: No purpose clustering data available")
            return fig
        
        summary_df = pd.DataFrame(purpose_summary)
        
        # Report number of purposes
        print(f"  Plotting {len(summary_df)} purposes in clustering analysis:")
        for purpose in summary_df['purpose']:
            print(f"    • {purpose}")
        
        # Classify by suitability
        summary_df['category'] = summary_df['suitability'].apply(
            lambda x: 'High (≥0.8)' if x >= 0.8 else 'Moderate (0.6-0.8)' if x >= 0.6 else 'Low (<0.6)'
        )
        
        category_colors = {
            'High (≥0.8)': '#27ae60',
            'Moderate (0.6-0.8)': '#f39c12',
            'Low (<0.6)': '#e74c3c'
        }
        
        # PCA Biplot
        
        # Prepare data for PCA
        feature_cols = ['regularity', 'predictability', 'charging_flex', 
                       'dwell_flex', 'temporal_sens']
        X = summary_df[feature_cols].values
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot points
        for category in ['High (≥0.8)', 'Moderate (0.6-0.8)', 'Low (<0.6)']:
            cat_mask = summary_df['category'] == category
            if cat_mask.any():
                sizes = (summary_df[cat_mask]['trip_count'] / summary_df['trip_count'].max()) * 1000 + 200
                ax.scatter(X_pca[cat_mask, 0], X_pca[cat_mask, 1],
                           s=sizes, alpha=0.7,
                           color=category_colors[category],
                           edgecolors='white', linewidth=2,
                           label=category, zorder=10)
        
        # Add labels with collision avoidance
        # Try to use adjustText if available, otherwise use manual adjustment
        try:
            from adjustText import adjust_text
            
            texts = []
            for i, row in summary_df.iterrows():
                text = ax.text(X_pca[i, 0], X_pca[i, 1], row['purpose'],
                              fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor=category_colors[row['category']],
                                      alpha=0.9,
                                      linewidth=1.5),
                              zorder=11)
                texts.append(text)
            
            # Adjust text positions to avoid overlaps
            adjust_text(texts, 
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                       expand_points=(1.5, 1.5),
                       expand_text=(1.2, 1.2),
                       force_points=0.5,
                       force_text=0.5,
                       ax=ax)
            
        except ImportError:
            # Fallback: manual label positioning with offset adjustment
            print("  Note: adjustText not available, using manual positioning")
            
            label_positions = []
            min_distance = 0.3  # Minimum distance between labels
            
            for i, row in summary_df.iterrows():
                x, y = X_pca[i, 0], X_pca[i, 1]
                
                # Find non-overlapping position
                offset_x, offset_y = 0.15, 0.15
                test_x, test_y = x + offset_x, y + offset_y
                
                # Check for collisions and adjust
                attempts = 0
                max_attempts = 20
                offsets = [(0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15),
                          (0.25, 0.05), (-0.25, 0.05), (0.05, 0.25), (0.05, -0.25),
                          (0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)]
                
                for offset_x, offset_y in offsets:
                    test_x, test_y = x + offset_x, y + offset_y
                    too_close = False
                    
                    for prev_x, prev_y in label_positions:
                        dist = np.sqrt((test_x - prev_x)**2 + (test_y - prev_y)**2)
                        if dist < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        break
                
                label_positions.append((test_x, test_y))
                
                # Draw connecting line if label moved far
                dist_moved = np.sqrt(offset_x**2 + offset_y**2)
                if dist_moved > 0.2:
                    ax.plot([x, test_x], [y, test_y], 
                            'gray', linewidth=0.5, alpha=0.5, zorder=9)
                
                ax.text(test_x, test_y, row['purpose'],
                        fontsize=9, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor=category_colors[row['category']],
                                alpha=0.9,
                                linewidth=1.5),
                        zorder=11)
        
        # Add component loadings as arrows with labels (same as propensity_group_analysis.py)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Scale factor for arrows
        pc1_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        pc2_range = X_pca[:, 1].max() - X_pca[:, 1].min()
        data_scale = max(pc1_range, pc2_range)
        scale_factor = data_scale * 0.35
        
        feature_labels = {
            'regularity': 'Regularity',
            'predictability': 'Predictability',
            'charging_flex': 'Charging Flex',
            'dwell_flex': 'Dwell Flex',
            'temporal_sens': 'Temporal Sens'
        }
        
        # Calculate label positions with same logic as propensity_group_analysis
        label_info = []
        for i, feature in enumerate(feature_cols):
            arrow_x = loadings[i, 0] * scale_factor
            arrow_y = loadings[i, 1] * scale_factor
            
            angle = np.arctan2(arrow_y, arrow_x)
            angle_deg = np.degrees(angle)
            
            # Position along arrow direction (same as propensity_group_analysis)
            label_x = arrow_x * 1.08
            label_y = arrow_y * 1.08
            
            # Calculate text rotation to match arrow angle
            text_angle = angle_deg
            
            # Determine alignment based on arrow direction
            if -90 <= angle_deg <= 90:
                ha = 'left'  # Arrow points right
            else:
                ha = 'right'  # Arrow points left
            
            # Normalize rotation to avoid upside-down text
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
                'text_angle': text_angle
            })
        
        # Draw arrows and labels
        for info in label_info:
            # Draw arrow
            ax.arrow(0, 0, info['arrow_x'], info['arrow_y'],
                     head_width=data_scale*0.025,
                     head_length=data_scale*0.025,
                     fc='#8B4513', ec='#8B4513',
                     alpha=0.9, linewidth=2.0, zorder=10,
                     length_includes_head=True)
            
            # Manual perpendicular offset for overlapping labels
            label_x_adj = info['label_x']
            label_y_adj = info['label_y']
            
            # Calculate perpendicular direction (rotate 90 degrees)
            angle = np.arctan2(info['arrow_y'], info['arrow_x'])
            perp_x = -np.sin(angle)  # Perpendicular to arrow
            perp_y = np.cos(angle)
            
            offset_dist = data_scale * 0.02  # Perpendicular offset distance
            
            if info['feature'] == 'regularity':
                # Move Regularity higher (positive perpendicular)
                label_x_adj = info['label_x'] + perp_x * offset_dist
                label_y_adj = info['label_y'] + perp_y * offset_dist
            elif info['feature'] == 'predictability':
                # Move Predictability lower (negative perpendicular)
                label_x_adj = info['label_x'] - perp_x * offset_dist
                label_y_adj = info['label_y'] - perp_y * offset_dist
            
            # Draw label rotated to match arrow
            ax.text(label_x_adj, label_y_adj, feature_labels[info['feature']],
                    fontsize=11, fontweight='bold',
                    ha=info['ha'], va=info['va'],
                    color='#8B4513',
                    rotation=info['text_angle'],
                    rotation_mode='anchor',
                    zorder=11,
                    bbox=dict(boxstyle='round,pad=0.25',
                            facecolor='white',
                            edgecolor='#8B4513',
                            alpha=0.9,
                            linewidth=1.0))
        
        var_explained_1 = pca.explained_variance_ratio_[0] * 100
        var_explained_2 = pca.explained_variance_ratio_[1] * 100
        
        ax.set_xlabel(f'PC1 ({var_explained_1:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_explained_2:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.set_title('PCA Biplot: Multi-component Analysis',
                     fontsize=15, fontweight='bold', pad=15)
        
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Set axis limits to -3 to 3 for better visibility
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        # Position legend at bottom in one row
        ax.legend(title='Suitability Category', 
                  loc='upper center', 
                  bbox_to_anchor=(0.5, -0.12),
                  ncol=3,
                  fontsize=11,
                  markerscale=0.5,
                  title_fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
        
        # Adjust layout to make room for legend at bottom
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legend at the bottom
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Purpose clustering analysis saved to: {output_path}")
        
        return fig


def visualize_trip_purpose_analysis(df: pd.DataFrame,
                                    purpose_weights_config: Dict,
                                    output_dir: Path,
                                    region_name: str = "Region") -> Dict[str, plt.Figure]:
    """
    Generate comprehensive trip purpose analysis visualizations.
    
    Creates all verification figures for Stage 3: Trip Purpose Analysis.
    Automatically standardizes purpose names to match configuration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trip data from Stage 3 with purpose, purpose_weight, range, and
        suitability columns
    purpose_weights_config : dict
        Configuration dictionary with R_p, P_p, D_p, S_p weights per purpose
    output_dir : Path
        Directory to save output figures
    region_name : str
        Name of region for plot titles
        
    Returns
    -------
    dict
        Dictionary mapping figure names to Figure objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TRIP PURPOSE ANALYSIS VISUALIZATION: {region_name}")
    print(f"{'='*80}\n")
    
    # Standardize purpose names before processing
    print("Standardizing trip purpose names...")
    df_standardized = standardize_purpose_names(df, purpose_column='purpose', inplace=False)
    
    # Report standardization results
    original_purposes = df['purpose'].unique()
    standardized_purposes = df_standardized['purpose'].unique()
    print(f"  Original unique purposes: {len(original_purposes)}")
    print(f"  Standardized unique purposes: {len(standardized_purposes)}")
    
    if len(original_purposes) != len(standardized_purposes):
        print(f"  ✓ Merged {len(original_purposes) - len(standardized_purposes)} duplicate purposes")
    
    visualizer = TripPurposeVisualizer(purpose_weights_config)
    
    figures = {}
    
    # Figure 1: Purpose Weight Overview
    print("Generating Figure 1: Purpose Weight Overview...")
    fig1 = visualizer.create_purpose_weight_overview(
        df_standardized, output_path=str(output_dir / "purpose_weight_overview.png")
    )
    figures['purpose_weight_overview'] = fig1
    
    # Figure 2: Range Feasibility by Purpose
    print("Generating Figure 2: Range Feasibility by Purpose...")
    fig2 = visualizer.create_range_feasibility_by_purpose(
        df_standardized, output_path=str(output_dir / "range_feasibility_by_purpose.png")
    )
    figures['range_feasibility_by_purpose'] = fig2
    
    # Figure 3: Purpose-Distance Scatter
    print("Generating Figure 3: Purpose Weight vs Distance...")
    fig3 = visualizer.create_purpose_distance_scatter(
        df_standardized, output_path=str(output_dir / "purpose_distance_scatter.png")
    )
    figures['purpose_distance_scatter'] = fig3
    
    # Figure 4: Weighted Suitability Ridges
    print("Generating Figure 4: Weighted Suitability Distribution...")
    fig4 = visualizer.create_weighted_suitability_ridges(
        df_standardized, output_path=str(output_dir / "weighted_suitability_ridges.png")
    )
    figures['weighted_suitability_ridges'] = fig4
    
    # Figure 5: Purpose Clustering Analysis
    print("Generating Figure 5: Purpose Clustering Analysis...")
    fig5 = visualizer.create_purpose_clustering_analysis(
        df_standardized, output_path=str(output_dir / "purpose_clustering_analysis.png")
    )
    figures['purpose_clustering_analysis'] = fig5
    
    print(f"\n✓ All trip purpose visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return figures

