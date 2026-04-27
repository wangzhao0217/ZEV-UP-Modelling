"""
Charging Infrastructure Visualization Module

This module provides comprehensive visualization tools for analyzing
charging infrastructure accessibility, capacity, and spatial distribution.

Author: EV Modeling Team
Created: 2025-11-03
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path


class ChargingInfrastructureVisualizer:
    """
    Comprehensive visualization suite for charging infrastructure analysis.
    
    This class provides methods to create various visualizations analyzing
    charging infrastructure accessibility, spatial distribution, and capacity.
    """
    
    def __init__(self, data: gpd.GeoDataFrame):
        """
        Initialize visualizer with charging infrastructure data.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            GeoDataFrame containing charging infrastructure analysis results
        """
        self.data = data.copy()
        
        # Define color schemes
        self.accessibility_colors = {
            'excellent': '#2ca02c',  # Green
            'good': '#98df8a',       # Light green
            'fair': '#ff7f0e',       # Orange
            'poor': '#d62728'        # Red
        }
        
        # Category order for plotting
        self.accessibility_order = ['poor', 'fair', 'good', 'excellent']
        
    def create_comprehensive_dashboard(self, 
                                      output_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive 9-panel dashboard of charging infrastructure analysis.
        
        Panels:
        - A) Spatial map of gap severity
        - B) Gap severity category distribution  
        - C) Bottleneck identification map
        - D) Charger density (1km, 2km, 5km)
        - E) Accessibility vs public charging reliance
        - F) Infrastructure vs capacity factors
        - G) Gap severity vs adoption propensity
        - H) Distance distribution by accessibility category
        - I) Summary metrics table
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # A) Gap severity spatial map (most important new metric)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_gap_severity_spatial(ax1)
        
        # B) Gap severity distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gap_severity_distribution(ax2)
        
        # C) Bottleneck identification
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_bottleneck_map(ax3)
        
        # D) Charger density
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_charger_density(ax4)
        
        # E) Accessibility vs reliance (with bottlenecks highlighted)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_accessibility_vs_reliance_enhanced(ax5)
        
        # F) Infrastructure vs capacity
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_infrastructure_vs_capacity(ax6)
        
        # G) Gap severity vs adoption propensity (NEW)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_gap_severity_vs_adoption(ax7)
        
        # H) Distance by category (NEW)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_distance_by_category(ax8)
        
        # I) Summary metrics (NEW)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_summary_metrics(ax9)
        
        plt.suptitle('Comprehensive Charging Infrastructure & Gap Analysis Dashboard',
                    fontsize=20, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✓ Dashboard saved to: {output_path}")
        
        return fig
    
    def _plot_spatial_accessibility(self, ax: plt.Axes):
        """Plot spatial distribution of charging accessibility."""
        # Create color mapping
        self.data['color'] = self.data['charging_accessibility_category'].map(
            self.accessibility_colors
        )
        
        # Plot with colors
        self.data.plot(ax=ax, color=self.data['color'], 
                      edgecolor='white', linewidth=0.5)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.accessibility_colors[cat], 
                 edgecolor='white', label=cat.capitalize())
            for cat in self.accessibility_order
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 title='Accessibility', frameon=True, fancybox=True)
        
        ax.set_title('A) Spatial Distribution of Charging Accessibility',
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    def _plot_category_distribution(self, ax: plt.Axes):
        """Plot distribution of accessibility categories."""
        category_counts = self.data['charging_accessibility_category'].value_counts()
        
        # Order by accessibility level
        ordered_cats = [cat for cat in self.accessibility_order 
                       if cat in category_counts.index]
        counts = [category_counts[cat] for cat in ordered_cats]
        colors = [self.accessibility_colors[cat] for cat in ordered_cats]
        
        bars = ax.bar(range(len(ordered_cats)), counts, color=colors,
                     edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(self.data)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(ordered_cats)))
        ax.set_xticklabels([cat.capitalize() for cat in ordered_cats],
                          fontsize=11)
        ax.set_ylabel('Number of Data Zones', fontsize=11, fontweight='bold')
        ax.set_title('B) Accessibility Category Distribution',
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    def _plot_distance_distribution(self, ax: plt.Axes):
        """Plot histogram of distance to nearest charger."""
        distances = self.data['nearest_charger_distance'] / 1000  # Convert to km
        
        ax.hist(distances, bins=30, color='steelblue', 
               edgecolor='white', alpha=0.7)
        
        # Add median line
        median_dist = distances.median()
        ax.axvline(median_dist, color='red', linestyle='--', 
                  linewidth=2, label=f'Median: {median_dist:.1f} km')
        
        # Add mean line
        mean_dist = distances.mean()
        ax.axvline(mean_dist, color='orange', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_dist:.1f} km')
        
        ax.set_xlabel('Distance to Nearest Charger (km)', 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Data Zones', fontsize=11, fontweight='bold')
        ax.set_title('C) Distance to Nearest Charger Distribution',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=10, frameon=True, fancybox=True)
        ax.grid(alpha=0.3)
    
    def _plot_charger_density(self, ax: plt.Axes):
        """Plot charger density at different radii."""
        radii = ['1km', '2km', '5km']
        columns = ['chargers_within_1km', 'chargers_within_2km', 
                  'chargers_within_5km']
        
        # Calculate statistics
        stats_data = []
        for radius, col in zip(radii, columns):
            stats_data.append({
                'radius': radius,
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'max': self.data[col].max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Plot grouped bars
        x = np.arange(len(radii))
        width = 0.25
        
        bars1 = ax.bar(x - width, stats_df['median'], width, 
                      label='Median', color='steelblue', edgecolor='white')
        bars2 = ax.bar(x, stats_df['mean'], width,
                      label='Mean', color='orange', edgecolor='white')
        bars3 = ax.bar(x + width, stats_df['max'], width,
                      label='Max', color='green', edgecolor='white', alpha=0.7)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Search Radius', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Chargers', fontsize=11, fontweight='bold')
        ax.set_title('D) Charger Density by Search Radius',
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(radii)
        ax.legend(fontsize=10, frameon=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    def _plot_accessibility_vs_reliance(self, ax: plt.Axes):
        """Plot accessibility score vs public charging reliance."""
        # Color by category
        for category in self.accessibility_order:
            mask = self.data['charging_accessibility_category'] == category
            if mask.any():
                ax.scatter(self.data.loc[mask, 'accessibility_score'],
                          self.data.loc[mask, 'public_charging_reliance'],
                          c=self.accessibility_colors[category],
                          label=category.capitalize(),
                          alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
        
        # Add trend line
        x = self.data['accessibility_score']
        y = self.data['public_charging_reliance']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2,
               label=f'Trend (r={x.corr(y):.2f})')
        
        ax.set_xlabel('Accessibility Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Public Charging Reliance', fontsize=11, fontweight='bold')
        ax.set_title('E) Accessibility vs Public Charging Reliance',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9, frameon=True, fancybox=True, loc='best')
        ax.grid(alpha=0.3)
    
    def _plot_infrastructure_vs_capacity(self, ax: plt.Axes):
        """Plot infrastructure factor vs capacity factor."""
        # Color by category
        for category in self.accessibility_order:
            mask = self.data['charging_accessibility_category'] == category
            if mask.any():
                ax.scatter(self.data.loc[mask, 'infrastructure_factor'],
                          self.data.loc[mask, 'capacity_factor'],
                          c=self.accessibility_colors[category],
                          label=category.capitalize(),
                          alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
        
        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1,
               label='Equal Infrastructure & Capacity')
        
        ax.set_xlabel('Infrastructure Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel('Capacity Factor', fontsize=11, fontweight='bold')
        ax.set_title('F) Infrastructure vs Capacity Factors',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9, frameon=True, fancybox=True, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    
    def create_pca_biplot(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Create PCA biplot to understand multivariate relationships.
        
        Shows the relationships between all numeric infrastructure variables
        using PCA with accessibility category coloring.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Select numeric features
        features = [
            'nearest_charger_distance',
            'chargers_within_1km',
            'chargers_within_2km',
            'chargers_within_5km',
            'accessibility_score',
            'public_charging_reliance',
            'infrastructure_factor',
            'capacity_factor'
        ]
        
        # Prepare data
        X = self.data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot data points by category
        for category in self.accessibility_order:
            mask = self.data['charging_accessibility_category'] == category
            if mask.any():
                # Size by number of chargers within 5km
                sizes = (self.data.loc[mask, 'chargers_within_5km'] + 1) * 20
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=self.accessibility_colors[category],
                          label=category.capitalize(),
                          alpha=0.6, s=sizes, edgecolors='white', linewidths=0.5)
        
        # Plot feature vectors
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        scale_factor = 3.5  # Scale arrows for visibility
        
        feature_labels = {
            'nearest_charger_distance': 'Nearest Distance',
            'chargers_within_1km': 'Chargers (1km)',
            'chargers_within_2km': 'Chargers (2km)',
            'chargers_within_5km': 'Chargers (5km)',
            'accessibility_score': 'Accessibility',
            'public_charging_reliance': 'Public Reliance',
            'infrastructure_factor': 'Infrastructure',
            'capacity_factor': 'Capacity'
        }
        
        # Draw arrows and labels
        for i, feature in enumerate(features):
            ax.arrow(0, 0, 
                    loadings[i, 0] * scale_factor, 
                    loadings[i, 1] * scale_factor,
                    head_width=0.15, head_length=0.15,
                    fc='darkred', ec='darkred', alpha=0.7,
                    linewidth=2, zorder=10)
            
            # Label at arrow tip
            label_x = loadings[i, 0] * scale_factor * 1.15
            label_y = loadings[i, 1] * scale_factor * 1.15
            
            ax.text(label_x, label_y, feature_labels[feature],
                   fontsize=11, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='white',
                           edgecolor='darkred',
                           alpha=0.9,
                           linewidth=1.5),
                   zorder=11)
        
        # Formatting
        var_explained_1 = pca.explained_variance_ratio_[0] * 100
        var_explained_2 = pca.explained_variance_ratio_[1] * 100
        
        ax.set_xlabel(f'PC1 ({var_explained_1:.1f}% variance)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_explained_2:.1f}% variance)',
                     fontsize=14, fontweight='bold')
        ax.set_title('Charging Infrastructure: PCA Biplot Analysis',
                    fontsize=16, fontweight='bold', pad=15)
        
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        # Legend
        ax.legend(title='Accessibility Category',
                 loc='upper right',
                 fontsize=11,
                 markerscale=1.5,
                 title_fontsize=12,
                 frameon=True,
                 fancybox=True,
                 shadow=True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✓ PCA biplot saved to: {output_path}")
        
        return fig
    
    def create_detailed_metrics_summary(self, 
                                       output_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed metrics summary with correlation heatmap.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Metrics for analysis
        metrics = [
            'nearest_charger_distance',
            'chargers_within_1km',
            'chargers_within_2km',
            'chargers_within_5km',
            'accessibility_score',
            'public_charging_reliance',
            'infrastructure_factor',
            'capacity_factor'
        ]
        
        # A) Correlation heatmap
        ax1 = fig.add_subplot(gs[0, :])
        corr_matrix = self.data[metrics].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, square=True,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax1, linewidths=0.5)
        
        ax1.set_title('A) Correlation Matrix of Infrastructure Metrics',
                     fontsize=14, fontweight='bold', pad=15)
        
        # B) Box plots by accessibility category
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_accessibility_boxplot(ax2)
        
        # C) Statistics table
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_statistics_table(ax3)
        
        plt.suptitle('Charging Infrastructure: Detailed Metrics Analysis',
                    fontsize=16, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✓ Metrics summary saved to: {output_path}")
        
        return fig
    
    def _plot_accessibility_boxplot(self, ax: plt.Axes):
        """Plot boxplot of accessibility score by category."""
        # Prepare data
        plot_data = []
        for category in self.accessibility_order:
            mask = self.data['charging_accessibility_category'] == category
            scores = self.data.loc[mask, 'accessibility_score'].values
            plot_data.append(scores)
        
        # Create boxplot
        bp = ax.boxplot(plot_data, labels=[cat.capitalize() 
                                           for cat in self.accessibility_order],
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red',
                                    markeredgecolor='darkred', markersize=6))
        
        # Color boxes
        for patch, category in zip(bp['boxes'], self.accessibility_order):
            patch.set_facecolor(self.accessibility_colors[category])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Accessibility Score', fontsize=11, fontweight='bold')
        ax.set_title('B) Accessibility Score Distribution by Category',
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    def _plot_statistics_table(self, ax: plt.Axes):
        """Create statistics table."""
        # Calculate statistics by category
        stats_data = []
        for category in self.accessibility_order:
            mask = self.data['charging_accessibility_category'] == category
            stats_data.append([
                category.capitalize(),
                mask.sum(),
                f"{mask.sum()/len(self.data)*100:.1f}%",
                f"{self.data.loc[mask, 'accessibility_score'].mean():.2f}",
                f"{self.data.loc[mask, 'nearest_charger_distance'].mean()/1000:.1f} km",
                f"{self.data.loc[mask, 'chargers_within_5km'].mean():.1f}"
            ])
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Category', 'Count', '%', 'Avg Score',
                                  'Avg Dist', 'Avg Chargers\n(5km)'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.12, 0.12, 0.15, 0.15, 0.18])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(6):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by category
        for i, category in enumerate(self.accessibility_order):
            table[(i+1, 0)].set_facecolor(self.accessibility_colors[category])
            table[(i+1, 0)].set_text_props(weight='bold')
            for j in range(1, 6):
                table[(i+1, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('C) Statistics by Accessibility Category',
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    def _plot_gap_severity_spatial(self, ax: plt.Axes):
        """Plot spatial distribution of infrastructure gap severity."""
        if 'gap_severity_category' not in self.data.columns:
            ax.text(0.5, 0.5, 'Gap Severity\nData Not Available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Define gap severity colors
        severity_colors = {
            'low': '#2ca02c',       # Green
            'medium': '#ff7f0e',    # Orange
            'high': '#d62728',      # Red
            'critical': '#8b0000'   # Dark red
        }
        
        # Create color mapping
        self.data['severity_color'] = self.data['gap_severity_category'].map(
            severity_colors
        )
        
        # Plot with colors
        self.data.plot(ax=ax, color=self.data['severity_color'], 
                      edgecolor='white', linewidth=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=severity_colors[cat], 
                 edgecolor='white', label=cat.capitalize())
            for cat in ['low', 'medium', 'high', 'critical']
            if cat in self.data['gap_severity_category'].values
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 title='Gap Severity\n(Investment Priority)', 
                 frameon=True, fancybox=True, fontsize=10)
        
        ax.set_title('A) Infrastructure Gap Severity (Priority Investment Zones)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    def _plot_gap_severity_distribution(self, ax: plt.Axes):
        """Plot distribution of gap severity categories."""
        if 'gap_severity_category' not in self.data.columns:
            ax.text(0.5, 0.5, 'Gap Severity\nData Not Available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return
        
        severity_counts = self.data['gap_severity_category'].value_counts()
        severity_order = ['low', 'medium', 'high', 'critical']
        severity_colors_map = {
            'low': '#2ca02c',
            'medium': '#ff7f0e',
            'high': '#d62728',
            'critical': '#8b0000'
        }
        
        # Order by severity level
        ordered_cats = [cat for cat in severity_order 
                       if cat in severity_counts.index]
        counts = [severity_counts[cat] for cat in ordered_cats]
        colors = [severity_colors_map[cat] for cat in ordered_cats]
        
        bars = ax.bar(range(len(ordered_cats)), counts, color=colors,
                     edgecolor='white', linewidth=2)
        
        # Add value labels with gap severity metrics
        for i, bar in enumerate(bars):
            height = bar.get_height()
            pct = height/len(self.data)*100
            cat = ordered_cats[i]
            
            # Calculate mean gap severity for this category
            cat_data = self.data[self.data['gap_severity_category'] == cat]
            if 'gap_severity_index' in cat_data.columns:
                mean_severity = cat_data['gap_severity_index'].mean()
                label_text = f'{int(height)}\n({pct:.1f}%)\nμ={mean_severity:.2f}'
            else:
                label_text = f'{int(height)}\n({pct:.1f}%)'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text,
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(ordered_cats)))
        ax.set_xticklabels([cat.capitalize() for cat in ordered_cats],
                          fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Data Zones', fontsize=11, fontweight='bold')
        ax.set_title('B) Gap Severity Distribution\n(μ = mean gap severity index)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    def _plot_bottleneck_map(self, ax: plt.Axes):
        """Plot spatial identification of charging bottlenecks."""
        if 'charging_bottleneck' not in self.data.columns:
            ax.text(0.5, 0.5, 'Bottleneck\nData Not Available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Create colors based on bottleneck status
        bottleneck_colors = {
            True: '#d62728',   # Red for bottlenecks
            False: '#e0e0e0'   # Gray for non-bottlenecks
        }
        
        self.data['bottleneck_color'] = self.data['charging_bottleneck'].map(
            bottleneck_colors
        )
        
        # Plot
        self.data.plot(ax=ax, color=self.data['bottleneck_color'], 
                      edgecolor='white', linewidth=0.3, alpha=0.8)
        
        # Statistics
        bottleneck_count = self.data['charging_bottleneck'].sum()
        total_count = len(self.data)
        bottleneck_pct = (bottleneck_count / total_count) * 100
        
        # Legend with statistics
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', edgecolor='white',
                 label=f'Bottleneck\n(n={bottleneck_count}, {bottleneck_pct:.1f}%)'),
            Patch(facecolor='#e0e0e0', edgecolor='white',
                 label=f'Non-bottleneck\n(n={total_count-bottleneck_count})')
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 title='Charging Bottleneck\n(High reliance +\nPoor access)', 
                 frameon=True, fancybox=True, fontsize=9)
        
        ax.set_title('C) Charging Bottleneck Identification\n(R > 0.7 AND S < 0.6)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    def _plot_accessibility_vs_reliance_enhanced(self, ax: plt.Axes):
        """Plot accessibility score vs public charging reliance with bottlenecks highlighted."""
        # Plot non-bottlenecks first
        if 'charging_bottleneck' in self.data.columns:
            non_bottleneck = self.data[~self.data['charging_bottleneck']]
            ax.scatter(non_bottleneck['accessibility_score'],
                      non_bottleneck['public_charging_reliance'],
                      c='lightgray', alpha=0.5, s=30, 
                      edgecolors='white', linewidths=0.5,
                      label='Non-bottleneck')
            
            # Plot bottlenecks
            bottleneck = self.data[self.data['charging_bottleneck']]
            ax.scatter(bottleneck['accessibility_score'],
                      bottleneck['public_charging_reliance'],
                      c='#d62728', alpha=0.7, s=60,
                      edgecolors='darkred', linewidths=1,
                      label='Bottleneck', marker='X')
        else:
            # Fallback to category colors
            for category in self.accessibility_order:
                mask = self.data['charging_accessibility_category'] == category
                if mask.any():
                    ax.scatter(self.data.loc[mask, 'accessibility_score'],
                              self.data.loc[mask, 'public_charging_reliance'],
                              c=self.accessibility_colors[category],
                              label=category.capitalize(),
                              alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
        
        # Add threshold lines
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='High reliance threshold (0.7)')
        ax.axvline(x=0.6, color='orange', linestyle='--', linewidth=1.5,
                  alpha=0.7, label='Poor access threshold (0.6)')
        
        # Add trend line for all data
        x = self.data['accessibility_score']
        y = self.data['public_charging_reliance']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x.sort_values(), p(x.sort_values()), 
               "b--", alpha=0.5, linewidth=2, label=f'Trend (r={np.corrcoef(x, y)[0,1]:.2f})')
        
        ax.set_xlabel('Accessibility Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Public Charging Reliance', fontsize=11, fontweight='bold')
        ax.set_title('E) Accessibility vs Reliance (Bottleneck Threshold)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=8, frameon=True, fancybox=True, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    
    def _plot_gap_severity_vs_adoption(self, ax: plt.Axes):
        """Plot gap severity index vs adoption propensity."""
        if 'gap_severity_index' not in self.data.columns or 'final_adoption_propensity' not in self.data.columns:
            ax.text(0.5, 0.5, 'Gap Severity or\nAdoption Data\nNot Available',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Color by gap severity category if available
        if 'gap_severity_category' in self.data.columns:
            severity_colors_map = {
                'low': '#2ca02c',
                'medium': '#ff7f0e',
                'high': '#d62728',
                'critical': '#8b0000'
            }
            
            for cat in ['low', 'medium', 'high', 'critical']:
                mask = self.data['gap_severity_category'] == cat
                if mask.any():
                    ax.scatter(self.data.loc[mask, 'final_adoption_propensity'],
                              self.data.loc[mask, 'gap_severity_index'],
                              c=severity_colors_map[cat],
                              label=cat.capitalize(),
                              alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
        else:
            # Continuous color mapping
            scatter = ax.scatter(self.data['final_adoption_propensity'],
                               self.data['gap_severity_index'],
                               c=self.data['gap_severity_index'],
                               cmap='RdYlGn_r', alpha=0.6, s=50,
                               edgecolors='white', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='Gap Severity Index')
        
        # Highlight critical quadrant (high adoption + high gap)
        median_adoption = self.data['final_adoption_propensity'].median()
        q75_gap = self.data['gap_severity_index'].quantile(0.75)
        
        ax.axvline(median_adoption, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(q75_gap, color='gray', linestyle=':', alpha=0.5)
        
        # Add quadrant label
        ax.text(0.95, 0.95, 'Priority:\nHigh Adoption\n+ High Gap',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, fontweight='bold')
        
        # Add trend line
        x = self.data['final_adoption_propensity']
        y = self.data['gap_severity_index']
        if len(x) > 1 and x.std() > 0:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x.sort_values(), p(x.sort_values()), 
                   "b--", alpha=0.5, linewidth=2, 
                   label=f'Trend (r={np.corrcoef(x, y)[0,1]:.2f})')
        
        ax.set_xlabel('Final Adoption Propensity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gap Severity Index', fontsize=11, fontweight='bold')
        ax.set_title('G) Gap Severity vs Adoption Propensity\n(Investment Priority Targeting)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9, frameon=True, fancybox=True, loc='upper left')
        ax.grid(alpha=0.3)
    
    def _plot_distance_by_category(self, ax: plt.Axes):
        """Plot distance distribution by accessibility category."""
        # Prepare data
        categories = []
        distances_by_cat = []
        colors = []
        
        for cat in self.accessibility_order:
            mask = self.data['charging_accessibility_category'] == cat
            if mask.any():
                categories.append(cat.capitalize())
                distances_by_cat.append(
                    self.data.loc[mask, 'nearest_charger_distance'] / 1000
                )
                colors.append(self.accessibility_colors[cat])
        
        # Create violin plot
        parts = ax.violinplot(distances_by_cat, positions=range(len(categories)),
                             showmeans=True, showmedians=True)
        
        # Color the violin plots
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('white')
            pc.set_linewidth(2)
        
        # Style the other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylabel('Distance to Nearest Charger (km)', fontsize=11, fontweight='bold')
        ax.set_title('H) Distance Distribution by Accessibility Category',
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    def _plot_summary_metrics(self, ax: plt.Axes):
        """Plot summary metrics table."""
        ax.axis('off')
        
        # Compile summary statistics
        metrics = []
        
        # Basic accessibility
        metrics.append(['Median Distance', 
                       f"{self.data['nearest_charger_distance'].median()/1000:.2f} km"])
        metrics.append(['Mean Accessibility',
                       f"{self.data['accessibility_score'].mean():.3f}"])
        metrics.append(['Mean Public Reliance',
                       f"{self.data['public_charging_reliance'].mean():.3f}"])
        
        # Infrastructure factors
        if 'infrastructure_factor' in self.data.columns:
            metrics.append(['Mean Infrastructure Factor',
                           f"{self.data['infrastructure_factor'].mean():.3f}"])
        if 'capacity_factor' in self.data.columns:
            metrics.append(['Mean Capacity Factor',
                           f"{self.data['capacity_factor'].mean():.3f}"])
        
        # Charger density
        metrics.append(['Avg Chargers (1km)',
                       f"{self.data['chargers_within_1km'].mean():.1f}"])
        metrics.append(['Avg Chargers (5km)',
                       f"{self.data['chargers_within_5km'].mean():.1f}"])
        
        # Gap severity
        if 'gap_severity_index' in self.data.columns:
            metrics.append(['Mean Gap Severity',
                           f"{self.data['gap_severity_index'].mean():.3f}"])
            if 'gap_severity_category' in self.data.columns:
                critical_pct = (self.data['gap_severity_category'] == 'critical').sum() / len(self.data) * 100
                metrics.append(['Critical Gaps',
                               f"{critical_pct:.1f}%"])
        
        # Adoption propensity
        if 'final_adoption_propensity' in self.data.columns:
            metrics.append(['Mean Adoption',
                           f"{self.data['final_adoption_propensity'].mean():.3f}"])
        
        # Bottlenecks
        if 'charging_bottleneck' in self.data.columns:
            bottleneck_pct = self.data['charging_bottleneck'].sum() / len(self.data) * 100
            metrics.append(['Bottleneck Areas',
                           f"{bottleneck_pct:.1f}%"])
        
        # Create table
        table = ax.table(cellText=metrics,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', fontsize=11)
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(color='white')
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
        
        ax.set_title('I) Summary Metrics',
                    fontsize=12, fontweight='bold', pad=20)


def visualize_charging_infrastructure(data: gpd.GeoDataFrame,
                                     output_dir: str = "output/charging_visualization") -> Dict[str, Path]:
    """
    Generate all charging infrastructure visualizations.
    
    Parameters
    ----------
    data : gpd.GeoDataFrame
        Charging infrastructure analysis data
    output_dir : str
        Directory to save visualizations
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping visualization names to file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = ChargingInfrastructureVisualizer(data)
    
    # Generate visualizations
    output_files = {}
    
    print("Generating charging infrastructure visualizations...")
    print("-" * 60)
    
    # 1. Comprehensive dashboard
    dashboard_path = output_path / "charging_infrastructure_dashboard.png"
    visualizer.create_comprehensive_dashboard(str(dashboard_path))
    output_files['dashboard'] = dashboard_path
    
    # 2. PCA biplot
    pca_path = output_path / "charging_infrastructure_pca_biplot.png"
    visualizer.create_pca_biplot(str(pca_path))
    output_files['pca_biplot'] = pca_path
    
    # 3. Detailed metrics
    metrics_path = output_path / "charging_infrastructure_detailed_metrics.png"
    visualizer.create_detailed_metrics_summary(str(metrics_path))
    output_files['detailed_metrics'] = metrics_path
    
    print("-" * 60)
    print(f"✅ All visualizations saved to: {output_path}")
    print(f"   Total files: {len(output_files)}")
    
    return output_files

