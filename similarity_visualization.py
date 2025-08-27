#!/usr/bin/env python3
"""
Similarity Visualization Utilities for Media AI Pipeline
Provides methods to visualize and explain similarity scores
"""
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimilarityVisualizer:
    def __init__(self):
        """Initialize the similarity visualizer"""
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        sns.set_palette("husl")
        
        # Define color scheme
        self.colors = {
            'semantic': '#3498db',    # Blue
            'syntactic': '#e74c3c',   # Red
            'hybrid': '#9b59b6',      # Purple
            'similarity': '#2ecc71',  # Green
            'background': '#ecf0f1'   # Light gray
        }
        
        logger.info("Similarity Visualizer initialized")
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray, 
                                labels: List[str] = None, 
                                title: str = "Similarity Matrix",
                                save_path: Optional[str] = None) -> str:
        """Create a heatmap visualization of similarity scores"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                similarity_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                square=True,
                ax=ax,
                xticklabels=labels[:len(similarity_matrix)] if labels else False,
                yticklabels=labels[:len(similarity_matrix)] if labels else False
            )
            
            ax.set_title(f"{title}\n(Higher values = More similar)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Heatmap saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return "heatmap_displayed"
                
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {e}")
            return ""
    
    def create_similarity_distribution(self, similarity_scores: List[float],
                                     similarity_type: str = "Combined",
                                     save_path: Optional[str] = None) -> str:
        """Create a distribution plot of similarity scores"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with KDE
            ax1.hist(similarity_scores, bins=20, alpha=0.7, color=self.colors.get(similarity_type.lower(), 'blue'), 
                    edgecolor='black', density=True)
            
            # Add KDE curve
            from scipy import stats
            kde = stats.gaussian_kde(similarity_scores)
            x_range = np.linspace(min(similarity_scores), max(similarity_scores), 100)
            ax1.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
            
            ax1.set_xlabel('Similarity Score')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{similarity_type} Similarity Score Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            box_plot = ax2.boxplot(similarity_scores, vert=True, patch_artist=True,
                                  boxprops=dict(facecolor=self.colors.get(similarity_type.lower(), 'blue')))
            ax2.set_ylabel('Similarity Score')
            ax2.set_title(f'{similarity_type} Similarity Score Box Plot')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"""
            Mean: {np.mean(similarity_scores):.3f}
            Median: {np.median(similarity_scores):.3f}
            Std Dev: {np.std(similarity_scores):.3f}
            Min: {min(similarity_scores):.3f}
            Max: {max(similarity_scores):.3f}
            """
            
            ax2.text(0.02, 0.98, stats_text.strip(), transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Distribution plot saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return "distribution_displayed"
                
        except Exception as e:
            logger.error(f"Error creating similarity distribution: {e}")
            return ""
    
    def create_comparison_chart(self, results: List[Dict], 
                               query: str = "Search Query",
                               save_path: Optional[str] = None) -> str:
        """Create a comparison chart showing semantic vs syntactic scores"""
        try:
            # Extract data
            items = []
            semantic_scores = []
            syntactic_scores = []
            combined_scores = []
            
            for i, result in enumerate(results[:10]):  # Top 10 results
                filename = result.get('metadata', {}).get('filename', f'Item {i+1}')
                # Truncate long filenames
                if len(filename) > 30:
                    filename = filename[:27] + '...'
                
                items.append(filename)
                semantic_scores.append(result.get('semantic_score', 0))
                syntactic_scores.append(result.get('syntactic_score', 0))
                combined_scores.append(result.get('similarity_score', 0))
            
            if not items:
                logger.warning("No items to visualize")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(items))
            width = 0.25
            
            # Create bars
            bars1 = ax.bar(x - width, semantic_scores, width, label='Semantic', 
                          color=self.colors['semantic'], alpha=0.8)
            bars2 = ax.bar(x, syntactic_scores, width, label='Syntactic', 
                          color=self.colors['syntactic'], alpha=0.8)
            bars3 = ax.bar(x + width, combined_scores, width, label='Combined', 
                          color=self.colors['hybrid'], alpha=0.8)
            
            # Customize chart
            ax.set_xlabel('Media Items')
            ax.set_ylabel('Similarity Score')
            ax.set_title(f'Similarity Scores Comparison\nQuery: "{query}"')
            ax.set_xticks(x)
            ax.set_xticklabels(items, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.01:  # Only label significant values
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            add_value_labels(bars3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Comparison chart saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return "comparison_displayed"
                
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return ""
    
    def create_cluster_visualization(self, clusters: List[List[Dict]], 
                                   save_path: Optional[str] = None) -> str:
        """Create a visualization of semantic clusters"""
        try:
            if not clusters:
                logger.warning("No clusters to visualize")
                return ""
            
            # Prepare data
            cluster_sizes = [len(cluster) for cluster in clusters]
            cluster_labels = [f'Cluster {i+1}\n({size} items)' 
                            for i, size in enumerate(cluster_sizes)]
            
            # Create pie chart for cluster sizes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
            wedges, texts, autotexts = ax1.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
            ax1.set_title('Semantic Cluster Distribution')
            
            # Bar chart with cluster details
            y_pos = np.arange(len(clusters))
            ax2.barh(y_pos, cluster_sizes, color=colors, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f'Cluster {i+1}' for i in range(len(clusters))])
            ax2.set_xlabel('Number of Items')
            ax2.set_title('Items per Cluster')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add cluster details as text
            cluster_info = []
            for i, cluster in enumerate(clusters):
                sample_files = [item.get('metadata', {}).get('filename', 'Unknown') 
                              for item in cluster[:3]]  # First 3 files
                sample_text = ', '.join([f[:20] + '...' if len(f) > 20 else f for f in sample_files])
                if len(cluster) > 3:
                    sample_text += f' (and {len(cluster)-3} more)'
                cluster_info.append(f"Cluster {i+1}: {sample_text}")
            
            info_text = '\n'.join(cluster_info)
            fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Cluster visualization saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return "cluster_displayed"
                
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {e}")
            return ""
    
    def create_similarity_timeline(self, items_with_dates: List[Tuple[str, float, str]], 
                                  save_path: Optional[str] = None) -> str:
        """Create a timeline showing similarity scores over time"""
        try:
            if not items_with_dates:
                logger.warning("No timeline data to visualize")
                return ""
            
            # Sort by date
            sorted_items = sorted(items_with_dates, key=lambda x: x[2])
            
            dates = [item[2] for item in sorted_items]
            scores = [item[1] for item in sorted_items]
            names = [item[0] for item in sorted_items]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create scatter plot
            scatter = ax.scatter(range(len(dates)), scores, 
                               c=scores, cmap='viridis', s=100, alpha=0.7)
            
            # Add trend line
            z = np.polyfit(range(len(dates)), scores, 1)
            p = np.poly1d(z)
            ax.plot(range(len(dates)), p(range(len(dates))), "r--", alpha=0.8, linewidth=2)
            
            # Customize plot
            ax.set_xlabel('Time (Items ordered by creation date)')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Similarity Scores Over Time')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Similarity Score')
            
            # Add annotations for highest scoring items
            top_indices = np.argsort(scores)[-3:]  # Top 3
            for idx in top_indices:
                ax.annotate(names[idx][:20], (idx, scores[idx]),
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Timeline saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return "timeline_displayed"
                
        except Exception as e:
            logger.error(f"Error creating similarity timeline: {e}")
            return ""
    
    def explain_similarity_score(self, score: float, similarity_type: str = "semantic") -> str:
        """Generate human-readable explanation of similarity score"""
        explanations = {
            "semantic": {
                (0.9, 1.0): "Extremely similar in meaning and context",
                (0.8, 0.9): "Highly similar content with strong thematic overlap",
                (0.7, 0.8): "Moderately similar with related concepts",
                (0.6, 0.7): "Somewhat similar with some common themes",
                (0.5, 0.6): "Weakly similar with minimal overlap",
                (0.0, 0.5): "Low similarity with different content"
            },
            "syntactic": {
                (0.9, 1.0): "Extremely similar structure and patterns",
                (0.8, 0.9): "Highly similar code/UI structure",
                (0.7, 0.8): "Moderately similar structural elements",
                (0.6, 0.7): "Somewhat similar patterns",
                (0.5, 0.6): "Weakly similar structure",
                (0.0, 0.5): "Different structural patterns"
            },
            "hybrid": {
                (0.9, 1.0): "Extremely similar in both content and structure",
                (0.8, 0.9): "Highly similar across multiple dimensions",
                (0.7, 0.8): "Moderately similar overall",
                (0.6, 0.7): "Somewhat similar",
                (0.5, 0.6): "Weakly similar",
                (0.0, 0.5): "Low overall similarity"
            }
        }
        
        score_ranges = explanations.get(similarity_type.lower(), explanations["semantic"])
        
        for (low, high), explanation in score_ranges.items():
            if low <= score < high:
                return f"{explanation} (Score: {score:.3f})"
        
        return f"Similarity score: {score:.3f}"
    
    def generate_similarity_report(self, results: List[Dict], query: str,
                                 include_explanations: bool = True) -> str:
        """Generate a comprehensive text report of similarity results"""
        try:
            report_lines = [
                "=" * 60,
                f"SIMILARITY ANALYSIS REPORT",
                "=" * 60,
                f"Query: {query}",
                f"Results Found: {len(results)}",
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "SUMMARY STATISTICS:",
                "-" * 30
            ]
            
            if results:
                # Calculate statistics
                similarity_scores = [r.get('similarity_score', 0) for r in results]
                semantic_scores = [r.get('semantic_score', 0) for r in results if r.get('semantic_score')]
                syntactic_scores = [r.get('syntactic_score', 0) for r in results if r.get('syntactic_score')]
                
                report_lines.extend([
                    f"Average Similarity: {np.mean(similarity_scores):.3f}",
                    f"Highest Similarity: {max(similarity_scores):.3f}",
                    f"Lowest Similarity: {min(similarity_scores):.3f}",
                    f"Standard Deviation: {np.std(similarity_scores):.3f}",
                ])
                
                if semantic_scores:
                    report_lines.append(f"Average Semantic Score: {np.mean(semantic_scores):.3f}")
                if syntactic_scores:
                    report_lines.append(f"Average Syntactic Score: {np.mean(syntactic_scores):.3f}")
                
                report_lines.extend([
                    "",
                    "TOP RESULTS:",
                    "-" * 30
                ])
                
                # Show top 10 results
                for i, result in enumerate(results[:10], 1):
                    filename = result.get('metadata', {}).get('filename', f'Item {i}')
                    similarity_score = result.get('similarity_score', 0)
                    similarity_type = result.get('similarity_type', 'unknown')
                    
                    report_lines.append(f"{i:2}. {filename}")
                    report_lines.append(f"    Similarity Score: {similarity_score:.3f}")
                    report_lines.append(f"    Type: {similarity_type}")
                    
                    if include_explanations:
                        explanation = self.explain_similarity_score(similarity_score, similarity_type)
                        report_lines.append(f"    Explanation: {explanation}")
                    
                    # Add semantic and syntactic breakdown if available
                    if result.get('semantic_score') is not None:
                        report_lines.append(f"    Semantic: {result['semantic_score']:.3f}")
                    if result.get('syntactic_score') is not None:
                        report_lines.append(f"    Syntactic: {result['syntactic_score']:.3f}")
                    
                    # Add content summary if available
                    if 'analysis' in result and 'summary' in result['analysis']:
                        summary = result['analysis']['summary'][:100] + '...' if len(result['analysis']['summary']) > 100 else result['analysis']['summary']
                        report_lines.append(f"    Summary: {summary}")
                    
                    report_lines.append("")
                
                # Distribution analysis
                report_lines.extend([
                    "SCORE DISTRIBUTION:",
                    "-" * 30
                ])
                
                # Create histogram bins
                hist, bin_edges = np.histogram(similarity_scores, bins=5)
                for i in range(len(hist)):
                    bin_range = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                    percentage = (hist[i] / len(similarity_scores)) * 100
                    bar = "█" * int(percentage / 5)  # Visual bar
                    report_lines.append(f"{bin_range}: {hist[i]:2} items ({percentage:5.1f}%) {bar}")
            
            else:
                report_lines.append("No results found.")
            
            report_lines.append("\n" + "=" * 60)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating similarity report: {e}")
            return f"Error generating report: {e}"
    
    def save_similarity_data(self, results: List[Dict], query: str, 
                           filename: str = "similarity_data.json") -> str:
        """Save similarity results to JSON for further analysis"""
        try:
            data = {
                'query': query,
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_results': len(results),
                'results': results
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Similarity data saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving similarity data: {e}")
            return ""

def main():
    """Example usage of similarity visualization"""
    visualizer = SimilarityVisualizer()
    
    # Example similarity matrix
    similarity_matrix = np.array([
        [1.00, 0.85, 0.62, 0.41],
        [0.85, 1.00, 0.73, 0.38],
        [0.62, 0.73, 1.00, 0.29],
        [0.41, 0.38, 0.29, 1.00]
    ])
    
    labels = ["Code_Screenshot_1.png", "Code_Screenshot_2.png", 
              "Terminal_Output.png", "Design_Mockup.png"]
    
    print("🎨 Creating similarity visualizations...")
    
    # Create heatmap
    heatmap_path = visualizer.create_similarity_heatmap(
        similarity_matrix, labels, "Code Similarity Matrix"
    )
    print(f"Heatmap created: {heatmap_path}")
    
    # Example similarity scores
    scores = [0.85, 0.73, 0.68, 0.62, 0.59, 0.55, 0.51, 0.47, 0.43, 0.38]
    dist_path = visualizer.create_similarity_distribution(scores, "Semantic")
    print(f"Distribution plot created: {dist_path}")
    
    # Example explanation
    explanation = visualizer.explain_similarity_score(0.78, "semantic")
    print(f"Score explanation: {explanation}")

if __name__ == "__main__":
    main()
