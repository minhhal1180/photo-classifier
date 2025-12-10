import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict

# Set style
sns.set_style("whitegrid")
plt.switch_backend('agg')

def save_label_distribution(
    label_counts: pd.Series,
    output_path: Path,
    category_map: Optional[Dict[str, str]] = None,
    figsize: tuple = (10, 6),
    dpi: int = 160
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Map labels
    if category_map:
        display_labels = [category_map.get(label, label) for label in label_counts.index]
    else:
        display_labels = label_counts.index
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(display_labels))
    
    bars = ax.bar(positions, label_counts.values, color='#2a9d8f', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Phân bố dữ liệu theo danh mục', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Số lượng ảnh', fontsize=12)
    ax.set_xlabel('Danh mục', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(display_labels, rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def save_feature_importance(
    feature_series: pd.Series,
    output_path: Path,
    top_k: int = 15,
    figsize: tuple = (10, 8),
    dpi: int = 160
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get top features
    top_features = feature_series.sort_values(ascending=False).head(top_k)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1], edgecolor='black')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_title(f'Top {top_k} Features quan trọng nhất', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: Path,
    category_map: Optional[Dict[str, str]] = None,
    figsize: tuple = (10, 8),
    dpi: int = 160
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Map labels
    if category_map:
        display_labels = [category_map.get(label, label) for label in labels]
    else:
        display_labels = labels
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize for color map
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    
    # Ticks and labels
    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=30, ha='right')
    ax.set_yticklabels(display_labels)
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dự đoán', fontsize=12)
    ax.set_ylabel('Thực tế', fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(j, i, f"{value}",
                   ha='center', va='center',
                   color='white' if value > thresh else 'black',
                   fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def save_training_summary(
    output_dir: Path,
    label_counts: pd.Series,
    feature_importance: pd.Series,
    confusion_matrix: np.ndarray,
    labels: List[str],
    category_map: Optional[Dict[str, str]] = None,
    dpi: int = 160
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Label distribution
    paths['label_distribution'] = save_label_distribution(
        label_counts,
        output_dir / 'label_distribution.png',
        category_map=category_map,
        dpi=dpi
    )
    
    # Feature importance
    paths['feature_importance'] = save_feature_importance(
        feature_importance,
        output_dir / 'feature_importance.png',
        dpi=dpi
    )
    
    # Confusion matrix
    paths['confusion_matrix'] = save_confusion_matrix(
        confusion_matrix,
        labels,
        output_dir / 'confusion_matrix.png',
        category_map=category_map,
        dpi=dpi
    )
    
    return paths
