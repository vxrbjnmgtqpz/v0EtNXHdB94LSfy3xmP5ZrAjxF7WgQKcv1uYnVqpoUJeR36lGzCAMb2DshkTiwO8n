#!/usr/bin/env python3
"""
PNBTR Composite Scoring System
Combines multiple reconstruction metrics into a single accuracy score.
90%+ composite accuracy required for mastery.
"""

import numpy as np
from pathlib import Path
import yaml

# Default metric weights (can be overridden by config)
DEFAULT_WEIGHTS = {
    "SDR": 0.35,           # Signal-to-Distortion Ratio (most important)
    "DeltaFFT": 0.25,      # Spectral fidelity
    "EnvelopeDev": 0.20,   # Dynamic envelope preservation
    "PhaseSkew": 0.10,     # Temporal alignment
    "DynamicRange": 0.05,  # Dynamic range preservation
    "FrequencyResponse": 0.05  # Frequency response retention
}

# Quality tiers for contextual scoring
QUALITY_TIERS = {
    "excellent": 0.95,  # 95%+ - Nearly perfect
    "very_good": 0.90,  # 90-95% - Mastery threshold
    "good": 0.80,       # 80-90% - Acceptable for most uses
    "fair": 0.70,       # 70-80% - Noticeable but usable
    "poor": 0.50,       # 50-70% - Significant degradation
    "very_poor": 0.0    # <50% - Unacceptable
}

def score_accuracy(metrics, weights=None, mode="composite"):
    """
    Calculate composite accuracy score from individual metrics.
    
    Args:
        metrics: Dict of metric scores (each 0.0-1.0)
        weights: Optional weight overrides
        mode: "composite", "conservative", "optimistic"
        
    Returns:
        float: Composite accuracy score [0.0, 1.0]
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Handle missing metrics gracefully
    available_metrics = {}
    total_weight = 0.0
    
    for metric_name, score in metrics.items():
        if metric_name in weights and not np.isnan(score):
            available_metrics[metric_name] = score
            total_weight += weights[metric_name]
    
    if total_weight == 0:
        return 0.0  # No valid metrics
    
    # Calculate weighted average
    weighted_sum = 0.0
    for metric_name, score in available_metrics.items():
        # Ensure score is in valid range
        score = np.clip(score, 0.0, 1.0)
        weight = weights[metric_name]
        weighted_sum += score * weight
    
    # Normalize by available weight
    composite_score = weighted_sum / total_weight
    
    # Apply scoring mode adjustments
    if mode == "conservative":
        # Penalize any metric that's significantly below average
        min_score = min(available_metrics.values())
        if min_score < 0.7:
            penalty = (0.7 - min_score) * 0.2  # Up to 20% penalty
            composite_score = max(0.0, composite_score - penalty)
    
    elif mode == "optimistic":
        # Boost score if most metrics are good
        good_metrics = sum(1 for score in available_metrics.values() if score >= 0.8)
        total_metrics = len(available_metrics)
        if good_metrics / total_metrics >= 0.75:  # 75% of metrics good
            boost = 0.05  # 5% boost
            composite_score = min(1.0, composite_score + boost)
    
    return float(np.clip(composite_score, 0.0, 1.0))

def evaluate_quality_tier(composite_score):
    """
    Determine quality tier based on composite score.
    
    Returns:
        str: Quality tier name
    """
    if composite_score >= QUALITY_TIERS["excellent"]:
        return "excellent"
    elif composite_score >= QUALITY_TIERS["very_good"]:
        return "very_good"
    elif composite_score >= QUALITY_TIERS["good"]:
        return "good"
    elif composite_score >= QUALITY_TIERS["fair"]:
        return "fair"
    elif composite_score >= QUALITY_TIERS["poor"]:
        return "poor"
    else:
        return "very_poor"

def meets_mastery_threshold(composite_score, threshold=0.90):
    """
    Check if score meets mastery threshold.
    
    Args:
        composite_score: Composite accuracy score
        threshold: Minimum required accuracy (default 90%)
        
    Returns:
        bool: True if mastery achieved
    """
    return composite_score >= threshold

def analyze_metric_performance(metrics, weights=None):
    """
    Detailed analysis of individual metric performance.
    Helps identify which aspects need improvement.
    
    Returns:
        dict: Performance analysis with strengths/weaknesses
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    analysis = {
        "strongest_metrics": [],
        "weakest_metrics": [],
        "critical_issues": [],
        "improvement_suggestions": []
    }
    
    # Sort metrics by performance
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    
    # Identify strongest and weakest
    if len(sorted_metrics) > 0:
        # Top performers (>90%)
        analysis["strongest_metrics"] = [
            {"metric": name, "score": score, "grade": "A"}
            for name, score in sorted_metrics 
            if score >= 0.90
        ]
        
        # Weak performers (<70%)
        analysis["weakest_metrics"] = [
            {"metric": name, "score": score, "grade": get_letter_grade(score)}
            for name, score in sorted_metrics 
            if score < 0.70
        ]
    
    # Critical issues (heavily weighted metrics that are poor)
    for metric_name, score in metrics.items():
        weight = weights.get(metric_name, 0)
        if weight >= 0.20 and score < 0.80:  # Important metric performing poorly
            analysis["critical_issues"].append({
                "metric": metric_name,
                "score": score,
                "weight": weight,
                "impact": "HIGH"
            })
    
    # Generate improvement suggestions
    for metric_name, score in metrics.items():
        if score < 0.90:
            suggestion = generate_improvement_suggestion(metric_name, score)
            if suggestion:
                analysis["improvement_suggestions"].append(suggestion)
    
    return analysis

def get_letter_grade(score):
    """Convert numeric score to letter grade"""
    if score >= 0.97:
        return "A+"
    elif score >= 0.93:
        return "A"
    elif score >= 0.90:
        return "A-"
    elif score >= 0.87:
        return "B+"
    elif score >= 0.83:
        return "B"
    elif score >= 0.80:
        return "B-"
    elif score >= 0.77:
        return "C+"
    elif score >= 0.73:
        return "C"
    elif score >= 0.70:
        return "C-"
    elif score >= 0.60:
        return "D"
    else:
        return "F"

def generate_improvement_suggestion(metric_name, score):
    """Generate specific improvement suggestions based on metric performance"""
    suggestions = {
        "SDR": {
            0.8: "Consider noise reduction or better signal alignment",
            0.6: "Significant signal distortion - check model prediction accuracy",
            0.4: "Severe reconstruction errors - model may need retraining",
            0.0: "Complete signal failure - check input/model compatibility"
        },
        "DeltaFFT": {
            0.8: "Minor spectral coloration - adjust frequency response training",
            0.6: "Noticeable frequency response changes - check FFT windowing",
            0.4: "Major spectral distortion - model may be filtering incorrectly",
            0.0: "Spectral content unrecognizable - fundamental model issue"
        },
        "EnvelopeDev": {
            0.8: "Slight dynamic compression - improve envelope tracking",
            0.6: "Dynamic range loss - check amplitude prediction accuracy",
            0.4: "Severe dynamic flattening - model needs envelope training",
            0.0: "Complete dynamic loss - envelope processing broken"
        },
        "PhaseSkew": {
            0.8: "Minor timing offset - improve signal alignment",
            0.6: "Noticeable delay - check temporal prediction accuracy",
            0.4: "Significant timing errors - model latency too high",
            0.0: "Complete timing failure - phase alignment broken"
        },
        "DynamicRange": {
            0.8: "Slight range compression - improve quiet signal handling",
            0.6: "Dynamic range loss - check bit depth preservation",
            0.4: "Major range reduction - model compressing signal",
            0.0: "No dynamic range - signal severely flattened"
        },
        "FrequencyResponse": {
            0.8: "Minor frequency coloration - adjust EQ modeling",
            0.6: "Frequency response deviations - check filter design",
            0.4: "Significant frequency changes - model altering spectrum",
            0.0: "Frequency response unrecognizable - major spectral issues"
        }
    }
    
    if metric_name not in suggestions:
        return None
    
    metric_suggestions = suggestions[metric_name]
    
    # Find appropriate suggestion based on score
    for threshold in sorted(metric_suggestions.keys(), reverse=True):
        if score <= threshold:
            return {
                "metric": metric_name,
                "score": score,
                "suggestion": metric_suggestions[threshold],
                "priority": "HIGH" if score < 0.5 else "MEDIUM" if score < 0.8 else "LOW"
            }
    
    return None

def load_scoring_config(config_path=None):
    """
    Load scoring configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: ../config/thresholds.yaml)
        
    Returns:
        dict: Scoring configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "thresholds.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}, using defaults")
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "mastery_threshold": 0.90,
            "quality_tiers": QUALITY_TIERS.copy()
        }
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "mastery_threshold": 0.90,
            "quality_tiers": QUALITY_TIERS.copy()
        }

def generate_score_report(metrics, weights=None, detailed=True):
    """
    Generate comprehensive scoring report.
    
    Args:
        metrics: Dict of metric scores
        weights: Optional weight overrides
        detailed: Include detailed analysis
        
    Returns:
        dict: Complete scoring report
    """
    composite_score = score_accuracy(metrics, weights)
    quality_tier = evaluate_quality_tier(composite_score)
    mastery = meets_mastery_threshold(composite_score)
    
    report = {
        "composite_score": composite_score,
        "composite_percentage": composite_score * 100,
        "quality_tier": quality_tier,
        "mastery_achieved": mastery,
        "individual_metrics": {}
    }
    
    # Individual metric details
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    
    for metric_name, score in metrics.items():
        report["individual_metrics"][metric_name] = {
            "score": score,
            "percentage": score * 100,
            "weight": weights.get(metric_name, 0.0),
            "grade": get_letter_grade(score),
            "contribution": score * weights.get(metric_name, 0.0)
        }
    
    # Detailed analysis if requested
    if detailed:
        report["analysis"] = analyze_metric_performance(metrics, weights)
    
    return report

def print_score_summary(metrics, weights=None):
    """Print formatted scoring summary to console"""
    report = generate_score_report(metrics, weights, detailed=True)
    
    print(f"\n🎯 PNBTR Reconstruction Score Report")
    print("=" * 50)
    
    # Composite score
    score_pct = report["composite_percentage"]
    tier = report["quality_tier"]
    mastery_icon = "✅" if report["mastery_achieved"] else "❌"
    
    print(f"{mastery_icon} Composite Score: {score_pct:.1f}% ({tier.upper()})")
    
    if report["mastery_achieved"]:
        print("🏆 MASTERY THRESHOLD ACHIEVED!")
    else:
        needed = 90 - score_pct
        print(f"📈 Need +{needed:.1f}% for mastery")
    
    print("\n📊 Individual Metrics:")
    print("-" * 50)
    
    # Sort by contribution (score * weight)
    metrics_sorted = sorted(
        report["individual_metrics"].items(),
        key=lambda x: x[1]["contribution"],
        reverse=True
    )
    
    for metric_name, details in metrics_sorted:
        score_pct = details["percentage"]
        grade = details["grade"]
        weight = details["weight"]
        
        # Status icon
        if score_pct >= 90:
            icon = "✅"
        elif score_pct >= 70:
            icon = "⚠️"
        else:
            icon = "❌"
        
        print(f"{icon} {metric_name:<18}: {score_pct:5.1f}% ({grade}) [weight: {weight:.2f}]")
    
    # Improvement suggestions
    if "analysis" in report and report["analysis"]["improvement_suggestions"]:
        print(f"\n💡 Improvement Suggestions:")
        print("-" * 50)
        
        for suggestion in report["analysis"]["improvement_suggestions"]:
            priority = suggestion["priority"]
            metric = suggestion["metric"]
            text = suggestion["suggestion"]
            
            priority_icon = "🔥" if priority == "HIGH" else "⚠️" if priority == "MEDIUM" else "💡"
            print(f"{priority_icon} {metric}: {text}")
    
    print("=" * 50)

# Threshold management for training

class ThresholdManager:
    """Manages dynamic threshold adjustment during training"""
    
    def __init__(self, initial_threshold=0.90):
        self.threshold = initial_threshold
        self.history = []
        self.adjustment_rate = 0.01  # 1% per adjustment
    
    def update_threshold(self, recent_scores, success_rate):
        """
        Dynamically adjust threshold based on training performance.
        
        Args:
            recent_scores: List of recent composite scores
            success_rate: Fraction of recent attempts that succeeded
        """
        if len(recent_scores) < 10:
            return  # Need more data
        
        avg_score = np.mean(recent_scores)
        
        # If success rate is too high (>80%), increase threshold
        if success_rate > 0.8 and avg_score > self.threshold + 0.05:
            old_threshold = self.threshold
            self.threshold = min(0.99, self.threshold + self.adjustment_rate)
            print(f"📈 Threshold raised: {old_threshold:.3f} → {self.threshold:.3f}")
        
        # If success rate is too low (<20%), decrease threshold
        elif success_rate < 0.2 and avg_score < self.threshold - 0.05:
            old_threshold = self.threshold
            self.threshold = max(0.70, self.threshold - self.adjustment_rate)
            print(f"📉 Threshold lowered: {old_threshold:.3f} → {self.threshold:.3f}")
        
        self.history.append({
            "threshold": self.threshold,
            "avg_score": avg_score,
            "success_rate": success_rate
        })

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Scoring System Test")
    
    # Test metrics
    test_metrics = {
        "SDR": 0.92,
        "DeltaFFT": 0.88,
        "EnvelopeDev": 0.85,
        "PhaseSkew": 0.95,
        "DynamicRange": 0.90,
        "FrequencyResponse": 0.87
    }
    
    # Generate and print report
    print_score_summary(test_metrics)
    
    # Test mastery check
    composite = score_accuracy(test_metrics)
    mastery = meets_mastery_threshold(composite)
    print(f"\n🎯 Composite Score: {composite:.3f} ({composite*100:.1f}%)")
    print(f"🏆 Mastery Achieved: {mastery}")
    
    print("✅ Scoring system test complete") 