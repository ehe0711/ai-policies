import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Use the same keyword categories as other scripts
KEYWORD_CATEGORIES = {
    'Positive_AI_Stance': [
        # Embracing AI
        'embrace ai', 'leverage ai', 'ai-enabled', 'ai-driven', 'harness ai',
        'artificial intelligence tools', 'power of ai', 'benefits of ai', 'ai solutions',
        'ai capabilities', 'ai features', 'ai potential', 'ai opportunities',
        'transformative', 'revolutionary', 'innovative use', 'cutting-edge',
        
        # Promoting Proficiency
        'ai literacy', 'ai skills', 'ai competency', 'ai proficiency', 'ai expertise',
        'future-ready', 'digital literacy', 'technological proficiency', 'digital fluency',
        'ai-ready', 'tech-savvy', 'digital competence', 'technological literacy',
        'computational thinking', 'digital skills', 'modern skills',
        
        # Professional Development
        'career readiness', 'workforce preparation', 'industry standard', 'upskill',
        'professional skills', 'competitive advantage', 'career development',
        'workforce skills', 'future workforce', 'professional growth', 'skill building',
        'career advancement', 'professional development', 'industry demands',
        
        # Positive Integration
        'integrate ai', 'incorporate ai', 'adopt ai', 'implement ai', 'embrace technology',
        'enhance learning', 'improve efficiency', 'optimize', 'streamline', 'empower',
        'augment', 'supplement', 'support learning', 'facilitate', 'enable growth',
        'accelerate learning', 'personalize learning', 'adaptive learning',
        
        # Growth Mindset
        'opportunity', 'innovation', 'advancement', 'progress', 'growth', 'improvement',
        'enhancement', 'development', 'success', 'achievement', 'excellence',
        'breakthrough', 'pioneering', 'forward-thinking', 'leading edge', 'advancement',
        'potential', 'possibilities', 'promising', 'beneficial'
    ],
    
    'Negative_AI_Stance': [
        # Direct Restrictions
        'prohibited', 'forbidden', 'not allowed', 'banned', 'restricted', 'block',
        'unauthorized use', 'prohibited use', 'ai ban', 'not permitted', 'disallowed',
        'prevent use', 'restrict access', 'limitation', 'controlled use', 'prohibited access',
        'unauthorized access', 'forbidden use', 'restricted access', 'block access',
        
        # Punishments & Consequences
        'academic dishonesty', 'disciplinary action', 'penalty', 'violation', 'misconduct',
        'consequences', 'punishment', 'suspension', 'termination', 'disciplinary measures',
        'sanctions', 'repercussions', 'penalties', 'academic integrity violation',
        'academic offense', 'cheating', 'plagiarism', 'misuse', 'abuse',
        
        # Downplaying & Limitations
        'over-reliance', 'dependency', 'limitations of ai', 'ai limitations',
        'reduce ai use', 'minimize ai use', 'restrict ai', 'overuse', 'overdependence',
        'excessive use', 'limited utility', 'constrained use', 'replacement',
        'substitution', 'artificial', 'superficial', 'shallow learning',
        
        # Risks & Concerns
        'ai threat', 'ai risk', 'dangerous', 'harmful', 'detrimental', 'hazard',
        'negative impact', 'adverse effects', 'compromise', 'danger', 'threat',
        'problem', 'issue', 'concern', 'worry', 'drawback', 'disadvantage',
        'vulnerability', 'exposure', 'susceptibility'
    ],
    
    'Balanced_Cautionary': [
        # Ethical Considerations
        'ethical ai', 'responsible ai', 'ai ethics', 'ethical considerations',
        'ethical guidelines', 'ethical use', 'responsible use', 'moral implications',
        'ethical framework', 'ethical standards', 'ethical principles', 'moral considerations',
        'ethical responsibility', 'ethical awareness', 'ethical judgment', 'moral compass',
        
        # Thoughtful Implementation
        'appropriate use', 'mindful use', 'considered use', 'balanced approach',
        'careful consideration', 'thoughtful integration', 'measured approach',
        'strategic implementation', 'controlled adoption', 'guided use', 'supervised use',
        'monitored use', 'structured approach', 'systematic implementation',
        
        # Human-AI Balance
        'human oversight', 'human judgment', 'human review', 'human-in-the-loop',
        'ai assistance', 'ai support', 'complement human', 'augment human',
        'human-centered', 'human-first', 'human element', 'human factor',
        'human intelligence', 'human wisdom', 'human discretion', 'human guidance',
        
        # Safety & Privacy
        'data privacy', 'information security', 'data protection', 'safeguards',
        'privacy measures', 'security protocols', 'data safety', 'privacy standards',
        'confidentiality', 'data governance', 'information protection',
        
        # Forward-looking Considerations
        'sustainable use', 'long-term implications', 'future impact', 'societal impact',
        'evolving technology', 'emerging technology', 'ongoing assessment',
        'continuous evaluation', 'regular review', 'adaptive policy', 'policy evolution',
        'future considerations', 'long-term effects', 'broader implications'
    ]
}

def count_keyword_phrases(text, phrases):
    """Count occurrences of phrases in text, handling multi-word phrases."""
    if pd.isna(text):
        return 0
    text = text.lower()
    count = 0
    for phrase in phrases:
        count += len(re.findall(rf'\b{re.escape(phrase)}\b', text))
    return count

def classify_policy(row):
    """Classify a policy based on its weighted scores."""
    # Get normalized scores
    scores = {
        'Positive': row['Positive_AI_Stance_normalized'],
        'Negative': row['Negative_AI_Stance_normalized'],
        'Cautionary': row['Balanced_Cautionary_normalized']
    }
    
    # Get the dominant category
    max_category = max(scores.items(), key=lambda x: x[1])
    
    # If the maximum score is 0, classify as "Neutral"
    if max_category[1] == 0:
        return "Neutral"
    
    # Calculate the ratio between the max score and the average of others
    other_scores = [v for k, v in scores.items() if k != max_category[0]]
    avg_others = sum(other_scores) / len(other_scores) if other_scores else 0
    
    # If the maximum score is significantly higher than others (50% higher),
    # classify as that category
    if avg_others == 0 or (max_category[1] / avg_others) > 1.5:
        return max_category[0]
    else:
        return "Mixed"

def analyze_policies():
    """Analyze and classify policies."""
    # Read the dataset
    df = pd.read_csv('../data/ai_policy_dataset.csv')
    
    # Calculate keyword counts
    for category, phrases in KEYWORD_CATEGORIES.items():
        df[category] = df['policy_summary'].apply(
            lambda x: count_keyword_phrases(x, phrases)
        )
    
    # Calculate normalized scores (per 100 words)
    df['word_count'] = df['policy_summary'].str.split().str.len()
    for category in KEYWORD_CATEGORIES.keys():
        df[f'{category}_normalized'] = df[category] / df['word_count'] * 100
    
    # Classify each policy
    df['Classification'] = df.apply(classify_policy, axis=1)
    
    return df

def plot_classification_breakdown(df):
    """Create visualizations for policy classification breakdown."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1.2])
    
    # Colors for classifications
    colors = {
        'Positive': '#2ecc71',    # Green
        'Negative': '#e74c3c',    # Red
        'Cautionary': '#3498db',  # Blue
        'Mixed': '#95a5a6',       # Gray
        'Neutral': '#f1c40f'      # Yellow
    }
    
    # Plot 1: Overall distribution
    overall_counts = df['Classification'].value_counts()
    bars = ax1.bar(overall_counts.index, overall_counts.values,
                   color=[colors[cat] for cat in overall_counts.index])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    ax1.set_title('Overall Distribution of Policy Classifications',
                  pad=20, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Policies')
    
    # Plot 2: Distribution by domain
    domain_counts = pd.crosstab(df['domain_type'], df['Classification'])
    domain_counts_pct = domain_counts.div(domain_counts.sum(axis=1), axis=0) * 100
    
    domain_counts_pct.plot(kind='bar', stacked=True, ax=ax2,
                          color=[colors[cat] for cat in domain_counts_pct.columns])
    
    ax2.set_title('Policy Classification Distribution by Domain',
                  pad=20, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Domain Type')
    ax2.set_ylabel('Percentage of Policies')
    
    # Rotate x-labels
    ax2.tick_params(axis='x', rotation=45)
    
    # Add legend
    ax2.legend(title='Classification', bbox_to_anchor=(1.05, 1),
               loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = '../results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    plt.savefig(f'{results_path}/policy_classification_breakdown.png',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def print_classification_summary(df):
    """Print detailed summary of policy classifications."""
    print("\nPolicy Classification Summary")
    print("=" * 50)
    
    # Overall statistics
    print("\nOverall Distribution:")
    overall_counts = df['Classification'].value_counts()
    for category, count in overall_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{category}: {count} policies ({percentage:.1f}%)")
    
    # Distribution by domain
    print("\nDistribution by Domain:")
    domain_counts = pd.crosstab(df['domain_type'], df['Classification'])
    domain_percentages = domain_counts.div(domain_counts.sum(axis=1), axis=0) * 100
    
    for domain in domain_counts.index:
        print(f"\n{domain}:")
        for category in domain_counts.columns:
            count = domain_counts.loc[domain, category]
            percentage = domain_percentages.loc[domain, category]
            print(f"{category}: {count} policies ({percentage:.1f}%)")

def main():
    """Main function to run the analysis."""
    print("Loading and analyzing policies...")
    df = analyze_policies()
    
    print("\nGenerating visualizations...")
    plot_classification_breakdown(df)
    
    print("\nGenerating classification summary...")
    print_classification_summary(df)
    
    print("\nAnalysis complete! Visualizations saved in ../results/policy_classification_breakdown.png")

if __name__ == "__main__":
    main()
