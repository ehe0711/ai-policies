import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Configure style
plt.style.use('default')
sns.set_palette("husl")

# Define focused keyword categories
KEYWORD_CATEGORIES = {
    'Positive_AI_Stance': [
        # Embracing AI
        'embrace ai', 'leverage ai', 'ai-enabled', 'ai-driven', 'harness ai',
        'power of ai', 'benefits of ai', 'ai solutions',
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
        # Count exact phrase matches
        count += len(re.findall(rf'\b{re.escape(phrase)}\b', text))
    return count

def analyze_policies(df):
    """Analyze policies for each category."""
    # Calculate counts for each category
    for category, phrases in KEYWORD_CATEGORIES.items():
        df[category] = df['policy_summary'].apply(lambda x: count_keyword_phrases(x, phrases))
    return df

def set_style():
    """Set professional plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Modern color palette
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    sns.set_palette(colors)
    
    # Typography
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def ensure_results_directory():
    """Create results directory if it doesn't exist"""
    import os
    results_path = '../results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path

def create_boxplot(df, results_path, categories, friendly_names, domain_names):
    """Create boxplot showing distribution of stances"""
    plt.figure(figsize=(15, 8))
    df_melted = df.melt(id_vars=['domain_type'],
                        value_vars=categories,
                        var_name='Category',
                        value_name='Count')
    
    # Replace category names with friendly names
    df_melted['Category'] = df_melted['Category'].map(friendly_names)
    df_melted['domain_type'] = df_melted['domain_type'].map(domain_names)
    
    sns.boxplot(x='domain_type', y='Count', hue='Category', data=df_melted)
    plt.title('Distribution of Policy Stances by Domain', pad=20, fontweight='bold')
    plt.xlabel('Domain Type', labelpad=10)
    plt.ylabel('Number of References', labelpad=10)
    plt.legend(title='Policy Stance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{results_path}/stance_distribution_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_violin_plot(df, results_path, categories, friendly_names, domain_names):
    """Create violin plot showing stance distributions"""
    plt.figure(figsize=(15, 8))
    df_melted = df.melt(id_vars=['domain_type'],
                        value_vars=categories,
                        var_name='Category',
                        value_name='Count')
    
    df_melted['Category'] = df_melted['Category'].map(friendly_names)
    df_melted['domain_type'] = df_melted['domain_type'].map(domain_names)
    
    sns.violinplot(x='domain_type', y='Count', hue='Category', data=df_melted, split=True)
    plt.title('Detailed Distribution of Policy Stances', pad=20, fontweight='bold')
    plt.xlabel('Domain Type', labelpad=10)
    plt.ylabel('Number of References', labelpad=10)
    plt.legend(title='Policy Stance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{results_path}/stance_distribution_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap(df, results_path, categories, friendly_names, domain_names):
    """Create correlation heatmap between stances"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[categories].corr()
    
    # Use friendly names for the heatmap
    correlation_matrix.index = [friendly_names[i] for i in correlation_matrix.index]
    correlation_matrix.columns = [friendly_names[i] for i in correlation_matrix.columns]
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', center=0,
                fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Between Policy Stances', pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_path}/stance_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_time_trend_proxy(df, results_path, categories, friendly_names, domain_names):
    """Create a visualization showing cumulative stance adoption"""
    plt.figure(figsize=(15, 8))
    
    for category in categories:
        # Sort values to create a cumulative trend
        values = df[category].sort_values(ascending=True)
        cumulative = np.arange(len(values)) / len(values)
        plt.plot(values, cumulative, label=friendly_names[category], linewidth=2)
    
    plt.title('Cumulative Distribution of Policy Stances', pad=20, fontweight='bold')
    plt.xlabel('Number of References', labelpad=10)
    plt.ylabel('Proportion of Policies', labelpad=10)
    plt.legend(title='Policy Stance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_path}/stance_cumulative_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(df):
    """Create comprehensive professional visualizations comparing domains."""
    set_style()
    results_path = ensure_results_directory()
    categories = ['Positive_AI_Stance', 'Negative_AI_Stance', 'Balanced_Cautionary']
    friendly_names = {'Positive_AI_Stance': 'Positive', 
                     'Negative_AI_Stance': 'Negative', 
                     'Balanced_Cautionary': 'Cautionary'}
    # Map domain types to display names
    domain_names = {
        'HighSchool': 'High School',
        'CollegeGrad': 'College/University',
        'Industry': 'Industry/Workplace'
    }
    
    # 1. Enhanced Bar Plot
    plt.figure(figsize=(12, 7))
    means = df.groupby('domain_type')[categories].mean()
    ax = means.plot(kind='bar', width=0.8)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.title('AI Policy Stance Across Educational and Professional Domains', 
              pad=20, fontweight='bold')
    plt.xlabel('Domain Type', labelpad=10)
    plt.ylabel('Average Number of References', labelpad=10)
    
    # Customize legend
    plt.legend(title='Policy Stance', 
              labels=[friendly_names[cat] for cat in categories],
              title_fontsize=11,
              bbox_to_anchor=(1.02, 1),
              loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{results_path}/policy_stance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced Stance Balance
    plt.figure(figsize=(10, 7))
    ratios = means['Positive_AI_Stance'] / means['Negative_AI_Stance']
    
    # Create gradient colors based on ratio values
    colors = ['#2ecc71' if x > 1 else '#e74c3c' for x in ratios]
    bars = plt.bar(range(len(ratios)), ratios, color=colors)
    
    # Add horizontal line at ratio=1
    plt.axhline(y=1, color='#7f8c8d', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Balance of Positive vs. Negative AI Stance by Domain',
              pad=20, fontweight='bold')
    plt.xlabel('Domain Type', labelpad=10)
    plt.ylabel('Ratio (Positive to Negative References)', labelpad=10)
    
    plt.xticks(range(len(ratios)), [domain_names[idx] for idx in ratios.index])
    plt.tight_layout()
    plt.savefig(f'{results_path}/stance_balance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced Proportions
    plt.figure(figsize=(12, 7))
    proportions = means.div(means.sum(axis=1), axis=0) * 100
    
    ax = proportions.plot(kind='bar', stacked=True)
    
    # Add percentage labels
    for c in ax.containers:
        # Add labels
        labels = [f'{v:.1f}%' if v >= 5 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center')
    
    plt.title('Distribution of Policy Stances Across Domains',
              pad=20, fontweight='bold')
    plt.xlabel('Domain Type', labelpad=10)
    plt.ylabel('Percentage of Total References', labelpad=10)
    
    plt.legend(title='Policy Stance',
              labels=[friendly_names[cat] for cat in categories],
              title_fontsize=11,
              bbox_to_anchor=(1.02, 1),
              loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{results_path}/stance_proportions.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Enhanced Radar Chart
    fig = plt.figure(figsize=(12, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    # Use custom colors for each domain
    domain_colors = {'HighSchool': '#2ecc71', 
                    'CollegeGrad': '#3498db', 
                    'Industry': '#e74c3c'}
    
    for domain in df['domain_type'].unique():
        values = means.loc[domain].values.flatten()
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=3, 
                label=domain_names[domain], 
                color=domain_colors[domain])
        ax.fill(angles, values, alpha=0.25, color=domain_colors[domain])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([friendly_names[cat] for cat in categories])
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.title('AI Policy Profiles Across Domains',
              pad=20, fontweight='bold', y=1.05)
    
    # Customize legend
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.1),
              ncol=3, columnspacing=1)
    
    plt.tight_layout()
    plt.savefig(f'{results_path}/policy_profiles.png',
                dpi=300, bbox_inches='tight')

def main():
    # Create results directory
    import os
    os.makedirs('../results', exist_ok=True)
    
    # Read and analyze data
    df = pd.read_csv('../data/ai_policy_dataset.csv')
    df = analyze_policies(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("Analysis complete! Check the 'results' folder for visualizations:")
    print("1. policy_stance_comparison.png - Compare policy stances across domains")
    print("2. stance_balance.png - Ratio of positive to negative stances")
    print("3. stance_proportions.png - Relative proportions of different stances")
    print("4. policy_profiles.png - Radar chart showing domain policy profiles")

if __name__ == "__main__":
    main()
