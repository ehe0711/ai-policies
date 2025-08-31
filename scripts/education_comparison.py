import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Configure style
plt.style.use('seaborn-v0_8-darkgrid')

# Define focused keyword categories
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

def analyze_education_policies():
    """Analyze and compare high school vs college AI policies with detailed categorization."""
    # Read the dataset
    df = pd.read_csv('../data/ai_policy_dataset.csv')
    
    # Filter for educational institutions
    edu_df = df[df['domain_type'].isin(['HighSchool', 'CollegeGrad'])]
    
    # Calculate keyword counts for each category
    for category, phrases in KEYWORD_CATEGORIES.items():
        edu_df[category] = edu_df['policy_summary'].apply(
            lambda x: count_keyword_phrases(x, phrases)
        )
    
    # Calculate normalized scores (per 100 words) for fair comparison
    edu_df['word_count'] = edu_df['policy_summary'].str.split().str.len()
    for category in KEYWORD_CATEGORIES.keys():
        edu_df[f'{category}_normalized'] = edu_df[category] / edu_df['word_count'] * 100
    
    # Aggregate results by domain type
    normalized_columns = [f'{cat}_normalized' for cat in KEYWORD_CATEGORIES.keys()]
    results = edu_df.groupby('domain_type')[normalized_columns].mean()
    
    return results, edu_df

def plot_comparison(results):
    """Create visualizations comparing high school and college policies."""
    plt.style.use('seaborn-v0_8-darkgrid')  # Set style for each plot
    plt.figure(figsize=(15, 8))
    
    # Set background color
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('#ffffff')
    
    # Create grouped bar plot
    x = np.arange(len(results.index))
    width = 0.25  # Width of the bars
    
    # Define colors for each category
    colors = {
        'Positive_AI_Stance': '#2ecc71',    # Green
        'Negative_AI_Stance': '#e74c3c',    # Red
        'Balanced_Cautionary': '#3498db'     # Blue
    }
    
    # Plot bars for each category
    for i, (category, color) in enumerate(colors.items()):
        normalized_col = f'{category}_normalized'
        position = x + (i - 1) * width
        bars = plt.bar(position, results[normalized_col], width,
                      label=category.replace('_', ' '),
                      color=color, alpha=0.8)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=10)
    
    # Customize the plot
    plt.xlabel('Education Level', fontsize=12, labelpad=10)
    plt.ylabel('Normalized Frequency (per 100 words)', fontsize=12, labelpad=10)
    plt.title('Comparison of AI Policy Approaches: High School vs College',
              pad=20, fontsize=14, fontweight='bold')
    
    # Adjust x-axis
    plt.xticks(x, results.index, fontsize=11)
    
    # Add legend with custom position
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=10, frameon=True)
    
    # Add grid with custom style
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure results directory exists and save the plot
    results_path = '../results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_path = f'{results_path}/education_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved plot to {save_path}")
    plt.close()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = '../results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_path = f'{results_path}/education_comparison_detailed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved plot to {save_path}")
    plt.close()

def print_statistical_analysis(edu_df):
    """Print detailed statistical analysis of the differences."""
    # Prepare data for each education level
    hs_data = edu_df[edu_df['domain_type'] == 'HighSchool']
    college_data = edu_df[edu_df['domain_type'] == 'CollegeGrad']
    
    print("\nDetailed Statistical Analysis:")
    print("-" * 50)
    
    for category in KEYWORD_CATEGORIES.keys():
        normalized_category = f'{category}_normalized'
        print(f"\n{category.replace('_', ' ')} Analysis:")
        print("-" * 50)
        
        # Calculate statistics
        hs_mean = hs_data[normalized_category].mean()
        college_mean = college_data[normalized_category].mean()
        hs_median = hs_data[normalized_category].median()
        college_median = college_data[normalized_category].median()
        
        print(f"High School Average: {hs_mean:.2f}")
        print(f"College Average: {college_mean:.2f}")
        print(f"High School Median: {hs_median:.2f}")
        print(f"College Median: {college_median:.2f}")
        
        # Calculate and print difference
        diff = college_mean - hs_mean
        print(f"Absolute Difference (College - High School): {diff:.2f}")
        
        # Calculate and print ratio where possible
        if hs_mean != 0:
            ratio = college_mean / hs_mean
            print(f"Ratio (College / High School): {ratio:.2f}")
            
        # Print predominance
        if abs(diff) < 0.1:  # Threshold for considering them similar
            print("Finding: Similar emphasis in both levels")
        else:
            predominant = "College" if diff > 0 else "High School"
            print(f"Finding: More emphasized in {predominant} policies")

def main():
    """Main function to run the enhanced education comparison analysis."""
    # Create results directory if it doesn't exist
    results_path = '../results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Run analysis
    print("Loading and analyzing data...")
    results, edu_df = analyze_education_policies()
    
    print("\nCreating visualization...")
    plot_comparison(results)
    
    print("\nGenerating statistical analysis...")
    print_statistical_analysis(edu_df)
    
    print(f"\nAnalysis complete! Visualization saved in {results_path}/education_comparison.png")

if __name__ == "__main__":
    main()
