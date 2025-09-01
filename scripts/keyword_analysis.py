import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from collections import Counter

# Configure style
plt.style.use('seaborn-v0_8-darkgrid')

# Use a modern color palette
COLORS = {
    'Positive_AI_Stance': '#2ecc71',    # Green
    'Negative_AI_Stance': '#e74c3c',    # Red
    'Balanced_Cautionary': '#3498db',   # Blue
    'HighSchool': '#27ae60',            # Darker green
    'CollegeGrad': '#2980b9',          # Darker blue
    'Industry': '#c0392b'              # Darker red
}

# Import the same keyword categories used in policy_analysis.py
KEYWORD_CATEGORIES = {
    'Positive_AI_Stance': [
        # Embracing AI
        'embrace ai', 'leverage ai', 'ai-enabled', 'ai-driven', 'ai tools', 'harness ai',
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

def count_keyword_occurrences(text, phrases):
    """Count individual occurrences of each phrase in the text."""
    if pd.isna(text):
        return Counter()
    text = text.lower()
    counts = Counter()
    for phrase in phrases:
        count = len(re.findall(rf'\b{re.escape(phrase)}\b', text))
        if count > 0:
            counts[phrase] = count
    return counts

def analyze_domain_keywords(df, domain_type):
    """Analyze keyword usage for a specific domain."""
    domain_data = df[df['domain_type'] == domain_type]
    results = {}
    for category, phrases in KEYWORD_CATEGORIES.items():
        category_counts = Counter()
        for text in domain_data['policy_summary']:
            counts = count_keyword_occurrences(text, phrases)
            category_counts.update(counts)
        results[category] = category_counts
    return results

def create_detailed_domain_visualization(domain_results, domain_name, output_path):
    """Create a detailed visualization for a single domain showing all three categories."""
    plt.figure(figsize=(20, 18))  # Increased height for more spacing
    
    # Create gridspec for more control over spacing
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
    
    # Create three subplots, one for each category
    for idx, (category, counts) in enumerate(domain_results.items(), 1):
        # Sort keywords by frequency
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        keywords, frequencies = zip(*sorted_items[:15]) if sorted_items else ([], [])
        
        # Create subplot with gridspec
        plt.subplot(gs[idx-1])
        
        # Create horizontal bars
        bars = plt.barh(range(len(frequencies)), frequencies, 
                       color=COLORS[category], alpha=0.7)
        
        # Use two-line titles to prevent overlap issues
        if idx == 1:
            # For the first category, put everything in the title
            plt.title(f'Keyword Analysis: {domain_name}\n\n{category.replace("_", " ")} Keywords', 
                     pad=30, fontsize=14)
        else:
            # For other categories
            plt.title(f'{category.replace("_", " ")} Keywords', 
                     pad=20, fontsize=14)
        plt.yticks(range(len(keywords)), keywords, fontsize=10)
        
        # Add frequency labels
        for i, v in enumerate(frequencies):
            plt.text(v, i, f' {v}', va='center', fontsize=10)
            
        # Remove spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Add gridlines
        plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove main title as we're using category titles as main titles
    # plt.suptitle removed entirely
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_comparison(all_domain_results, output_path):
    """Create a summary visualization comparing top keywords across domains."""
    plt.figure(figsize=(20, 18))  # Further increased height
    
    # Number of top keywords to show per category
    n_top = 5
    
    # Create a subplot for each category with more vertical spacing
    # Add gridspec for more control over spacing
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.6)
    
    # Create a subplot for each category
    for cat_idx, category in enumerate(KEYWORD_CATEGORIES.keys(), 1):
        plt.subplot(gs[cat_idx-1])
        
        # Get top keywords for each domain
        domain_data = []
        max_freq = 0
        for domain, results in all_domain_results.items():
            if category in results and results[category]:
                top_n = sorted(results[category].items(), key=lambda x: x[1], reverse=True)[:n_top]
                domain_data.append((domain, top_n))
                max_freq = max(max_freq, max(freq for _, freq in top_n))
        
        # Plot data
        y_positions = []
        y_labels = []
        colors = []
        frequencies = []
        
        for domain_idx, (domain, keywords) in enumerate(domain_data):
            for kw_idx, (keyword, freq) in enumerate(keywords):
                y_pos = domain_idx * (n_top + 1) + kw_idx
                y_positions.append(y_pos)
                y_labels.append(f'{keyword} ({freq})')
                colors.append(COLORS[domain])
                frequencies.append(freq)
        
        # Create horizontal bars
        plt.barh(y_positions, frequencies, color=colors, alpha=0.7)
        
        # Put titles directly above each chart instead of using suptitle
        if cat_idx == 1:
            # Add two-line title for first chart
            plt.title(f'Domain Comparison Summary\n\nTop {n_top} {category.replace("_", " ")} Keywords by Domain', 
                     pad=30, fontsize=14)
        else:
            # Regular title for other categories
            plt.title(f'Top {n_top} {category.replace("_", " ")} Keywords by Domain', 
                     pad=20, fontsize=14)
        plt.yticks(y_positions, y_labels, fontsize=10)
        
        # Add domain labels
        domain_label_positions = [(n_top + 1) * i + (n_top - 1) / 2 
                                for i in range(len(domain_data))]
        ax2 = plt.twinx()
        ax2.set_ylim(plt.gca().get_ylim())
        ax2.set_yticks(domain_label_positions)
        ax2.set_yticklabels([d[0] for d in domain_data], fontsize=12, fontweight='bold')
        
        # Remove spines and add grid
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove the overall title as we've moved it to the first chart
    plt.tight_layout()
    
    # We're already using GridSpec for spacing, so no need for additional adjustment
    
    # Save the plot with extra padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def main():
    """Main function to run the keyword analysis."""
    # Read the dataset
    df = pd.read_csv('../data/ai_policy_dataset.csv')
    
    # Create results directory if it doesn't exist
    results_path = '../results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Analyze keywords for each domain
    domains = ['HighSchool', 'CollegeGrad', 'Industry']
    all_domain_results = {}
    
    # Create detailed visualizations for each domain
    for domain in domains:
        print(f"\nAnalyzing {domain} policies...")
        domain_results = analyze_domain_keywords(df, domain)
        all_domain_results[domain] = domain_results
        
        # Create detailed visualization for this domain
        output_path = f'{results_path}/{domain.lower()}_detailed_analysis.png'
        create_detailed_domain_visualization(domain_results, domain, output_path)
        print(f"Created detailed visualization for {domain}")
    
    # Create summary comparison
    create_summary_comparison(all_domain_results, f'{results_path}/domain_comparison_summary.png')
    print("\nCreated summary comparison visualization")
    
    print("\nAnalysis complete! Visualizations saved in the results directory:")
    print("- domain_comparison_summary.png (Overview)")
    for domain in domains:
        print(f"- {domain.lower()}_detailed_analysis.png (Detailed {domain} analysis)")

if __name__ == "__main__":
    main()
