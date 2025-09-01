# AI Policy Gap Analysis

This project analyzes how **AI policies are framed across high schools, colleges, and industry**.  
By applying a lightweight lexicon-based NLP approach, the project classifies policies into **positive**, **negative**, and **cautionary** stances. The goal is to reveal whether schools are preparing students for the AI-enabled workplaces they will eventually enter.

---

## 📊 Project Overview
- **Dataset**: ~150 policies (20 high school, 50 college, 80 industry) collected from handbooks, syllabi, and corporate AI frameworks.
- **Method**: Summarized policies using GPT-5-mini and classified language using lexicon dictionaries.
- **Outputs**: Domain-level and cross-domain visualizations showing how AI is framed (opportunity, risk, or responsibility).

---

## 📁 Repository Structure

```text
├── keywords
    └── balanced_stance.png
    └── positive_stance.png
    └── negative_stance.png
├── data
│   └── ai_policy_dataset.csv      # Collected and summarized policy dataset
│
├── scripts
│   ├── policy_analysis.py         # Core lexicon-based classification
│   ├── classification_breakdown.py # Keyword frequency breakdowns
│   ├── education_comparison.py    # Focused comparisons between HS and College
│   └── keyword_analysis.py        # Keyword-level analysis and support functions
│
├── results
│   ├── highschool_detailed_analysis.png
│   ├── collegegrad_detailed_analysis.png
│   ├── industry_detailed_analysis.png
│   ├── domain_comparison_summary.png
│   ├── policy_stance_comparison.png
│   ├── stance_balance.png
│   ├── stance_proportions.png
│   └── policy_profiles.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### requirements.txt
```text
pandas
numpy
matplotlib
seaborn
```

---

## 🚀 Usage

Run the main analysis from the `scripts/` directory:

```bash
python policy_analysis.py
```

This will:
- Load the dataset (`data/ai_policy_dataset.csv`)
- Run stance classification using keyword dictionaries
- Save all results and plots into the `results/` directory

You can also run specific scripts:
- `classification_breakdown.py` → Creates keyword breakdown visualizations per domain  
- `education_comparison.py` → Focused comparison of high school vs. college policies  
- `keyword_analysis.py` → Helper functions for keyword-level analysis  

---

## 📂 Dataset

The dataset (`ai_policy_dataset.csv`) contains:
- **domain_type** → HighSchool, CollegeGrad, or Industry  
- **policy_summary** → Short standardized summaries of each policy  

Published dataset can be found at https://huggingface.co/datasets/ehe07/ai-policies. 

---

## 📈 Results & Visualizations

Generated outputs include:
- **Detailed domain analysis**: Top keywords per stance within high school, college, and industry policies.  
- **Cross-domain comparison**: Side-by-side bar charts of stance frequency.  
- **Stance balance**: Ratio of positive to negative mentions per domain.  
- **Stance proportions**: Distribution of stance categories as percentages.  
- **Policy profiles (radar chart)**: Domain “profiles” showing how balanced (or imbalanced) their stance is.

## Keywords Used

To enhance reproducibility, illustrations of all keywords by sentiment category are provided in the **keywords** folder. 

---

## 🔍 Key Insights

- **High schools** → Mostly negative framing, centered on plagiarism and prohibition.  
- **Colleges** → More balanced, mixing cautionary oversight with negative enforcement.  
- **Industry** → Largely positive, with cautionary notes about responsibility and ethics.  

This highlights a **preparedness gap**: students are taught to see AI as a risk, while employers expect fluency and responsible adoption.

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 🤝 Contributions

Contributions are welcome! Open an issue or submit a pull request for improvements.  
For major changes, please discuss them first in an issue.

---

## 🔗 Citation

If you use this repo, please cite:

```text
Author: Ethan He
Title: AI Policy Gap Analysis
Year: 2025
URL: https://github.com/ehe0711/ai-policies
```
