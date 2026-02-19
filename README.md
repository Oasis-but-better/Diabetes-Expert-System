# Hybrid Diabetes Diagnosis Expert System

## Advanced Rule-Based + Case-Based Reasoning System

---

## Overview

This system combines **forward-chaining rule-based reasoning** with **case-based reasoning (CBR)** to create a comprehensive diabetes diagnosis expert system that implements ADA 2025 Standards of Care.

### Key Features

**Dual Reasoning Engines**: FOL rules + CBR experience learning  
**Expanded Rule Set**: 20 ADA-compliant rules (12 original + 8 new)  
**LLM Explanations**: Natural language via Groq/Llama 3.3 70B  
**Confidence Scoring**: Hybrid rule + similarity confidence  
**Auto-Learning**: Novel case retention for continuous improvement  

---

## Quick Start

```bash
# 1. Setup
chmod +x setup.sh && ./setup.sh

# 2. Add API key (optional)
echo "GROQ_API_KEY=your_key_here" >> .env

# 3. Run
python run_hybrid.py
```

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for complete documentation.

---

## Project Structure

```
hybrid_diabetes_system/
├── src/                    # Source code
│   ├── logic_parser.py    # FOL parser
│   ├── logic_solver.py    # Unification & forward chaining
│   ├── diabetes_solver.py # Enhanced solver
│   ├── case_library.py    # CBR case management
│   ├── similarity.py      # Similarity metrics
│   ├── retrieval.py       # K-NN retrieval
│   ├── adaptation.py      # Solution adaptation
│   └── hybrid_reasoner.py # Integration layer (NEW)
│
├── knowledge_bases/
│   └── rules_expanded.kb  # 20 ADA rules (NEW)
│
├── data/
│   └── case_library.json  # CBR case database
│
├── docs/
│   └── USER_GUIDE.md      # Complete documentation
│
├── run_hybrid.py          # Main entry point
├── setup.sh               # Automated setup
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## What's New in This Hybrid System

### 1. Enhanced Rule Set (20 Rules)

**Original (12 rules)**:
- Rules 1-4: Lab value abstraction
- Rules 5-7: Diagnostic criteria
- Rules 8-9: Type 1/Type 2 classification
- Rules 10-12: Emergency detection & treatment

**New ADA 2025 Additions (8 rules)**:
- **Rules 13-14**: Prediabetes detection (IFG & IGT)
- **Rule 15**: Screening guidelines for at-risk populations
- **Rule 16**: LADA (Latent Autoimmune Diabetes) detection
- **Rule 17**: Gestational diabetes history consideration
- **Rule 18**: Metabolic syndrome screening
- **Rule 19**: Cardiovascular risk assessment
- **Rule 20**: Lifestyle intervention prioritization

### 2. Hybrid Architecture

```
Patient Data
     │
     ├──► Rule Engine (FOL) ──┐
     │                         │
     └──► CBR System ──────────┤
                               ▼
                        Solution Fusion
                               │
                               ▼
                        LLM Explanation
                               │
                               ▼
                         Final Diagnosis
```

**Priority Levels**:
1. Emergency rules (DKA, HHS) → Override all
2. Confirmed diagnosis (rules) → Primary
3. CBR insights → Supporting evidence
4. Combined confidence → Fusion score

### 3. Integration Features

- **Safety Overrides**: Emergency rules always take precedence
- **Confidence Fusion**: `rule_confidence + (CBR_similarity × 0.2)`
- **Explanation Synthesis**: Technical + natural language
- **Automatic Learning**: Novel cases (similarity < 0.75) retained

---

## Usage Examples

### Example 1: Emergency Detection

```bash
python run_hybrid.py --patient alex
```

**Result**: DKA emergency detected, 100% confidence, immediate ER referral

### Example 2: Prediabetes with Risk Factors

```bash
python run_hybrid.py --patient taylor
```

**Result**: Prediabetes (IFG + IGT), 82% confidence, lifestyle intervention recommended

### Example 3: Discordant Results

```bash
python run_hybrid.py --patient sam
```

**Result**: Conflicting test results, 78% confidence, repeat FPG required

---

## Test Scenarios

The system includes 5 pre-configured test cases:

| Patient | Scenario | Expected Outcome |
|---------|----------|------------------|
| Alex | Young, DKA emergency | Type 1 + emergency |
| Sam | Discordant FPG/A1C | Repeat testing |
| Jordan | Asymptomatic Type 2 | Confirmed Type 2 |
| Taylor | Prediabetes + family Hx | Intensive lifestyle |
| Morgan | LADA candidate | Endocrine referral |

---

## Documentation

- **[USER_GUIDE.md](docs/USER_GUIDE.md)**: Complete user documentation
- **[rules_expanded.kb](knowledge_bases/rules_expanded.kb)**: Full rule set with comments
- **[case_library.json](data/case_library.json)**: Sample cases

---

## Technical Details

### Dependencies

- Python 3.8+
- LangGraph (CBR workflow)
- LangChain (LLM integration)
- Groq API (optional, for explanations)
- colorama (terminal formatting)

### Architecture Highlights

- **FOL Parser**: Custom first-order logic parser
- **Forward Chaining**: Exhaustive inference with conflict resolution
- **K-NN Retrieval**: Similarity-based case matching (k=3)
- **LLM Integration**: Groq/Llama 3.3 70B for explanations
- **Hybrid Fusion**: Rule priority with CBR enrichment

---

## Educational Value

This project demonstrates:

1. **Knowledge Representation**: First-order logic for medical rules
2. **Rule-Based Systems**: Forward chaining with conflict resolution
3. **Case-Based Reasoning**: 4R cycle (Retrieve-Reuse-Revise-Retain)
4. **Hybrid AI**: Combining symbolic and experience-based reasoning
5. **LLM Integration**: Using LLMs for explanation generation
6. **Domain Modeling**: ADA clinical guidelines in AI

---

## Important Disclaimers

**Educational Use Only**: This system is for learning purposes and is **NOT** for clinical use.

**Not Medical Software**: Do not use for actual patient diagnosis or treatment decisions.

**Consult Professionals**: Always seek qualified medical advice for health concerns.

---

## Contributing

Suggestions for improvement:

1. **Add More Rules**: Extend ADA coverage (gestational diabetes, MODY, etc.)
2. **Enhanced CBR**: Implement case adaptation strategies
3. **Temporal Reasoning**: Track lab trends over time
4. **Uncertainty Handling**: Bayesian confidence intervals
5. **Multi-Modal**: Incorporate images, EHR data

---

## License

MIT License - Educational and research use

---

## Acknowledgments

- **ADA 2025 Standards**: Clinical guidelines
- **Original Systems**: Rule-based and CBR foundations
- **Groq**: Free LLM API access

---

## Support

- Documentation: See `docs/USER_GUIDE.md`
- Issues: Review troubleshooting section

---

**Version**: 1.0.0  
**Date**: February 2026  
