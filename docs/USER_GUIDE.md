# Hybrid Diabetes Diagnosis System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [System Components](#system-components)
5. [Usage Examples](#usage-examples)
6. [Understanding Results](#understanding-results)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Introduction

The Hybrid Diabetes Diagnosis System is an advanced expert system that combines:

- **Rule-Based Reasoning**: Forward-chaining inference with ADA 2025 standards
- **Case-Based Reasoning**: Experience-based learning from similar patients
- **LLM Explanations**: Natural language explanations via Groq/Llama 3.3

### When to Use This System

✓ Educational demonstrations of hybrid AI reasoning  
✓ Understanding diabetes diagnostic criteria  
✓ Learning about expert system architectures  
✓ Research on knowledge-based AI  

⚠️ **NOT for clinical use** - This is an educational tool, not medical software

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Optional: Groq API key for LLM explanations (free at console.groq.com)

### Step-by-Step Installation

#### Option 1: Automated Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Edit .env and add your GROQ_API_KEY (optional)
nano .env
```

#### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add API key
nano .env  # Or use your preferred editor
```

### Verifying Installation

```bash
# Run system test
python run_hybrid.py --patient alex

# Expected output: Diagnosis report for Alex
```

---

## Quick Start

### Running All Test Scenarios

```bash
python run_hybrid.py
```

This runs all 5 pre-configured patient scenarios and displays:
- Patient data
- Rule-based analysis
- Similar cases from CBR
- Final diagnosis with confidence
- Clinical explanations
- Recommendations

### Running a Specific Patient

```bash
python run_hybrid.py --patient jordan
```

Available patients: `alex`, `sam`, `jordan`, `taylor`, `morgan`

### Verbose Mode (Detailed Trace)

```bash
python run_hybrid.py --verbose
```

Shows:
- All derived facts from rules
- Complete reasoning chain
- Similarity scores for all cases
- Detailed explanations

### Saving Reports

```bash
python run_hybrid.py --save-reports
```

Saves JSON reports to `output/` directory.

---

## System Components

### 1. Rule-Based Engine (`src/logic_parser.py`, `logic_solver.py`, `diabetes_solver.py`)

**Purpose**: Systematic application of ADA diagnostic criteria

**Features**:
- First-order logic representation
- Forward chaining inference
- Conflict resolution strategies
- Explanation tracking

**Rules** (20 total):
- Rules 1-4: Abstraction (lab values → clinical concepts)
- Rules 5-7: Diagnostic criteria
- Rules 8-9: Type 1/Type 2 classification
- Rules 10-12: Emergency detection & treatment
- Rules 13-20: Enhanced ADA 2025 compliance (NEW)

### 2. Case-Based Reasoner (`src/case_library.py`, `similarity.py`, `retrieval.py`)

**Purpose**: Learn from similar historical cases

**CBR Cycle**:
1. **Retrieve**: Find top-3 similar cases via k-NN
2. **Reuse**: Adapt solution from best match
3. **Revise**: Validate with rule engine
4. **Retain**: Store novel cases (similarity < 0.75)

**Similarity Metrics**:
- Numeric features: Scaled absolute difference
- Categorical features: Exact match (1.0) or mismatch (0.0)
- Global similarity: Weighted average

### 3. Hybrid Integration Layer (`src/hybrid_reasoner.py`)

**Purpose**: Fuse rule-based and CBR results

**Integration Strategy**:
```
Priority Levels:
1. Emergency rules (DKA, HHS) → Override everything
2. Confirmed diabetes (rule-based) → Primary diagnosis
3. CBR insights → Supporting evidence
4. Combined confidence → Rule certainty + CBR similarity
```

**Confidence Scoring**:
- Emergency: 1.0 (definitive)
- Confirmed diagnosis: 0.8 + (0.2 × CBR_similarity)
- Provisional: 0.6 + (0.2 × CBR_similarity)
- CBR only: CBR_similarity

### 4. LLM Explanation Generator

**Purpose**: Translate technical output to natural language

**Model**: Llama 3.3 70B Versatile (via Groq)

**Explanation Template**:
```
"Based on your [test results], you have [diagnosis]. 
Your case is similar to [similar_case], who [outcome]. 
We recommend [recommendations]."
```

---

## Usage Examples

### Example 1: Emergency Detection

```bash
python run_hybrid.py --patient alex
```

**Input**:
- Age: 14, BMI: 19
- Random glucose: 350 mg/dL
- Ketones: Positive
- Symptoms: Polydipsia, weight loss

**Output**:
```
🚨 MEDICAL EMERGENCY: Diabetic Ketoacidosis (DKA)

PRIMARY DIAGNOSIS: Type 1 Diabetes Mellitus with DKA
Confidence: 100%

RULE-BASED ANALYSIS:
  ✓ VeryHighGlucose (350 > 250)
  ✓ KetoneStatus(Patient, Positive)
  ✓ MedicalEmergencyDKA
  ✓ SuspectedType1 (age: 14, BMI: 19)

RECOMMENDATIONS:
  • Emergency Room (IMMEDIATE)
  • Insulin Therapy
  • Autoantibody Panel
  • Diabetes Education

EXPLANATION:
This is a medical emergency requiring immediate attention. The 
combination of very high blood sugar (350 mg/dL) and positive 
ketones indicates diabetic ketoacidosis (DKA). Based on the 
young age and lean build, this is likely Type 1 diabetes...
```

### Example 2: Prediabetes Detection

```bash
python run_hybrid.py --patient taylor
```

**Input**:
- Age: 42, BMI: 28.5
- FPG: 115 mg/dL (impaired)
- A1C: 6.1% (impaired)
- Family history: Yes

**Output**:
```
PRIMARY DIAGNOSIS: Prediabetes (IFG + IGT)
Confidence: 82%

RULE-BASED ANALYSIS:
  ✓ FpgImpaired (100-125 range)
  ✓ A1cImpaired (5.7-6.4 range)
  ✓ PrediabetesIFG
  ✓ PrediabetesIGT
  ✓ HighRiskScreening (family history)

SIMILAR CASES:
  1. case_004 (similarity: 0.91)
     → Prediabetes, enrolled in DPP

RECOMMENDATIONS:
  • Intensive Lifestyle Intervention
  • DPP Program (Diabetes Prevention Program)
  • Nutrition Counseling
  • Annual Screening

EXPLANATION:
You have prediabetes, indicated by both your fasting glucose 
(115 mg/dL) and A1C (6.1%). With lifestyle changes, many people 
prevent or delay Type 2 diabetes. Your case is very similar to 
Patient_004, who successfully reversed prediabetes through a 
structured lifestyle program...
```

### Example 3: Discordant Results

```bash
python run_hybrid.py --patient sam
```

**Input**:
- Age: 45, BMI: 27
- FPG: 130 mg/dL (high)
- A1C: 6.0% (normal-ish)
- No symptoms

**Output**:
```
PRIMARY DIAGNOSIS: Discordant Results - Repeat Testing Required
Confidence: 78%

RULE-BASED ANALYSIS:
  ✓ FpgHigh (≥126)
  ✓ A1cNormal (<6.5)
  ✓ DiscordantResults
  ✓ Recommend(Patient, RepeatFPG)

RECOMMENDATIONS:
  • Repeat FPG (confirm diagnosis)
  • A1C Monitoring
  • Lifestyle Changes

EXPLANATION:
Your test results show a discrepancy: your fasting glucose is 
elevated (130 mg/dL, above the diabetes threshold of 126), but 
your A1C is in the prediabetes range (6.0%). ADA guidelines 
require repeat testing to confirm diabetes when asymptomatic...
```

---

## Understanding Results

### Result Structure

Each diagnosis includes:

1. **Primary Diagnosis**: Final integrated conclusion
2. **Confidence Score**: 0-100% (rule + CBR fusion)
3. **Rule-Based Analysis**: What the rules derived
4. **Similar Cases**: Top 3 historical matches
5. **Explanation**: Natural language reasoning
6. **Recommendations**: Next clinical steps

### Confidence Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 90-100% | Definitive | Trust diagnosis |
| 70-89% | High confidence | Generally reliable |
| 50-69% | Moderate | Consider additional tests |
| <50% | Low | Insufficient data |

### Emergency Indicators

🚨 **Red Alert**: Immediate action required
- DKA (glucose >250 + ketones)
- HHS (glucose >600 in elderly)

⚠️ **Yellow Warning**: Prompt follow-up needed
- Discordant results
- Prediabetes with risk factors

ℹ️ **Blue Info**: Routine monitoring
- Normal results
- Follow-up recommendations

---

## Customization

### Adding New Rules

Edit `knowledge_bases/rules_expanded.kb`:

```prolog
% Your new rule
MyCondition(p) && MyOtherCondition(p) => MyConclusion(p)
MyConclusion(p) => Recommend(p, MyAction)
```

Example - Add obesity screening:

```prolog
% Rule 21: Obesity Screening
BMI(p, b) && b >= 30 && Age(p, a) && a >= 18 => ObesityScreening(p)
ObesityScreening(p) => Recommend(p, NutritionalCounseling)
ObesityScreening(p) => Recommend(p, ExercisePrescription)
```

### Adding Cases to Library

Edit `data/case_library.json`:

```json
{
  "id": "case_008",
  "features": {
    "age": 50,
    "bmi": 30.0,
    "fpg": 140,
    "a1c": 7.5,
    "random_glucose": null,
    "fpg_repeat": null,
    "ketones": "negative",
    "symptoms": []
  },
  "solution": {
    "diagnosis": "Type 2 Diabetes Mellitus",
    "status": "Confirmed",
    "recommendations": [
      "lifestyle-changes",
      "metformin",
      "smbg"
    ]
  },
  "outcome": "successful",
  "notes": "Your notes here"
}
```

### Adjusting Similarity Threshold

Edit `.env`:

```bash
SIMILARITY_THRESHOLD=0.70  # Lower = retain more cases
```

### Changing LLM Model

Edit `src/hybrid_reasoner.py`:

```python
# Replace Groq with OpenAI
from langchain_openai import ChatOpenAI
self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)

# Or use Anthropic
from langchain_anthropic import ChatAnthropic
self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```

---

## Troubleshooting

### Common Issues

#### 1. "GROQ_API_KEY not found"

**Solution**: Either:
- Add API key to `.env` file
- Run without LLM: `python run_hybrid.py --rules-only`

#### 2. "ModuleNotFoundError: No module named 'langgraph'"

**Solution**:
```bash
pip install -r requirements.txt
```

#### 3. "FileNotFoundError: rules_expanded.kb"

**Solution**: Ensure you're running from project root:
```bash
cd hybrid_diabetes_system
python run_hybrid.py
```

#### 4. "No cases in library"

**Solution**: Check `data/case_library.json` exists and is valid JSON

#### 5. Low confidence scores

**Possible causes**:
- Insufficient patient data (missing FPG, A1C)
- No similar cases in library
- Ambiguous presentation

**Solution**: Add more detailed patient data

### Getting Help

- Check documentation: `docs/`
- Run tests: `pytest tests/`
- Enable verbose: `--verbose` flag
- Review logs in `output/`

---

## API Reference

### HybridDiabetesSystem Class

```python
from src.hybrid_reasoner import HybridDiabetesSystem

system = HybridDiabetesSystem(
    rules_kb_path="knowledge_bases/rules_expanded.kb",
    case_library_path="data/case_library.json",
    use_llm=True,
    similarity_threshold=0.75
)

result = system.diagnose(patient_data)
```

### Patient Data Format

```python
patient_data = {
    # Required
    "patient_id": str,
    "age": int,
    "bmi": float,
    
    # Lab values (at least one required)
    "fpg": Optional[float],  # mg/dL
    "a1c": Optional[float],  # %
    "random_glucose": Optional[float],  # mg/dL
    
    # Optional
    "fpg_repeat": Optional[float],
    "a1c_repeat": Optional[float],
    "ketones": str,  # "positive" or "negative"
    "symptoms": List[str],
    "family_history": bool,
    "gestational_history": bool,
    "hypertension": bool,
    "dyslipidemia": bool,
    "smoking_history": bool
}
```

### Result Object

```python
result.patient_id              # str
result.primary_diagnosis       # str
result.confidence_score        # float (0-1)
result.is_emergency           # bool
result.rule_diagnosis         # str
result.cbr_diagnosis          # str
result.similar_cases          # List[Dict]
result.final_recommendations  # List[str]
result.llm_explanation        # str
result.case_retained          # bool

# Export to dict
result_dict = result.to_dict()
```

---

## Advanced Topics

### Extending the Rule Engine

See `src/logic_solver.py` for:
- Custom unification strategies
- Alternative conflict resolution
- Backward chaining implementation

### Custom Similarity Metrics

See `src/similarity.py`:

```python
def custom_similarity(f1: CaseFeatures, f2: CaseFeatures) -> float:
    """Your custom similarity function"""
    # Weight age more heavily
    age_sim = 1.0 - abs(f1.age - f2.age) / 100
    # ... other features
    return weighted_average
```

### Batch Processing

```python
patients = [patient1, patient2, patient3]
results = [system.diagnose(p) for p in patients]

# Save all reports
for result in results:
    save_report(result, output_dir)
```

---

## License and Disclaimer

### Educational Use Only

This system is for educational and research purposes only. It is **not intended for clinical use** and should **not be used to make medical decisions**.

### Medical Disclaimer

Always consult qualified healthcare professionals for medical advice. This system:
- May contain errors or limitations
- Does not replace clinical judgment
- Is not validated for patient care
- Should not delay appropriate medical treatment

### License

MIT License - See LICENSE file for details

---

## Support

For questions or issues:
- Email: [course instructor]
- GitHub: [repository link]
- Documentation: `docs/`

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Course**: B552 - Knowledge-Based AI  
**Institution**: Indiana University, Bloomington
