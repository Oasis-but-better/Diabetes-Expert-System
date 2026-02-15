#!/usr/bin/env python3
"""
Hybrid Diabetes Diagnosis System - Main Runner

This script demonstrates the hybrid expert system combining rule-based
and case-based reasoning for diabetes diagnosis.

Usage:
    python run_hybrid.py [--patient PATIENT_ID] [--verbose] [--rules-only]
    
Examples:
    python run_hybrid.py                    # Run all test scenarios
    python run_hybrid.py --patient alex     # Run specific patient
    python run_hybrid.py --verbose          # Show detailed trace
    python run_hybrid.py --rules-only       # Disable CBR

Author: B552 Knowledge-Based AI
Date: February 2026
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from hybrid_reasoner import HybridDiabetesSystem, HybridDiagnosisResult
from colorama import init, Fore, Back, Style
init(autoreset=True)


# Test scenarios from original assignments
TEST_SCENARIOS = {
    "alex": {
        "patient_id": "Alex",
        "age": 14,
        "bmi": 19.0,
        "fpg": None,
        "a1c": None,
        "random_glucose": 350,
        "fpg_repeat": None,
        "ketones": "positive",
        "symptoms": ["polydipsia", "weight-loss"],
        "description": "Young patient with DKA - Emergency scenario"
    },
    "sam": {
        "patient_id": "Sam",
        "age": 45,
        "bmi": 27.0,
        "fpg": 130,
        "a1c": 6.0,
        "random_glucose": None,
        "fpg_repeat": None,
        "ketones": "negative",
        "symptoms": [],
        "description": "Discordant results - Requires repeat testing"
    },
    "jordan": {
        "patient_id": "Jordan",
        "age": 55,
        "bmi": 32.0,
        "fpg": 150,
        "a1c": None,
        "random_glucose": None,
        "fpg_repeat": 148,
        "ketones": "negative",
        "symptoms": ["tired"],
        "description": "Type 2 diabetes - Confirmed by repeat FPG"
    },
    "taylor": {
        "patient_id": "Taylor",
        "age": 42,
        "bmi": 28.5,
        "fpg": 115,
        "a1c": 6.1,
        "random_glucose": None,
        "fpg_repeat": None,
        "ketones": "negative",
        "symptoms": [],
        "family_history": True,
        "description": "Prediabetes - High risk for progression"
    },
    "morgan": {
        "patient_id": "Morgan",
        "age": 35,
        "bmi": 22.0,
        "fpg": 135,
        "a1c": 6.8,
        "random_glucose": None,
        "fpg_repeat": None,
        "ketones": "negative",
        "symptoms": [],
        "gradual_onset": True,
        "description": "LADA candidate - Adult with low BMI"
    }
}


def print_banner():
    """Print system banner"""
    print("\n" + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "HYBRID DIABETES DIAGNOSIS EXPERT SYSTEM".center(80))
    print(Fore.CYAN + "Rule-Based Reasoning + Case-Based Reasoning + LLM Explanations".center(80))
    print("="*80 + "\n")
    
    # System status
    llm_status = "ENABLED ✓" if os.getenv("GROQ_API_KEY") else "DISABLED ✗ (set GROQ_API_KEY to enable)"
    print(f"LLM Explanations: {Fore.GREEN if 'ENABLED' in llm_status else Fore.YELLOW}{llm_status}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_patient_info(patient_data: Dict[str, Any]):
    """Print patient information in formatted table"""
    print(Fore.YELLOW + Style.BRIGHT + f"\n{'='*80}")
    print(Fore.YELLOW + Style.BRIGHT + f"PATIENT: {patient_data['patient_id']}")
    print(Fore.YELLOW + Style.BRIGHT + f"{'='*80}\n")
    
    if "description" in patient_data:
        print(Fore.CYAN + f"Scenario: {patient_data['description']}\n")
    
    print(Fore.WHITE + "Patient Data:")
    print(f"  Age:             {patient_data.get('age', 'N/A')} years")
    print(f"  BMI:             {patient_data.get('bmi', 'N/A')} kg/m²")
    print(f"  FPG:             {patient_data.get('fpg', 'N/A')} mg/dL")
    print(f"  FPG (repeat):    {patient_data.get('fpg_repeat', 'N/A')} mg/dL")
    print(f"  A1C:             {patient_data.get('a1c', 'N/A')}%")
    print(f"  Random Glucose:  {patient_data.get('random_glucose', 'N/A')} mg/dL")
    print(f"  Ketones:         {patient_data.get('ketones', 'N/A')}")
    
    symptoms = patient_data.get('symptoms', [])
    print(f"  Symptoms:        {', '.join(symptoms) if symptoms else 'None reported'}")
    
    # Risk factors
    risk_factors = []
    if patient_data.get('family_history'):
        risk_factors.append("Family history")
    if patient_data.get('hypertension'):
        risk_factors.append("Hypertension")
    if patient_data.get('gestational_history'):
        risk_factors.append("Prior GDM")
    
    if risk_factors:
        print(f"  Risk Factors:    {', '.join(risk_factors)}")
    print()


def print_result(result: HybridDiagnosisResult, verbose: bool = False):
    """Print diagnosis result in formatted output"""
    
    # Emergency alert
    if result.is_emergency:
        print(Fore.RED + Back.WHITE + Style.BRIGHT + "\n" + " " * 80)
        print(Fore.RED + Back.WHITE + Style.BRIGHT + "🚨 MEDICAL EMERGENCY DETECTED 🚨".center(80))
        print(Fore.RED + Back.WHITE + Style.BRIGHT + " " * 80 + "\n")
    
    # Primary diagnosis
    print(Fore.GREEN + Style.BRIGHT + "PRIMARY DIAGNOSIS:")
    print(Fore.WHITE + f"  {result.primary_diagnosis}")
    print(Fore.GREEN + f"  Confidence: {result.confidence_score:.1%}\n")
    
    # Rule-based analysis
    print(Fore.CYAN + Style.BRIGHT + "RULE-BASED ANALYSIS:")
    if result.rule_diagnosis:
        print(Fore.WHITE + f"  Diagnosis: {result.rule_diagnosis}")
    if result.rule_classification:
        print(Fore.WHITE + f"  Classification: {result.rule_classification}")
    
    if verbose and result.rule_derived_facts:
        print(Fore.CYAN + "\n  Derived Facts:")
        for fact in result.rule_derived_facts[:10]:  # Show first 10
            print(Fore.WHITE + f"    ✓ {fact}")
    print()
    
    # Case-based analysis
    if result.similar_cases:
        print(Fore.MAGENTA + Style.BRIGHT + "SIMILAR CASES (CBR):")
        for i, case_info in enumerate(result.similar_cases[:3], 1):
            case = case_info['case']
            sim = case_info['similarity']
            print(Fore.WHITE + f"  {i}. {case.id} (similarity: {sim:.2f})")
            print(Fore.WHITE + f"     → {case.solution.diagnosis}")
        print()
    
    # Explanation
    if result.llm_explanation:
        print(Fore.YELLOW + Style.BRIGHT + "CLINICAL EXPLANATION:")
        # Wrap text to 76 characters
        explanation = result.llm_explanation
        words = explanation.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 <= 78:
                line += word + " "
            else:
                print(Fore.WHITE + line)
                line = "  " + word + " "
        if line.strip():
            print(Fore.WHITE + line)
        print()
    
    # Recommendations
    if result.final_recommendations:
        print(Fore.GREEN + Style.BRIGHT + "RECOMMENDATIONS:")
        for rec in result.final_recommendations:
            # Format recommendation names nicely
            rec_formatted = rec.replace("_", " ").replace("-", " ").title()
            print(Fore.WHITE + f"  • {rec_formatted}")
        print()
    
    # Case learning
    if result.case_retained:
        print(Fore.BLUE + "✓ Novel case retained in library for future learning\n")
    
    print(Fore.YELLOW + "="*80 + "\n")


def save_report(result: HybridDiagnosisResult, output_dir: Path):
    """Save diagnosis report to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnosis_{result.patient_id}_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    print(Fore.GREEN + f"Report saved: {filepath}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hybrid Diabetes Diagnosis Expert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        Run all test scenarios
  %(prog)s --patient alex         Run only Alex scenario
  %(prog)s --verbose              Show detailed reasoning trace
  %(prog)s --rules-only           Disable CBR (rules only)
  %(prog)s --save-reports         Save diagnosis reports to files
        """
    )
    
    parser.add_argument(
        "--patient",
        choices=list(TEST_SCENARIOS.keys()),
        help="Run specific patient scenario"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed reasoning trace"
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Disable CBR (use rules only)"
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save diagnosis reports to output directory"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Set up paths
    base_dir = Path(__file__).parent
    rules_path = base_dir / "knowledge_bases" / "rules_expanded.kb"
    cases_path = base_dir / "data" / "case_library.json"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize system
    try:
        system = HybridDiabetesSystem(
            rules_kb_path=str(rules_path),
            case_library_path=str(cases_path),
            use_llm=not args.rules_only
        )
        print(Fore.GREEN + f"✓ System initialized successfully")
        print(Fore.GREEN + f"  Rules KB: {rules_path}")
        print(Fore.GREEN + f"  Case Library: {cases_path}")
        print(Fore.GREEN + f"  Cases loaded: {len(system.case_library)}\n")
    except Exception as e:
        print(Fore.RED + f"✗ System initialization failed: {e}")
        return 1
    
    # Select scenarios to run
    if args.patient:
        scenarios_to_run = {args.patient: TEST_SCENARIOS[args.patient]}
    else:
        scenarios_to_run = TEST_SCENARIOS
    
    # Run diagnoses
    results = []
    for patient_id, patient_data in scenarios_to_run.items():
        try:
            # Print patient info
            print_patient_info(patient_data)
            
            # Run diagnosis
            result = system.diagnose(patient_data)
            
            # Print result
            print_result(result, verbose=args.verbose)
            
            # Save report if requested
            if args.save_reports:
                save_report(result, output_dir)
            
            results.append(result)
            
        except Exception as e:
            print(Fore.RED + f"✗ Error diagnosing {patient_id}: {e}\n")
    
    # Summary
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "SUMMARY".center(80))
    print(Fore.CYAN + Style.BRIGHT + "="*80 + "\n")
    
    print(f"Patients Analyzed: {len(results)}")
    print(f"Emergencies Detected: {sum(1 for r in results if r.is_emergency)}")
    print(f"Average Confidence: {sum(r.confidence_score for r in results) / len(results):.1%}")
    print(f"Cases Retained: {sum(1 for r in results if r.case_retained)}")
    print(f"\nFinal Library Size: {len(system.case_library)} cases")
    
    print(Fore.CYAN + "\n" + "="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
