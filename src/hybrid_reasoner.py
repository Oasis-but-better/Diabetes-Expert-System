"""
Hybrid Diabetes Diagnosis System - Integration Layer

This module implements the core hybrid reasoning architecture that combines:
1. Rule-based forward chaining (FOL)
2. Case-based reasoning (CBR)
3. LLM-powered explanation generation

The hybrid system leverages the strengths of both approaches:
- Rules provide systematic, explainable baseline reasoning
- CBR offers experience-based insights from similar cases
- LLM generates natural language explanations

"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re

# Import rule-based components
from logic_parser import parse_kb, _parse_predicate
from diabetes_solver import forward_chaining_with_explanation, NoConflictResolution

# Import CBR components
from case_library import CaseLibrary, Case, CaseFeatures, CaseSolution
from retrieval import retrieve_top_k
from similarity import global_similarity


@dataclass
class HybridDiagnosisResult:
    """
    Complete diagnosis result from hybrid system
    
    Combines rule-based and CBR outputs into unified result
    """
    patient_id: str
    
    # Rule-based results - ADD DEFAULTS
    rule_derived_facts: List[Any] = field(default_factory=list)
    rule_explanations: Dict[str, Any] = field(default_factory=dict)
    rule_diagnosis: Optional[str] = None
    rule_classification: Optional[str] = None
    rule_recommendations: List[str] = field(default_factory=list)
    
    # CBR results
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    cbr_diagnosis: Optional[str] = None
    cbr_adapted_solution: Optional[Dict[str, Any]] = None
    
    # Hybrid synthesis
    primary_diagnosis: Optional[str] = None
    confidence_score: float = 0.0
    is_emergency: bool = False
    final_recommendations: List[str] = field(default_factory=list)
    
    # Explanation
    llm_explanation: Optional[str] = None
    reasoning_chain: List[str] = field(default_factory=list)
    
    # Learning
    case_retained: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "patient_id": self.patient_id,
            "primary_diagnosis": self.primary_diagnosis,
            "confidence_score": self.confidence_score,
            "is_emergency": self.is_emergency,
            "rule_diagnosis": self.rule_diagnosis,
            "rule_classification": self.rule_classification,
            "cbr_diagnosis": self.cbr_diagnosis,
            "similar_cases": self.similar_cases,
            "final_recommendations": self.final_recommendations,
            "llm_explanation": self.llm_explanation,
            "case_retained": self.case_retained
        }


class HybridDiabetesSystem:
    """
    Hybrid Expert System combining Rule-Based and Case-Based Reasoning
    
    Architecture:
    1. Parse patient data into FOL facts
    2. Run forward chaining with rule engine
    3. Simultaneously perform CBR retrieval
    4. Fuse rule-based and CBR conclusions
    5. Generate LLM explanation
    6. Optionally retain novel cases
    """
    
    def __init__(
        self,
        rules_kb_path: str,
        case_library_path: str,
        use_llm: bool = True,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize hybrid system
        
        Args:
            rules_kb_path: Path to expanded rules knowledge base
            case_library_path: Path to case library JSON
            use_llm: Whether to use LLM for explanation generation
            similarity_threshold: Threshold for case retention (lower = more novel)
        """
        self.rules_kb_path = rules_kb_path
        self.case_library_path = case_library_path
        self.use_llm = use_llm and bool(os.getenv("GROQ_API_KEY"))
        self.similarity_threshold = similarity_threshold
        
        # Load knowledge bases
        self.rules, self.initial_facts = self._load_rules()

        if self.initial_facts is None:
            self.initial_facts = []
        if self.rules is None:
            self.rules = []

        # Filter out any None rules or rules with None antecedents
        self.rules = [r for r in self.rules if r is not None and 
                    hasattr(r, 'antecedents') and r.antecedents is not None]
        
        self.case_library = CaseLibrary(case_library_path)
        
        # Initialize LLM if enabled
        self.llm = None
        if self.use_llm:
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
            except Exception as e:
                print(f"⚠️  LLM initialization failed: {e}")
                self.use_llm = False
    
    def _load_rules(self) -> Tuple[List[Any], List[Any]]:
        """Load rules from knowledge base file"""
        if os.path.exists(self.rules_kb_path):
            facts, rules = parse_kb(self.rules_kb_path)
            if facts is None:
                facts = []
            if rules is None:
                rules = []
            return rules, facts
        return [], []
    
    def diagnose(self, patient_data: Dict[str, Any]) -> HybridDiagnosisResult:
        """
        Perform hybrid diagnosis on patient data
        
        Args:
            patient_data: Dictionary containing patient features
            
        Returns:
            HybridDiagnosisResult with complete analysis
        """
        patient_id = patient_data.get("patient_id", "Unknown")
        
        # Step 1: Convert patient data to FOL facts
        patient_facts = self._patient_to_facts(patient_data)
        
        # Step 2: Run rule-based inference
        rule_result = self._run_rule_engine(patient_facts)
        
        # Step 3: Run CBR retrieval and adaptation
        cbr_result = self._run_cbr(patient_data)
        
        # Step 4: Fuse results
        hybrid_result = self._fuse_results(
            patient_id=patient_id,
            patient_data=patient_data,
            rule_result=rule_result,
            cbr_result=cbr_result
        )
        
        # Step 5: Generate explanation
        if self.use_llm:
            hybrid_result.llm_explanation = self._generate_llm_explanation(
                patient_data=patient_data,
                rule_result=rule_result,
                cbr_result=cbr_result,
                hybrid_result=hybrid_result
            )
        else:
            hybrid_result.llm_explanation = self._generate_template_explanation(
                rule_result=rule_result,
                cbr_result=cbr_result
            )
        
        # Step 6: Decide on case retention
        if cbr_result["max_similarity"] < self.similarity_threshold:
            self._retain_case(patient_data, hybrid_result)
            hybrid_result.case_retained = True
        
        return hybrid_result
    
    def _patient_to_facts(self, patient_data: Dict[str, Any]) -> List[str]:
        """
        Convert patient dictionary to FOL facts
        
        All numeric comparisons are pre-computed as Boolean facts here.
        The rule parser does NOT support inline arithmetic.
        """
        facts = []
        p = "Patient"  # Patient constant
        
        # Extract common values for complex conditions
        age = patient_data.get("age")
        bmi = patient_data.get("bmi")
        fpg = patient_data.get("fpg")
        a1c = patient_data.get("a1c")
        random_glucose = patient_data.get("random_glucose")
        
        # Demographic facts
        if age is not None:
            facts.append(f"Age({p}, {age})")
            if age < 35:
                facts.append(f"YoungAge({p})")
            else:
                facts.append(f"OlderAge({p})")
        
        if bmi is not None:
            facts.append(f"BMI({p}, {bmi})")
            if bmi < 25:
                facts.append(f"LowBMI({p})")
            else:
                facts.append(f"HighBMI({p})")
        
        # Lab value facts and abstractions
        if fpg is not None:
            facts.append(f"FpgValue({p}, {fpg})")
            if fpg >= 126:
                facts.append(f"FpgHigh({p})")
            elif 100 <= fpg < 126:
                facts.append(f"FpgImpaired({p})")
            else:
                facts.append(f"FpgNormal({p})")
        
        if "fpg_repeat" in patient_data and patient_data["fpg_repeat"] is not None:
            fpg_repeat = patient_data["fpg_repeat"]
            facts.append(f"FpgRepeatValue({p}, {fpg_repeat})")
            if fpg_repeat >= 126:
                facts.append(f"FpgRepeatHigh({p})")
        
        if a1c is not None:
            facts.append(f"A1cValue({p}, {a1c})")
            if a1c >= 6.5:
                facts.append(f"A1cHigh({p})")
            elif 5.7 <= a1c < 6.5:
                facts.append(f"A1cImpaired({p})")
            else:
                facts.append(f"A1cNormal({p})")
            
            if a1c >= 10.0:
                facts.append(f"VeryHighA1c({p})")
        
        if "a1c_repeat" in patient_data and patient_data["a1c_repeat"] is not None:
            a1c_repeat = patient_data["a1c_repeat"]
            if a1c_repeat >= 6.5:
                facts.append(f"A1cRepeatHigh({p})")
        
        if random_glucose is not None:
            facts.append(f"RandomGlucoseValue({p}, {random_glucose})")
            if random_glucose >= 200:
                facts.append(f"RandomGlucoseHigh({p})")
            if random_glucose > 250:
                facts.append(f"VeryHighGlucose({p})")
        
        if "ketones" in patient_data:
            ketones = patient_data["ketones"]
            facts.append(f"KetoneStatus({p}, {ketones.capitalize()})")
            if ketones.lower() in ["negative", "none"]:
                facts.append(f"NoKetones({p})")
        
        # Symptom facts
        symptoms = patient_data.get("symptoms", [])
        for symptom in symptoms:
            symptom_capitalized = symptom.replace("-", "").replace("_", "").capitalize()
            facts.append(f"Symptom({p}, {symptom_capitalized})")
        
        # Risk factor facts
        if patient_data.get("family_history"):
            facts.append(f"FamilyHistory({p}, Diabetes)")
        
        if patient_data.get("strong_family_history"):
            facts.append(f"StrongFamilyHistory({p})")
        
        if patient_data.get("gestational_history"):
            facts.append(f"GestationalHistory({p})")
        
        if patient_data.get("hypertension"):
            facts.append(f"Hypertension({p})")
        
        if patient_data.get("dyslipidemia"):
            facts.append(f"Dyslipidemia({p})")
        
        if patient_data.get("smoking_history"):
            facts.append(f"SmokingHistory({p})")
        
        if patient_data.get("gradual_onset"):
            facts.append(f"GradualOnset({p})")
        
        if patient_data.get("no_autoantibodies"):
            facts.append(f"NoAutoantibodies({p})")
        
        # ===== COMPLEX BOOLEAN FACTS FOR NEW RULES =====
        
        # Rule 15: HighRiskScreening
        # Age >= 35 OR (BMI >= 25 AND has_risk_factors)
        needs_screening = False
        if age is not None and age >= 35:
            needs_screening = True
        elif bmi is not None and bmi >= 25:
            # Check for any risk factors
            if (patient_data.get("family_history") or
                patient_data.get("gestational_history") or
                patient_data.get("hypertension") or
                patient_data.get("dyslipidemia")):
                needs_screening = True
        
        if needs_screening:
            facts.append(f"HighRiskScreening({p})")
        
        # Rule 16: LADAIndicators
        # Will check after DiabetesConfirmed is derived, but can pre-compute age/BMI criteria
        # Age 30-50 AND BMI < 25 AND gradual onset
        if (age is not None and 30 <= age <= 50 and
            bmi is not None and bmi < 25 and
            patient_data.get("gradual_onset")):
            # Note: DiabetesConfirmed will be checked by the rule itself
            facts.append(f"LADAIndicators({p})")
        
        # Rule 18: MetabolicSyndrome
        # HighBMI AND Hypertension AND Dyslipidemia
        if (bmi is not None and bmi >= 25 and
            patient_data.get("hypertension") and
            patient_data.get("dyslipidemia")):
            facts.append(f"MetabolicSyndrome({p})")
        
        # Rule 19: CVDRiskHigh
        # DiabetesConfirmed AND (age >= 40 OR Hypertension OR SmokingHistory)
        # Note: DiabetesConfirmed check will be done by rule, we compute the risk factors
        has_cvd_risk_factors = False
        if age is not None and age >= 40:
            has_cvd_risk_factors = True
        elif patient_data.get("hypertension") or patient_data.get("smoking_history"):
            has_cvd_risk_factors = True
        
        # This fact will only matter if DiabetesConfirmed, but we add it anyway
        if has_cvd_risk_factors:
            facts.append(f"CVDRiskHigh({p})")
        
        # Rule 20: LifestylePriority
        # (PrediabetesIFG OR PrediabetesIGT) AND HighBMI
        # The Prediabetes facts will be derived by rules, BMI check here
        if bmi is not None and bmi >= 25:
            # Note: The rule itself will check for PrediabetesIFG/IGT
            facts.append(f"LifestylePriority({p})")
        
        # MedicalEmergencyHHS
        # VeryHighGlucose AND age >= 60 AND NoKetones
        glucose_source = random_glucose if random_glucose is not None else fpg
        if (glucose_source is not None and glucose_source > 250 and
            age is not None and age >= 60 and
            patient_data.get("ketones", "").lower() in ["negative", "none"]):
            facts.append(f"MedicalEmergencyHHS({p})")
        
        # MonogenicDiabetesSuspicion
        # DiabetesConfirmed AND YoungAge AND StrongFamilyHistory AND NoAutoantibodies
        # Most of these will be checked by the rule, but we ensure the boolean facts exist
        """ if (age is not None and age < 35 and
            patient_data.get("strong_family_history") and
            patient_data.get("no_autoantibodies")):
            facts.append(f"MonogenicDiabetesSuspicion({p})") """
        
        return facts 
    
    def _run_rule_engine(self, facts: List[str]) -> Dict[str, Any]:
        """
        Run forward chaining with rule engine
        
        Returns dict with:
        - derived_facts: All facts derived by rules
        - explanations: Explanation for each derived fact
        - diagnosis_facts: Extracted diagnosis-related facts
        """
        try:
            predicate_facts = []
            for fact_str in facts:
                try:
                    # Use the _parse_predicate function from logic_parser
                    pred = _parse_predicate(fact_str)
                    predicate_facts.append(pred)
                except Exception as e:
                    print(f"  Skipping unparseable fact: {fact_str} ({e})")
            
            # Run forward chaining
            derived_facts, explanations = forward_chaining_with_explanation(
                facts=predicate_facts,
                rules=self.rules if self.rules else [],
                strategy=NoConflictResolution(),
                verbose=False
            )
            
            # Ensure derived_facts is a list
            if derived_facts is None:
                derived_facts = []
            if explanations is None:
                explanations = {}
            
            # Extract key conclusions
            diagnosis = None
            classification = None
            recommendations = []
            is_emergency = False
            
            for fact in derived_facts:
                fact_str = str(fact)
                
                # Check for diagnoses
                if "DiabetesConfirmed" in fact_str:
                    diagnosis = "Diabetes Mellitus (Confirmed)"
                elif "PrediabetesIFG" in fact_str and not diagnosis:
                    diagnosis = "Prediabetes (Impaired Fasting Glucose)"
                elif "PrediabetesIGT" in fact_str and not diagnosis:
                    diagnosis = "Prediabetes (Impaired Glucose Tolerance)"
                elif "ProvisionalHyperglycemia" in fact_str and not diagnosis:
                    diagnosis = "Provisional Hyperglycemia"
                
                # Check for classification
                if "SuspectedType1" in fact_str:
                    classification = "Type 1 Diabetes (Suspected)"
                elif "SuspectedType2" in fact_str:
                    classification = "Type 2 Diabetes (Suspected)"
                elif "LADAIndicators" in fact_str:
                    classification = "LADA (Latent Autoimmune Diabetes)"
                elif "MonogenicDiabetesSuspicion" in fact_str:
                    classification = "Monogenic Diabetes (Suspected)"
                
                # Check for emergencies
                if "MedicalEmergencyDKA" in fact_str:
                    is_emergency = True
                    diagnosis = "Diabetic Ketoacidosis (DKA) - EMERGENCY"
                elif "MedicalEmergencyHHS" in fact_str:
                    is_emergency = True
                    diagnosis = "Hyperosmolar Hyperglycemic State (HHS) - EMERGENCY"
                
                # Extract recommendations
                if "Recommend" in fact_str:
                    # Parse recommendation from fact
                    # Format: Recommend(Patient, Constant(ActionName))
                    try:
                        # Extract the action name from Constant(ActionName)
                        rec_part = fact_str.split(",", 1)[1].strip().rstrip(")")
                        
                        # Remove "Constant(" wrapper if present
                        if "Constant(" in rec_part:
                            rec_name = rec_part.split("Constant(")[1].rstrip(")")
                        else:
                            rec_name = rec_part
                        
                        # Format recommendation nicely
                        rec_formatted = rec_name.replace("_", " ").replace("-", " ").title()
                        rec_formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', rec_formatted)
                        rec_formatted = rec_formatted.title().strip()
                        recommendations.append(rec_formatted)
                    except:
                        pass
            
            return {
                "derived_facts": derived_facts,
                "explanations": explanations,
                "diagnosis": diagnosis,
                "classification": classification,
                "recommendations": list(set(recommendations)),  # Remove duplicates
                "is_emergency": is_emergency
            }
        
        except Exception as e:
            print(f"⚠️  Rule engine error: {e}")
            return {
                "derived_facts": [],
                "explanations": {},
                "diagnosis": None,
                "classification": None,
                "recommendations": [],
                "is_emergency": False
            }
    
    def _run_cbr(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run case-based reasoning
        
        Returns dict with:
        - retrieved_cases: Top-k similar cases
        - adapted_solution: Solution adapted from best match
        - max_similarity: Highest similarity score
        """
        try:
            # Convert patient data to CaseFeatures
            features = CaseFeatures(
                age=patient_data.get("age"),
                bmi=patient_data.get("bmi"),
                fpg=patient_data.get("fpg"),
                a1c=patient_data.get("a1c"),
                random_glucose=patient_data.get("random_glucose"),
                fpg_repeat=patient_data.get("fpg_repeat"),
                ketones=patient_data.get("ketones", "negative"),
                symptoms=patient_data.get("symptoms", [])
            )
            
            # Retrieve similar cases
            retrieved = retrieve_top_k(features, self.case_library, k=3)
            
            # Adapt solution from best match
            adapted_solution = None
            if retrieved:
                best_case = retrieved[0]["case"]
                adapted_solution = {
                    "diagnosis": best_case.solution.diagnosis,
                    "status": best_case.solution.status,
                    "recommendations": best_case.solution.recommendations.copy()
                }
            
            max_similarity = retrieved[0]["similarity"] if retrieved else 0.0
            
            return {
                "retrieved_cases": retrieved,
                "adapted_solution": adapted_solution,
                "max_similarity": max_similarity
            }
        
        except Exception as e:
            print(f"⚠️  CBR error: {e}")
            return {
                "retrieved_cases": [],
                "adapted_solution": None,
                "max_similarity": 0.0
            }
    
    def _fuse_results(
        self,
        patient_id: str,
        patient_data: Dict[str, Any],
        rule_result: Dict[str, Any],
        cbr_result: Dict[str, Any]
    ) -> HybridDiagnosisResult:
        """
        Fuse rule-based and CBR results into unified diagnosis
        
        Priority:
        1. Emergency rules always override
        2. Rule-based diagnosis takes precedence for confirmed cases
        3. CBR provides supporting evidence and recommendations
        4. Confidence combines rule certainty with CBR similarity
        """
        result = HybridDiagnosisResult(patient_id=patient_id)
        
        # Copy rule-based results
        result.rule_derived_facts = rule_result["derived_facts"]
        result.rule_explanations = rule_result["explanations"]
        result.rule_diagnosis = rule_result["diagnosis"]
        result.rule_classification = rule_result["classification"]
        result.rule_recommendations = rule_result["recommendations"]
        
        # Copy CBR results
        result.similar_cases = cbr_result["retrieved_cases"]
        result.cbr_adapted_solution = cbr_result["adapted_solution"]
        if cbr_result["adapted_solution"]:
            result.cbr_diagnosis = cbr_result["adapted_solution"]["diagnosis"]
        
        # Determine emergency status
        result.is_emergency = rule_result["is_emergency"]
        
        # Fuse diagnosis (priority: emergency > rule > CBR)
        if result.is_emergency:
            result.primary_diagnosis = rule_result["diagnosis"]
            result.confidence_score = 1.0  # Emergency rules are definitive
        elif rule_result["diagnosis"]:
            result.primary_diagnosis = rule_result["diagnosis"]
            if rule_result["classification"]:
                result.primary_diagnosis = rule_result["classification"]
            # Confidence based on rule certainty and CBR agreement
            rule_confidence = 0.8 if "Confirmed" in rule_result["diagnosis"] else 0.6
            cbr_agreement = 0.2 * cbr_result["max_similarity"]
            result.confidence_score = min(1.0, rule_confidence + cbr_agreement)
        elif cbr_result["adapted_solution"]:
            result.primary_diagnosis = result.cbr_diagnosis
            result.confidence_score = cbr_result["max_similarity"]
        else:
            result.primary_diagnosis = "Insufficient data for diagnosis"
            result.confidence_score = 0.0
        
        # Merge recommendations (rules + CBR, prioritizing rules)
        all_recommendations = rule_result["recommendations"].copy()
        if cbr_result["adapted_solution"]:
            cbr_recs = cbr_result["adapted_solution"].get("recommendations", [])
            all_recommendations.extend(cbr_recs)

        # Smart deduplication - normalize and keep first occurrence
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            # Normalize for comparison (lowercase, no spaces/dashes)
            rec_normalized = rec.lower().replace(" ", "").replace("-", "")
            if rec_normalized not in seen:
                seen.add(rec_normalized)
                unique_recs.append(rec)
        
        result.final_recommendations = unique_recs
        
        return result
    
    def _generate_llm_explanation(
        self,
        patient_data: Dict[str, Any],
        rule_result: Dict[str, Any],
        cbr_result: Dict[str, Any],
        hybrid_result: HybridDiagnosisResult
    ) -> str:
        """Generate natural language explanation using LLM"""
        if not self.llm:
            return self._generate_template_explanation(rule_result, cbr_result)
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Build context for LLM
            system_prompt = """You are a clinical decision support assistant explaining diabetes 
            diagnoses to patients in clear, compassionate language. Your explanations should:
            1. Be medically accurate but accessible (avoid jargon)
            2. Reference specific test results that led to the diagnosis
            3. Mention similar cases when relevant to build trust
            4. Be 3-5 sentences maximum
            5. End with a clear next step
            
            Do not invent facts not present in the data provided."""
            
            # Format patient data
            patient_summary = f"""
            Age: {patient_data.get('age', 'N/A')}
            BMI: {patient_data.get('bmi', 'N/A')}
            Fasting Glucose: {patient_data.get('fpg', 'N/A')} mg/dL
            A1C: {patient_data.get('a1c', 'N/A')}%
            Random Glucose: {patient_data.get('random_glucose', 'N/A')} mg/dL
            Symptoms: {', '.join(patient_data.get('symptoms', [])) or 'None reported'}
            """.strip()
            
            # Format CBR context
            cbr_context = ""
            if cbr_result["retrieved_cases"]:
                best = cbr_result["retrieved_cases"][0]
                cbr_context = f"\nMost similar case: {best['case'].id} (similarity: {best['similarity']:.2f})"
            
            user_prompt = f"""
            Patient Data:
            {patient_summary}
            
            Rule Engine Diagnosis: {rule_result.get('diagnosis', 'N/A')}
            Classification: {rule_result.get('classification', 'N/A')}
            {cbr_context}
            
            Final Diagnosis: {hybrid_result.primary_diagnosis}
            Confidence: {hybrid_result.confidence_score:.2f}
            
            Explain this diagnosis to the patient clearly and compassionately.
            """
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            return response.content.strip()
        
        except Exception as e:
            print(f"⚠️  LLM explanation failed: {e}")
            return self._generate_template_explanation(rule_result, cbr_result)
    
    def _generate_template_explanation(
        self,
        rule_result: Dict[str, Any],
        cbr_result: Dict[str, Any]
    ) -> str:
        """Generate template-based explanation when LLM unavailable"""
        parts = []
        
        if rule_result["diagnosis"]:
            parts.append(f"Diagnosis: {rule_result['diagnosis']}")
        
        if rule_result["classification"]:
            parts.append(f"Classification: {rule_result['classification']}")
        
        if rule_result["is_emergency"]:
            parts.append("⚠️ EMERGENCY: Immediate medical attention required.")
        
        if cbr_result["retrieved_cases"]:
            best = cbr_result["retrieved_cases"][0]
            parts.append(f"Similar to case {best['case'].id} (similarity: {best['similarity']:.2f})")
        
        if rule_result["recommendations"]:
            recs = ", ".join(rule_result["recommendations"][:3])
            parts.append(f"Recommended: {recs}")
        
        return " ".join(parts)
    
    def _retain_case(
        self,
        patient_data: Dict[str, Any],
        diagnosis_result: HybridDiagnosisResult
    ):
        """Retain novel case in library for future learning"""
        try:
            # Create new case
            features = CaseFeatures(
                age=patient_data.get("age"),
                bmi=patient_data.get("bmi"),
                fpg=patient_data.get("fpg"),
                a1c=patient_data.get("a1c"),
                random_glucose=patient_data.get("random_glucose"),
                fpg_repeat=patient_data.get("fpg_repeat"),
                ketones=patient_data.get("ketones", "negative"),
                symptoms=patient_data.get("symptoms", [])
            )
            
            solution = CaseSolution(
                diagnosis=diagnosis_result.primary_diagnosis or "Unknown",
                status="Confirmed" if "Confirmed" in (diagnosis_result.primary_diagnosis or "") else "Provisional",
                recommendations=diagnosis_result.final_recommendations
            )
            
            new_case = Case(
                id=f"case_{patient_data.get('patient_id', 'unknown')}",
                features=features,
                solution=solution,
                outcome="pending",
                notes=f"Automatically retained. Confidence: {diagnosis_result.confidence_score:.2f}"
            )
            
            # Add to library
            self.case_library.add_case(new_case)
            
        except Exception as e:
            print(f"⚠️  Case retention failed: {e}")


def main():
    """Example usage of hybrid system"""
    
    # Initialize system
    system = HybridDiabetesSystem(
        rules_kb_path="knowledge_bases/rules_expanded.kb",
        case_library_path="data/case_library.json",
        use_llm=True
    )
    
    # Example patient
    patient = {
        "patient_id": "demo_patient",
        "age": 45,
        "bmi": 27.0,
        "fpg": 130,
        "a1c": 6.0,
        "symptoms": [],
        "family_history": True
    }
    
    # Diagnose
    result = system.diagnose(patient)
    
    # Display results
    print("\n" + "="*60)
    print("HYBRID DIABETES DIAGNOSIS SYSTEM")
    print("="*60)
    print(f"\nPatient: {result.patient_id}")
    print(f"Diagnosis: {result.primary_diagnosis}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"\nExplanation:\n{result.llm_explanation}")
    print(f"\nRecommendations:")
    for rec in result.final_recommendations:
        print(f"  • {rec}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
