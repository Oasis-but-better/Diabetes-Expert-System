"""
Enhanced Forward Chaining Engine with Conflict Resolution and Explanation

This module extends the base logic solver with:
1. Multiple conflict resolution strategies
2. Explanation tracking for derived facts
3. Detailed trace output for the diabetes diagnosis system
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from logic_parser import Predicate, Rule, Term, Variable, Constant
from logic_solver import unify, standardize_variables, _substitute

# Type definitions
Substitution = Dict[str, Term]


@dataclass
class DerivedFact:
    """
    Represents a derived fact with its explanation/provenance
    
    Attributes:
        fact: The predicate that was derived
        rule: The rule that derived it
        cycle: The cycle number when it was derived
        from_facts: The facts used in the derivation
        substitution: The variable substitution used
    """
    fact: Predicate
    rule: Rule
    cycle: int
    from_facts: List[Predicate]
    substitution: Substitution
    
    def __str__(self) -> str:
        return f"{self.fact} (derived in cycle {self.cycle})"
    
    def get_explanation(self) -> str:
        """Generate human-readable explanation of how this fact was derived"""
        rule_str = str(self.rule)
        from_facts_str = ", ".join(str(f) for f in self.from_facts)
        sub_str = ", ".join(f"{k}={v}" for k, v in self.substitution.items()) if self.substitution else "none"
        
        return f"""
Fact: {self.fact}
Derived by: {rule_str}
In cycle: {self.cycle}
Using facts: {from_facts_str}
With substitution: {sub_str}
"""


class ConflictResolutionStrategy:
    """Base class for conflict resolution strategies"""
    
    def select_rules(self, applicable_rules: List[Tuple[Rule, List[Substitution]]], 
                    facts: List[Predicate]) -> List[Tuple[Rule, List[Substitution]]]:
        """
        Select which rules to fire when multiple rules are applicable
        
        Args:
            applicable_rules: List of (rule, substitutions) pairs
            facts: Current working memory
            
        Returns:
            Filtered list of rules to fire
        """
        raise NotImplementedError


class NoConflictResolution(ConflictResolutionStrategy):
    """Strategy 1: Fire all applicable rules (no conflict resolution)"""
    
    def select_rules(self, applicable_rules: List[Tuple[Rule, List[Substitution]]], 
                    facts: List[Predicate]) -> List[Tuple[Rule, List[Substitution]]]:
        return applicable_rules


class SpecificityOrdering(ConflictResolutionStrategy):
    """
    Strategy 2: Prefer rules with more antecedents (more specific)
    
    More specific rules (with more conditions) fire before general rules.
    This implements a form of "best match" conflict resolution.
    """
    
    def select_rules(self, applicable_rules: List[Tuple[Rule, List[Substitution]]], 
                    facts: List[Predicate]) -> List[Tuple[Rule, List[Substitution]]]:
        if not applicable_rules:
            return []
        
        # Sort by number of antecedents (descending)
        sorted_rules = sorted(applicable_rules, 
                            key=lambda x: len(x[0].antecedents), 
                            reverse=True)
        
        # Return only the most specific rules (those with max antecedents)
        max_specificity = len(sorted_rules[0][0].antecedents)
        return [r for r in sorted_rules if len(r[0].antecedents) == max_specificity]


class RecencyOrdering(ConflictResolutionStrategy):
    """
    Strategy 3: Prefer rules that match recently added facts
    
    This prioritizes rules that use the most recently derived facts,
    implementing a "refractoriness" approach.
    """
    
    def __init__(self):
        self.fact_timestamps: Dict[str, int] = {}
        self.timestamp = 0
    
    def update_timestamps(self, new_facts: List[Predicate]):
        """Update timestamps for newly added facts"""
        for fact in new_facts:
            self.fact_timestamps[str(fact)] = self.timestamp
            self.timestamp += 1
    
    def select_rules(self, applicable_rules: List[Tuple[Rule, List[Substitution]]], 
                    facts: List[Predicate]) -> List[Tuple[Rule, List[Substitution]]]:
        if not applicable_rules:
            return []
        
        # Calculate average recency for each rule based on facts it uses
        rule_recencies = []
        for rule, subs in applicable_rules:
            max_recency = -1
            for fact in facts:
                fact_str = str(fact)
                if fact_str in self.fact_timestamps:
                    max_recency = max(max_recency, self.fact_timestamps[fact_str])
            rule_recencies.append((rule, subs, max_recency))
        
        # Sort by recency (descending)
        sorted_rules = sorted(rule_recencies, key=lambda x: x[2], reverse=True)
        
        # Return most recent
        if sorted_rules:
            max_recency = sorted_rules[0][2]
            return [(r, s) for r, s, rec in sorted_rules if rec == max_recency]
        return applicable_rules


def forward_chaining_with_explanation(
    facts: List[Predicate],
    rules: List[Rule],
    strategy: ConflictResolutionStrategy = None,
    verbose: bool = True
) -> Tuple[List[Predicate], Dict[str, DerivedFact]]:
    """
    Forward chaining with conflict resolution and explanation tracking
    
    Args:
        facts: Initial fact base
        rules: List of rules
        strategy: Conflict resolution strategy (default: no conflict resolution)
        verbose: Print detailed trace
        
    Returns:
        Tuple of (all_facts, explanations_dict)
    """
    if strategy is None:
        strategy = NoConflictResolution()
    
    # Initialize working memory
    known_facts = facts.copy()
    known_facts_set: Set[Predicate] = set(known_facts)
    
    # Explanation tracking
    explanations: Dict[str, DerivedFact] = {}
    
    # Mark initial facts
    for fact in facts:
        explanations[str(fact)] = DerivedFact(
            fact=fact,
            rule=None,  # type: ignore
            cycle=0,
            from_facts=[],
            substitution={}
        )
    
    if verbose:
        print("=" * 80)
        print("FORWARD CHAINING WITH CONFLICT RESOLUTION")
        print("=" * 80)
        print(f"\nStrategy: {strategy.__class__.__name__}")
        print(f"Initial facts: {len(facts)}")
        print(f"Rules: {len(rules)}\n")
    
    cycle = 1
    
    while True:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"CYCLE {cycle}")
            print(f"{'=' * 80}")
            print(f"Current working memory size: {len(known_facts)} facts\n")
        
        # Find all applicable rules
        applicable_rules: List[Tuple[Rule, List[Substitution]]] = []
        
        for rule_idx, rule in enumerate(rules):
            if verbose:
                print(f"Checking Rule {rule_idx + 1}: {rule}")
            
            # Standardize variables
            std_rule = standardize_variables(rule)
            
            # Find all substitutions that satisfy the rule
            substitutions = _find_all_substitutions(std_rule.antecedents, known_facts, {})
            
            if substitutions:
                if verbose:
                    print(f"  ✓ Rule applicable with {len(substitutions)} substitution(s)")
                applicable_rules.append((std_rule, substitutions))
            else:
                if verbose:
                    print(f"  ✗ Rule not applicable")
        
        if not applicable_rules:
            if verbose:
                print(f"\n{'=' * 80}")
                print("NO APPLICABLE RULES - HALTING")
                print(f"{'=' * 80}\n")
            break
        
        # Apply conflict resolution
        if verbose and len(applicable_rules) > 1:
            print(f"\n{'-' * 80}")
            print(f"CONFLICT RESOLUTION: {len(applicable_rules)} rules applicable")
            print(f"Strategy: {strategy.__class__.__name__}")
            print(f"{'-' * 80}")
        
        selected_rules = strategy.select_rules(applicable_rules, known_facts)
        
        if verbose and len(applicable_rules) > 1:
            print(f"Selected {len(selected_rules)} rule(s) to fire\n")
        
        # Fire selected rules
        new_derived: List[DerivedFact] = []
        
        for rule, substitutions in selected_rules:
            for theta in substitutions:
                # Apply substitution to get a ground consequent
                derived_fact = _substitute(rule.consequent, theta)
                
                # Only add if it's truly new
                if derived_fact not in known_facts_set:
                    # Track which facts were used
                    used_facts = []
                    for ant in rule.antecedents:
                        ant_sub = _substitute(ant, theta)
                        if ant_sub in known_facts:
                            used_facts.append(ant_sub)
                    
                    derived_fact_obj = DerivedFact(
                        fact=derived_fact,
                        rule=rule,
                        cycle=cycle,
                        from_facts=used_facts,
                        substitution=theta
                    )
                    
                    new_derived.append(derived_fact_obj)
                    known_facts_set.add(derived_fact)
                    explanations[str(derived_fact)] = derived_fact_obj
        
        # If no new facts derived, we've reached the fixed point
        if not new_derived:
            if verbose:
                print(f"\n{'=' * 80}")
                print("NO NEW FACTS DERIVED - HALTING")
                print(f"{'=' * 80}\n")
            break
        
        # Add new facts to working memory
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"DERIVED {len(new_derived)} NEW FACT(S):")
            print(f"{'-' * 80}")
        
        for derived in new_derived:
            if verbose:
                print(f"  • {derived.fact}")
            known_facts.append(derived.fact)
            
            # Update recency if using recency ordering
            if isinstance(strategy, RecencyOrdering):
                strategy.update_timestamps([derived.fact])
        
        cycle += 1
    
    if verbose:
        print(f"\nForward chaining complete.")
        print(f"Final working memory: {len(known_facts)} facts")
        print(f"Derived facts: {len(known_facts) - len(facts)}")
        print(f"Cycles: {cycle - 1}\n")
    
    return known_facts, explanations


def _find_all_substitutions(
    antecedents: List[Predicate],
    facts_list: List[Predicate],
    theta: Substitution
) -> List[Substitution]:
    """
    Find all substitutions that satisfy all antecedents
    Uses exhaustive depth-first search
    """
    if not antecedents:
        return [theta]
    
    first_ant = antecedents[0]
    rest_ants = antecedents[1:]
    
    # Apply current substitution
    first_ant_substituted = _substitute(first_ant, theta)
    
    results = []
    for fact in facts_list:
        unified_theta = unify(first_ant_substituted, fact, theta)
        if unified_theta is not None:
            sub_results = _find_all_substitutions(rest_ants, facts_list, unified_theta)
            results.extend(sub_results)
    
    return results


def print_explanations(explanations: Dict[str, DerivedFact], facts_to_explain: List[str] = None):
    """
    Print explanations for derived facts
    
    Args:
        explanations: Dictionary mapping fact strings to DerivedFact objects
        facts_to_explain: Optional list of specific facts to explain (defaults to all)
    """
    print("\n" + "=" * 80)
    print("EXPLANATION COMPONENT")
    print("=" * 80)
    
    if facts_to_explain is None:
        # Explain all derived facts (not initial facts)
        facts_to_explain = [k for k, v in explanations.items() if v.cycle > 0]
    
    if not facts_to_explain:
        print("\nNo derived facts to explain.")
        return
    
    print(f"\nExplaining {len(facts_to_explain)} derived fact(s):\n")
    
    for fact_str in sorted(facts_to_explain):
        if fact_str in explanations:
            derived = explanations[fact_str]
            if derived.cycle > 0:  # Only explain derived facts, not initial ones
                print("-" * 80)
                print(derived.get_explanation())
        else:
            print(f"\nNo explanation found for: {fact_str}")
