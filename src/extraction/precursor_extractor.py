import re
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cosine
from src.preprocessing.safety_taxonomy import HAZARDS, BARRIER_FAILURES
from src.preprocessing.feature_engineering import SentenceEmbeddingFeatures
from src.config import SEMANTIC_SIMILARITY_THRESHOLD
from src.extraction.normalization import CONSEQUENCE_NORMALIZATION, normalize_consequence

ACTIVITIES_MAPPING = {
    "Maintenance": [r"maintenance", r"repair(?:ing)?", r"servic(?:ing|e)", r"replac(?:ing|e)", r"work(?:ing)? on", r"inspect(?:ing|ion)?", r"overhaul", r"corrective maintenance", r"troubleshooting"],
    "Drilling": [r"drill(?:ing)?"],
    "Pressure Testing": [r"pressure test(?:ing)?", r"hydrotest(?:ing)?", r"pneumatic test", r"leak test"],
    "Welding": [r"weld(?:ing|er)?", r"torch cutting", r"gas cutting", r"flame cutting", r"grinding", r"soldering"],
    "Hot Work": [r"hot\s*work", r"spark(?:s)?"],
    "Electrical Work": [r"electrical work", r"wiring", r"electrician", r"electrical maintenance", r"panel maintenance", r"cable work", r"breaker maintenance"],
    "Confined Space Entry": [r"confined[- ]space", r"enter(?:ing|ed)?.*vessel", r"enter(?:ing|ed)?.*tank", r"vessel entry", r"tank entry", r"manhole entry", r"vessel inspection", r"tank cleaning", r"sump maintenance"],
    "Mechanical Lifting": [r"lift(?:ing)?", r"hoist(?:ing)?", r"rigging", r"load movement", r"crane operation", r"forklift operation"],
    "Crane Operation": [r"crane"],
    "Vehicle Movement": [r"driv(?:ing|e)", r"vehicle", r"truck", r"reversing", r"transport", r"forklift movement", r"transporting personnel", r"material delivery", r"site patrol", r"heavy haulage"],
    "Excavation": [r"excavat(?:ing|ion)?", r"digging", r"trench(?:ing)?"],
    "Working at Height": [r"height", r"scaffold(?:ing)?", r"roof", r"elevated", r"ladder work", r"platform"],
    "Chemical Handling": [r"chemical", r"toxic"],
    "Loading/Unloading": [r"load(?:ing)?", r"unload(?:ing)?"],
    "Equipment Cleaning": [r"clean(?:ing)?"],
    "Operations": [r"valve operation", r"process operation", r"alarm testing", r"non-routine task"],
    "Contractor Work": [r"contractor work", r"contractor"],
    "Pipefitting": [r"pipefitting", r"pipework"],
    "Painting": [r"paint(?:ing)?"],
    "Torquing": [r"torquing", r"torque"]
}

EQUIPMENT_REGEX = [
    r"\b(?:Pump|Compressor|Valve|Tank|Vessel|Spool|Crane|Truck|Panel|System|Block|Reactor|Winch|Separator|Heat Exchanger)\s+[A-Z0-9-]+\b",
    r"\b[A-Z0-9-]+\s+(?:pump|compressor|valve|tank|vessel|spool|crane|truck|panel|system|block|reactor|winch|separator|heat exchanger)\b",
    r"\bpump\b", r"\bcompressor\b", r"\bvalve\b", r"\btank\b", r"\bvessel\b", r"\bboiler\b", r"\bsump\b", r"\breactor\b", r"\bseparator\b", r"\bheat exchanger\b",
    r"\bpipeline\b", r"\bflange\b", r"\bcrane\b", r"\bforklift\b", r"\bdrill pipe\b", r"\bpipe spool\b", r"\bpipe rack\b",
    r"\bwelding machine\b", r"\bscaffold\b", r"\bladder\b", r"\belectrical panel\b", r"\bmcc panel\b", r"\bdcs system\b", 
    r"\bpressure vessel\b", r"\brigging\b", r"\blifting sling\b", r"\bshackle\b", r"\bgantry\b", r"\bchain block\b", r"\bwinch\b",
    r"\btorque wrench\b", r"\bpressure hose\b", r"\bhose\b", r"\bangle grinder\b", r"\btorch\b", r"\bcherry picker\b", r"\bexcavator\b", r"\bcrew bus\b", r"\blight vehicle\b",
    r"\bworkshop\b", r"\butility area\b", r"\bcamp facility\b", r"\bprocess module\b", r"\bgrating\b", r"\bwalkway\b"
]

UNSAFE_ACTIONS_MAPPING = {
    "Entering danger zone": [r"stepped into.*danger zone", r"entered exclusion zone", r"walked below.*load", r"stood.*load", r"stood under.*load", r"stood below.*suspended", r"entered.*swing radius"],
    "Opening pressurized equipment": [r"cracked open.*flange", r"opening pressurized", r"opened.*before.*depressurized"],
    "Bypassing interlock": [r"bypassed.*interlock", r"disabled.*alarm"],
    "Working without permit": [r"working without.*permit", r"permit.*expired", r"started.*without.*permit"],
    "Starting hot work without valid gas test": [r"cutting.*after.*gas test.*expired", r"hot work.*without.*gas test"],
    "Driving without seatbelt": [r"without seatbelt"],
    "Working at height without fall protection": [r"without.*fall protection", r"without connecting.*harness", r"not tied off"],
    "Entering confined space without authorization": [r"entered.*without.*confined-space permit"]
}

UNSAFE_CONDITIONS_MAPPING = {
    "Pressurized line": [r"line remained pressurized", r"pressurized line"],
    "Suspended load present": [r"suspended load was positioned", r"load was suspended"],
    "Missing barricade": [r"missing barricade", r"no barricade"],
    "Exposed electrical conductor": [r"exposed.*electrical", r"exposed wiring"],
    "Combustible material near hot work": [r"combustible.*near hot work", r"flammable.*near hot work"],
    "Unstable scaffold": [r"unstable scaffold", r"scaffold.*unstable"],
    "Poor ventilation": [r"poor ventilation", r"inadequate ventilation"],
    "Leaking hydrocarbon line": [r"leaking hydrocarbon", r"hydrocarbon leak"],
    "Defective guard": [r"defective guard", r"missing guard"],
    "Unprotected opening": [r"unprotected opening", r"open hole"]
}

CANONICAL_HAZARD_DESCRIPTIONS = {
    "Pressurized equipment": "pressure trapped in a hose, vessel, or piping system that has not been bled down or depressurized.",
    "Stored energy": "mechanical, electrical, or pneumatic energy remaining in equipment that is not properly locked out or isolated.",
    "Loss of Containment": "fluid, gas, or liquid started weeping, leaking, or spilling from a flange, hose, line, or vessel.",
    "Suspended load": "heavy object being lifted or hoisted by a crane, placing personnel at risk of crushing.",
    "Line-of-fire exposure": "personnel positioned in a danger zone, swing radius, or pinch point where they could be struck by moving objects.",
    "Fall from height": "working on an elevated platform, scaffold, roof, or ladder with potential to fall."
}

POTENTIAL_CONSEQUENCES_MAPPING = {
    "Pressurized equipment": ["Major Process Safety Event"],
    "Loss of containment": ["Major Process Safety Event"],
    "Fire/explosion": ["Fire Injury", "Explosion Injury", "Major Process Safety Event"],
    "H2S exposure": ["Toxic Exposure", "Loss of Consciousness", "Fatal Exposure"],
    "Chemical exposure": ["Toxic Exposure"],
    "Fall from height": ["Fall Injury", "Fatal Fall"],
    "Electrical energy": ["Electrical Injury"],
    "Suspended load": ["Crushing Injury", "Struck-By Injury"],
    "Vehicle interaction": ["Vehicle / Pedestrian Injury"]
}

class SafetyPrecursorExtractor:
    def __init__(self):
        self.embedder = SentenceEmbeddingFeatures()
        self.hazard_embeddings_cache = {}
        self._precompute_taxonomy_embeddings()

    def _precompute_taxonomy_embeddings(self):
        if self.embedder.model is not None:
            for hazard, desc in CANONICAL_HAZARD_DESCRIPTIONS.items():
                self.hazard_embeddings_cache[hazard] = self.embedder.transform([desc])[0]

    def _split_into_clauses(self, text: str) -> List[str]:
        return [c.strip() for c in re.split(r'[.?!;]| but | and | so | while ', text) if len(c.strip()) > 10]

    def _is_negated(self, text: str, match_span: Tuple[int, int], category: str = None) -> bool:
        search_window = text[max(0, match_span[0] - 30):match_span[0]].lower().strip()
        matched_text = text[match_span[0]:match_span[1]].lower()

        # Handle 'out of the line of fire', 'remained clear of'
        if category == "Line-of-fire exposure":
            extended_search = text[max(0, match_span[0] - 60):match_span[0]].lower().strip()
            if any(p in extended_search for p in ["out of the", "out of", "remained clear", "stayed clear", "nobody was in the"]):
                return True
            if "nobody" in search_window:
                return True

        if "depressurized" in matched_text:
            if re.search(r"\b(?:not|failed to)\b", search_window):
                return False

        neg_matches = list(re.finditer(r"\b(no|not|without|failed to|didn't|did not|wasn't|was not)\b", search_window))
        if neg_matches:
            last_neg_match = neg_matches[-1]
            text_between = search_window[last_neg_match.end():].strip()
            if not re.search(r"[.,;]", text_between):
                words_between = [w for w in text_between.split() if w]
                if len(words_between) <= 1:
                    return True
                    
        # "verified zero energy", "fully depressurized", "not pressurized"
        if category in ["Stored energy", "Pressurized equipment"]:
            if "zero energy" in search_window or "fully depressurize" in search_window or "not pressurized" in search_window:
                return True
                
        return False

    def _deduplicate_extractions(self, extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        # Priority: Exact/Regex (HIGH) > Synonym (MEDIUM) > Semantic (LOW)
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        seen_values = {}
        for ext in extractions:
            val = ext['value']
            conf_score = priority_map.get(ext.get('confidence', 'LOW'), 1)
            
            if val not in seen_values:
                seen_values[val] = ext
            else:
                existing_score = priority_map.get(seen_values[val].get('confidence', 'LOW'), 1)
                if conf_score > existing_score:
                    seen_values[val] = ext
        return list(seen_values.values())

    def _extract_activities(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        for activity, patterns in ACTIVITIES_MAPPING.items():
            for pattern in patterns:
                for match in re.finditer(r"\b" + pattern + r"\b", text_lower):
                    results.append({
                        "value": activity,
                        "evidence": text[match.start():match.end()],
                        "confidence": "HIGH" if len(pattern) > 8 else "MEDIUM",
                        "match_method": "regex"
                    })
        return self._deduplicate_extractions(results)

    def _extract_equipment(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for pattern in EQUIPMENT_REGEX:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                val = match.group()
                val_lower = val.lower()
                
                eq_type = "Unknown"
                if "pump" in val_lower: eq_type = "Pump"
                elif "compressor" in val_lower: eq_type = "Compressor"
                elif "valve" in val_lower: eq_type = "Valve"
                elif "tank" in val_lower: eq_type = "Tank"
                elif "vessel" in val_lower or "boiler" in val_lower or "sump" in val_lower or "reactor" in val_lower or "separator" in val_lower or "heat exchanger" in val_lower: eq_type = "Vessel"
                elif "pipeline" in val_lower or "line" in val_lower or "spool" in val_lower: eq_type = "Pipeline"
                elif "crane" in val_lower or "winch" in val_lower or "gantry" in val_lower or "chain block" in val_lower: eq_type = "Crane"
                elif "forklift" in val_lower: eq_type = "Forklift"
                elif "scaffold" in val_lower: eq_type = "Scaffold"
                elif "ladder" in val_lower: eq_type = "Ladder"
                elif "hose" in val_lower: eq_type = "Hose"
                elif "wrench" in val_lower: eq_type = "Torque Wrench"
                elif "grinder" in val_lower: eq_type = "Grinder"
                elif "torch" in val_lower: eq_type = "Torch"
                elif "truck" in val_lower or "vehicle" in val_lower or "bus" in val_lower or "cherry picker" in val_lower or "excavator" in val_lower: eq_type = "Vehicle"
                elif "panel" in val_lower or "dcs" in val_lower: eq_type = "Electrical Panel"
                elif "workshop" in val_lower or "area" in val_lower or "camp" in val_lower or "module" in val_lower: eq_type = "Facility"
                elif "rack" in val_lower or "grating" in val_lower: eq_type = "Structure"
                else: eq_type = val.title()

                if any(val_lower in r["value"].lower() and len(val) < len(r["value"]) for r in results):
                    continue
                results = [r for r in results if not (r["value"].lower() in val_lower and len(r["value"]) < len(val))]
                
                if not any(r["value"].lower() == val_lower for r in results):
                    results.append({
                        "value": val.title(),
                        "type": eq_type,
                        "evidence": val,
                        "confidence": "HIGH",
                        "match_method": "regex"
                    })
        return results

    def _extract_hazards(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        
        hazard_variants = {
            "Stored energy": ["stored energy", "tensioned line", "spring", "line had not been proven dead", "residual pressure", "energized", "trapped pressure", "line remained live", "still energized", "pressure remained", "depressurization incomplete", "unexpected startup", "whip from parted cable"],
            "Pressurized equipment": ["pressurized", "not depressurized", "pressure hose failed", "hose failed", "hose snapped", "high pressure release", "overpressure"],
            "Loss of containment": ["weeping", "fluid leak", "line leak", "spill", "loss of containment", "hose burst", "flange leak", "seal leaking", "pipe leaking", "liquid escaping", "fluid escaping", "spray", "discharge", "release", "hose snapped", "steam leak"],
            "Hydrocarbon release": ["hydrocarbon release", "oil leak", "gas leak", "hydrocarbon leaking", "hydrocarbon leaked"],
            "Fire/explosion": ["fire", "explosion", "combustible", "flash fire", "burning", "ignited"],
            "Flammable atmosphere": ["flammable atmosphere"],
            "Toxic gas": ["toxic gas", "toxic fumes"],
            "H2S exposure": ["h2s", "h2s detected", "hydrogen sulfide", "sour gas"],
            "Electrical energy": ["electrical", "live wire", "electrical shock", "electrocution"],
            "Suspended load": ["suspended load", "dropped suspended load", "swinging load", "rigging failure", "dropped load", "crane tip-over"],
            "Dropped object": ["dropped object", "fell from"],
            "Line-of-fire exposure": ["line of fire", "danger zone", "pinch point", "swing radius", "caught between objects", "caught in pinch point", "struck by moving object"],
            "Vehicle interaction": ["vehicle collision", "rollover", "loss of control", "pedestrian impact", "struck by vehicle", "vehicle incident"],
            "Moving machinery": ["moving machinery", "rotating machinery", "entanglement"],
            "Confined atmosphere": ["confined space", "confined-space", "vessel entry", "poor ventilation", "oxygen deficiency", "engulfment"],
            "Chemical exposure": ["chemical spill", "acid leak", "caustic exposure", "chemical exposure"],
            "Fall from height": ["fall from height", "falling from", "open grating", "unprotected edge", "fall hazard"],
            "Unidentified hazards": ["unidentified hazards", "simultaneous operations clash", "uncontrolled work"]
        }

        # L1/L2 Matching
        for std_haz, variants in hazard_variants.items():
            for variant in variants:
                for match in re.finditer(r"\b" + re.escape(variant) + r"\b", text_lower):
                    # Fire exclusion logic
                    if std_haz == "Fire/explosion" and "fire" in variant:
                        # Exclude "line of fire", "fire gas panel", etc
                        context_start = max(0, match.start() - 30)
                        context_end = min(len(text_lower), match.end() + 30)
                        context = text_lower[context_start:context_end]
                        if any(k in context for k in ["line of fire", "fire gas panel", "fire & gas", "fire extinguisher"]):
                            continue
                            
                    if not self._is_negated(text, match.span(), category=std_haz):
                        # Guardrails
                        if std_haz == "Fire/explosion" and not any(k in text_lower for k in ["ignition", "combustible", "fire", "explosion", "flame", "burn", "ignited"]):
                            continue
                        if std_haz == "H2S exposure" and not any(k in text_lower for k in ["h2s", "hydrogen sulfide", "sour gas"]):
                            continue
                        if std_haz == "Hydrocarbon release" and not any(k in text_lower for k in ["hydrocarbon", "oil", "gas", "fuel"]):
                            continue
                                
                        results.append({
                            "value": std_haz,
                            "evidence": text[match.start():match.end()],
                            "confidence": "HIGH" if len(variant) > 6 else "MEDIUM",
                            "match_method": "regex"
                        })

        # L3 Semantic Matching
        if self.embedder.model is not None:
            clauses = self._split_into_clauses(text)
            if clauses:
                clause_embeddings = self.embedder.transform(clauses)
                
                for idx, c_emb in enumerate(clause_embeddings):
                    clause = clauses[idx]
                    
                    if re.search(r"\b(no|not|without|failed to)\b", clause.lower()) and "depressurize" not in clause.lower():
                        continue

                    for canonical_haz, h_emb in self.hazard_embeddings_cache.items():
                        # Skip if already found via regex
                        if any(r["value"] == canonical_haz for r in results):
                            continue
                            
                        sim = 1 - cosine(c_emb, h_emb)
                        if sim >= SEMANTIC_SIMILARITY_THRESHOLD:
                            # Semantic restrictions
                            if canonical_haz == "Fire/explosion" and not any(k in clause.lower() for k in ["ignition", "fire", "explosion", "flame", "burn"]):
                                continue
                            if canonical_haz == "Fall from height" and not any(k in clause.lower() for k in ["height", "fall", "scaffold", "ladder", "roof", "platform", "elevat"]):
                                continue
                            if canonical_haz == "Suspended load" and not any(k in clause.lower() for k in ["load", "lift", "hoist", "crane", "swing", "suspend"]):
                                continue

                            results.append({
                                "value": canonical_haz,
                                "evidence": clause,
                                "confidence": "LOW",
                                "match_method": "semantic",
                                "similarity_score": round(float(sim), 4)
                            })

        return self._deduplicate_extractions(results)

    def _extract_barrier_failures(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        
        extended_mappings = dict(BARRIER_FAILURES)
        extended_mappings["ENERGY_ISOLATION_FAILURE"].extend([
            r"loto was not applied", r"equipment wasn't isolated", r"before loto was applied", r"without loto",
            r"line not proven dead", r"isolation not verified", r"lockout not applied", r"residual pressure remained", r"failed to isolate"
        ])
        extended_mappings["WORK_AUTHORIZATION_FAILURE"].extend([
            r"without a valid.*permit", r"without.*authorization", r"without.*permit", r"permit.*expired",
            r"no valid permit", r"permit not issued", r"unauthorized work"
        ])
        extended_mappings["GAS_TEST_FAILURE"].extend([
            r"gas test.*expired", r"no atmospheric testing", r"atmosphere not tested", r"gas reading validity expired"
        ])
        extended_mappings["FALL_PROTECTION_FAILURE"].extend([
            r"without connecting.*harness", r"harness not connected", r"no lifeline", r"lanyard unclipped", r"fall arrest missing"
        ])
        extended_mappings["LIFTING_CONTROL_FAILURE"].extend([
            r"defective sling", r"rigging failure", r"tag line missing", r"lifting zone not controlled", r"overloaded crane"
        ])
        extended_mappings["LINE_OF_FIRE_CONTROL_FAILURE"].extend([
            r"worker in danger zone", r"personnel in swing radius", r"pinch point exposure"
        ])
        extended_mappings["SAFETY_DEVICE_BYPASS"].extend([
            r"bypassed.*interlock"
        ])
        
        for failure_cat, phrases in extended_mappings.items():
            for phrase in phrases:
                for match in re.finditer(phrase, text_lower):
                    results.append({
                        "value": failure_cat,
                        "evidence": text[match.start():match.end()],
                        "confidence": "HIGH",
                        "match_method": "regex"
                    })
        return self._deduplicate_extractions(results)

    def _extract_unsafe_actions(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        for action, patterns in UNSAFE_ACTIONS_MAPPING.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    results.append({
                        "value": action,
                        "evidence": text[match.start():match.end()],
                        "confidence": "HIGH",
                        "match_method": "regex"
                    })
        return self._deduplicate_extractions(results)

    def _extract_unsafe_conditions(self, text: str) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        for cond, patterns in UNSAFE_CONDITIONS_MAPPING.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    results.append({
                        "value": cond,
                        "evidence": text[match.start():match.end()],
                        "confidence": "HIGH",
                        "match_method": "regex"
                    })
        return self._deduplicate_extractions(results)

    def _extract_consequences(self, text: str, hazards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        text_lower = text.lower()
        
        # Explicit consequence phrases mapped directly from text
        for canonical_cons, variants in CONSEQUENCE_NORMALIZATION.items():
            for variant in variants:
                for match in re.finditer(r"\b" + re.escape(variant) + r"\b", text_lower):
                    results.append({
                        "value": canonical_cons,
                        "evidence": text[match.start():match.end()],
                        "confidence": "HIGH",
                        "match_method": "regex"
                    })
                    
        # Inferred consequences from hazards
        for h in hazards:
            val = h["value"].lower()
            for key, cons_list in POTENTIAL_CONSEQUENCES_MAPPING.items():
                if key.lower() == val:
                    for cons in cons_list:
                        results.append({
                            "value": cons,
                            "evidence": h["evidence"], 
                            "confidence": "MEDIUM",
                            "match_method": "synonym"
                        })
        return self._deduplicate_extractions(results)

    def _generate_precursor_tags(self, extracted: Dict[str, Any]) -> List[str]:
        tags = set()
        for h in extracted.get("hazards", []):
            val = h["value"].lower()
            if "stored energy" in val: tags.add("stored-energy")
            if "pressurized" in val: tags.add("pressurized-system")
            if "line-of-fire" in val: tags.add("line-of-fire")
            if "suspended load" in val: tags.add("suspended-load")
            if "vehicle" in val: tags.add("vehicle-interaction")
            
        for b in extracted.get("barrier_failures", []):
            val = b["value"]
            if val == "ENERGY_ISOLATION_FAILURE": tags.add("missing-loto")
            if val == "WORK_AUTHORIZATION_FAILURE": tags.add("expired-permit")
            if val == "GAS_TEST_FAILURE": tags.add("gas-test-failure")
            if val == "FALL_PROTECTION_FAILURE": tags.add("fall-protection")
            if val == "SAFETY_DEVICE_BYPASS": tags.add("bypassed-interlock")
            if val == "BARRICADING_FAILURE": tags.add("missing-barricade")
            if val == "COMMUNICATION_FAILURE": tags.add("communication-failure")
            
        for c in extracted.get("unsafe_conditions", []):
            if "confined space" in c["value"].lower():
                tags.add("confined-space-control")
                
        return sorted(list(tags))

    def extract(self, text: str) -> Dict[str, Any]:
        if type(text) != str:
            raise ValueError("Input must be a string")

        activities = self._extract_activities(text)
        equipment = self._extract_equipment(text)
        hazards = self._extract_hazards(text)
        barrier_failures = self._extract_barrier_failures(text)
        unsafe_actions = self._extract_unsafe_actions(text)
        unsafe_conditions = self._extract_unsafe_conditions(text)
        consequences = self._extract_consequences(text, hazards)
        
        output = {
            "activities": activities,
            "equipment": equipment,
            "hazards": hazards,
            "barrier_failures": barrier_failures,
            "unsafe_actions": unsafe_actions,
            "unsafe_conditions": unsafe_conditions,
            "potential_consequences": consequences
        }
        
        output["precursor_tags"] = self._generate_precursor_tags(output)
        return output

    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.extract(text) for text in texts]
