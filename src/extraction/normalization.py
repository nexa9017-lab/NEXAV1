from typing import List

def _normalize_string(val: str, mappings: dict) -> str:
    val_lower = val.lower().strip()
    
    # 1. Exact match canonical
    for canonical in mappings.keys():
        if val_lower == canonical.lower():
            return canonical
            
    # 2. Exact match variants
    for canonical, variants in mappings.items():
        if val_lower in [v.lower() for v in variants]:
            return canonical

    # 3. Substring match (longest variants first)
    all_variants = []
    for canonical, variants in mappings.items():
        for variant in variants:
            all_variants.append((variant, canonical))
            
    all_variants.sort(key=lambda x: len(x[0]), reverse=True)
    
    for variant, canonical in all_variants:
        if variant.lower() in val_lower:
            return canonical

    # Fallback to Title Case if unknown
    return val.title()

ACTIVITY_NORMALIZATION = {
    "Maintenance": ["repair", "servicing", "inspection", "overhaul", "maintenance", "corrective maintenance", "fix", "replace", "pump overhaul", "electrical troubleshooting", "filter replacement", "system maintenance", "general maintenance"],
    "Hot Work": ["welding", "weld", "torch cutting", "gas cutting", "grinding", "flame cutting", "hot work", "soldering"],
    "Mechanical Lifting": ["lifting", "hoist", "rigging", "crane", "load movement", "forklift operation"],
    "Working at Height": ["scaffold", "elevated platform", "ladder", "roof", "height"],
    "Confined Space Entry": ["vessel entry", "tank entry", "confined space", "entering vessel", "manhole", "vessel inspection", "tank cleaning", "internal repair", "sump maintenance"],
    "Electrical Work": ["electrical", "panel", "cable", "breaker", "wiring"],
    "Pressure Testing": ["hydrotest", "pressure test", "pneumatic test", "leak test"],
    "Vehicle Movement": ["driving", "reversing", "vehicle", "transport", "forklift", "truck", "transporting personnel", "material delivery", "site patrol", "heavy haulage"],
    "Excavation": ["excavat", "digging", "trench"],
    "Loading/Unloading": ["load", "unload"],
    "Cleaning": ["clean", "wash"],
    "Operations": ["valve operation", "process operation", "alarm testing", "non-routine task", "manual handling"],
    "Contractor Work": ["contractor work"],
    "Pipefitting": ["pipefitting"],
    "Painting": ["painting"],
    "Torquing": ["torquing"]
}

HAZARD_NORMALIZATION = {
    "Stored Energy": ["stored energy", "tensioned line", "spring", "unexpected startup", "release of stored energy", "whip from parted cable"],
    "Pressurized Equipment": ["pressurized", "pressure", "hose", "flange", "high pressure release", "overpressure"],
    "Loss of Containment": ["weeping", "fluid leak", "line leak", "spill", "loss of containment", "steam leak"],
    "Hydrocarbon Release": ["hydrocarbon", "oil leak", "gas leak"],
    "Fire/Explosion": ["fire", "explosion", "combustible"],
    "Flammable Atmosphere": ["flammable", "gas test"],
    "Toxic Gas": ["toxic gas", "toxic fumes"],
    "H2S Exposure": ["h2s", "hydrogen sulfide"],
    "Electrical Energy": ["electrical", "live wire", "panel"],
    "Suspended Load": ["suspended load", "crane", "hoist", "lifting", "dropped suspended load", "swinging load", "crane tip-over", "rigging failure"],
    "Dropped Object": ["dropped object", "fell from"],
    "Line-of-Fire Exposure": ["line of fire", "danger zone", "pinch point", "swing radius", "struck by moving object", "caught in pinch point", "caught between objects"],
    "Vehicle Interaction": ["vehicle", "truck", "forklift", "pedestrian impact", "struck by vehicle", "vehicle collision", "rollover", "loss of control"],
    "Moving Machinery": ["moving machinery", "rotating"],
    "Confined Atmosphere": ["confined space", "vessel entry", "poor ventilation", "confined atmosphere", "oxygen deficiency", "engulfment"],
    "Chemical Exposure": ["chemical", "acid", "caustic"],
    "Fall from Height": ["height", "fall", "scaffold", "ladder", "roof", "platform", "open grating"],
    "Unidentified Hazards": ["unidentified hazards", "simultaneous operations clash", "uncontrolled work"]
}

BARRIER_NORMALIZATION = {
    "ENERGY_ISOLATION_FAILURE": ["loto missing", "isolation not verified", "loto not applied", "line not proven dead", "lockout not applied", "residual pressure", "line still energized", "failed to isolate"],
    "WORK_AUTHORIZATION_FAILURE": ["permit expired", "no valid permit", "permit not issued", "unauthorized", "ptw not issued", "jsa not discussed", "permit missing", "wrong permit type"],
    "GAS_TEST_FAILURE": ["gas test expired", "no atmospheric testing", "atmosphere not tested", "gas reading validity expired", "gas test not performed", "no gas check"],
    "FALL_PROTECTION_FAILURE": ["harness not connected", "no lifeline", "lanyard unclipped", "fall arrest missing", "harness not clipped", "not tied off", "scaffold tag missing", "missing toe boards"],
    "LIFTING_CONTROL_FAILURE": ["overloaded crane", "defective sling", "rigging failure", "tag line missing", "lifting zone not controlled", "damaged slings used", "outriggers not extended", "lift plan not followed", "no tag lines used"],
    "LINE_OF_FIRE_CONTROL_FAILURE": ["worker in danger zone", "stood below load", "personnel in swing radius", "pinch point exposure", "exclusion zone breached", "tension line snapped", "in the line of fire", "standing in pinch point"],
    "TRAFFIC_CONTROL_FAILURE": ["speeding", "seatbelt not worn", "using mobile phone", "fatigued driver", "brakes defective"],
    "SAFETY_DEVICE_BYPASS": ["interlock bridged", "alarm disabled", "safety valve isolated", "override left active", "bypassed interlock"],
    "BARRICADING_FAILURE": ["no barricade below", "barricades ignored", "missing barricade", "area not barricaded"],
    "COMMUNICATION_FAILURE": ["attendant absent", "fire watcher absent"],
    "PPE_FAILURE": ["missing ppe", "safety glasses not worn"],
    "PROCEDURE_NON_COMPLIANCE": ["procedure not followed", "combustibles not cleared", "sparks falling"]
}

EQUIPMENT_NORMALIZATION = {
    "Pump": ["pump"],
    "Compressor": ["compressor"],
    "Valve": ["valve"],
    "Tank": ["tank"],
    "Vessel": ["vessel", "boiler", "sump"],
    "Pipeline": ["pipeline", "line", "spool"],
    "Flange": ["flange"],
    "Crane": ["crane", "winch", "gantry", "chain block"],
    "Forklift": ["forklift"],
    "Drill Pipe": ["drill pipe", "drill"],
    "Welding Machine": ["welding machine", "welder"],
    "Scaffold": ["scaffold", "scaffolding"],
    "Ladder": ["ladder"],
    "Electrical Panel": ["electrical panel", "panel", "mcc panel", "dcs system", "fire & gas panel", "switchgear"],
    "Pressure Vessel": ["pressure vessel"],
    "Rigging": ["rigging", "sling", "shackle", "pallet"],
    "Hose": ["hose", "pressure hose"],
    "Torque Wrench": ["torque wrench", "wrench"],
    "Grinder": ["grinder"],
    "Torch": ["torch"],
    "Vehicle": ["truck", "bus", "vehicle", "car"],
    "Facility": ["process module", "workshop", "utility area", "camp facility"],
    "Structure": ["pipe rack", "grating", "walkway"]
}

TAGS_NORMALIZATION = {
    "stored-energy": ["stored-energy", "stored energy", "energy isolation", "pressure"],
    "pressurized-system": ["pressurized-system", "pressure"],
    "missing-loto": ["missing-loto", "loto"],
    "line-of-fire": ["line-of-fire", "danger zone", "swing radius"],
    "expired-permit": ["expired-permit", "permit", "work authorization"],
    "gas-test-failure": ["gas-test-failure", "gas test"],
    "fall-protection": ["fall-protection", "height", "harness"],
    "suspended-load": ["suspended-load", "lifting", "crane"],
    "bypassed-interlock": ["bypassed-interlock", "interlock", "bypass"],
    "missing-barricade": ["missing-barricade", "barricade"],
    "vehicle-interaction": ["vehicle-interaction", "vehicle", "traffic"],
    "confined-space-control": ["confined-space-control", "confined space"],
    "communication-failure": ["communication-failure", "communication"]
}

CONSEQUENCE_NORMALIZATION = {
    "Serious Burns": ["severe burns", "fatal burns", "burn"],
    "Electrical Injury": ["electric shock", "shock"],
    "Fatal Electrical Injury": ["electrocution", "fatal shock"],
    "Asphyxiation": ["asphyxiation", "suffocation"],
    "Toxic Exposure": ["toxic gas exposure", "toxic exposure", "fumes"],
    "Crushing Injury": ["crushed by load", "crush fatality", "crush injury", "crushing"],
    "Struck-By Injury": ["struck by", "impact injury"],
    "Fall Injury": ["fall injury", "fall"],
    "Fatal Fall": ["fatal fall", "fell to death"],
    "Vehicle / Pedestrian Injury": ["pedestrian impact", "run over", "vehicular injury"],
    "Major Equipment Damage": ["equipment destruction", "structural collapse", "equipment damage"],
    "Fire Injury": ["fire injury"],
    "Explosion Injury": ["explosion injury"],
    "Major Process Safety Event": ["process safety event", "catastrophic release", "facility explosion"],
    "Loss of Consciousness": ["loss of consciousness", "passed out"],
    "Fatal Exposure": ["fatal exposure", "fatality"],
}

# The synthetic generator sometimes mixes consequence terms directly into the "hazards" column.
# We will use this set to filter those out of the hazard evaluation.
CONSEQUENCE_TERMS_IN_GT = {
    "severe burns", "electric shock", "electrocution", "asphyxiation", "crushed by load",
    "pedestrian impact", "equipment destruction", "process safety event", "toxic gas exposure", 
    "toxic fumes", "fatality", "crush fatality", "fatal burns", "major explosion", 
    "catastrophic release", "multiple fatalities", "serious injury due to lack of hazard awareness", 
    "process incident", "fatal traffic accident", "severe whiplash", "crush injuries to pedestrian",
    "structural collapse", "multiple serious injuries", "major fire", "facility explosion"
}


def normalize_activity(activity: str) -> str:
    return _normalize_string(activity, ACTIVITY_NORMALIZATION)

def normalize_hazard(hazard: str) -> str:
    # If the hazard ground-truth is actually a consequence, we can return a special token or None
    if hazard.lower().strip() in CONSEQUENCE_TERMS_IN_GT:
        return "CONSEQUENCE_IGNORE"
    return _normalize_string(hazard, HAZARD_NORMALIZATION)

def normalize_barrier_failure(failure: str) -> str:
    return _normalize_string(failure, BARRIER_NORMALIZATION)

def normalize_equipment(equipment: str) -> str:
    return _normalize_string(equipment, EQUIPMENT_NORMALIZATION)

def normalize_precursor_tag(tag: str) -> str:
    return _normalize_string(tag, TAGS_NORMALIZATION).lower().replace(" ", "-")

def normalize_consequence(consequence: str) -> str:
    return _normalize_string(consequence, CONSEQUENCE_NORMALIZATION)

def normalize_list(items: List[str], norm_func) -> List[str]:
    normalized = []
    for item in items:
        n = norm_func(item)
        if n and n != "CONSEQUENCE_IGNORE" and n not in normalized:
            normalized.append(n)
    return normalized
