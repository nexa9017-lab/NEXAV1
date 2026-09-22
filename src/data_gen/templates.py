# Synthetic Data Templates and Dictionaries

SITES = ["Site Alpha", "Facility Bravo", "Platform Charlie", "Refinery Delta", "Terminal Echo", "Station Foxtrot"]
LOCATIONS = ["Pump Room", "Deck 3", "Boiler Area", "Loading Bay", "Main Workshop", "Drilling Floor", "Storage Tank A", "Control Room", "Pipe Rack 2"]
REPORT_TYPES = ["Unsafe Act", "Unsafe Condition", "Near Miss", "Incident"]

# Map of Life Saving Rules to their associated data for generation
LSR_DATA = {
    "Energy Isolation": {
        "activities": ["Maintenance", "Inspection", "Repair", "Pipefitting"],
        "hazards": ["Pressurized hydrocarbon release", "Electrical shock", "Stored energy discharge", "Steam leak", "H2S exposure"],
        "barrier_failures": ["LOTO not applied", "Isolation valve leaking", "Bleed valve closed", "Padlock missing", "Energy not verified zero"],
        "equipment": ["Pump P-204", "Compressor C-10", "Electrical Panel EP-5", "Valve V-99", "Heat Exchanger HX-2"],
        "potential_consequences": ["Fire, serious injury or fatal exposure", "Fatal electrocution", "Severe chemical burns", "Asphyxiation"],
        "precursors": ["stored energy", "inadequate isolation", "pressurized equipment", "live electrical", "toxic fluid"],
        "sif_templates": [
            "During {activity} of {equipment}, the technician started loosening the flange while the line was still pressurized. Isolation valve was closed but {barrier_failure}. {hazard} occurred. No injury occurred but potential was high.",
            "Operator was performing {activity} on {equipment}. It was discovered that {barrier_failure}, leaving the system vulnerable to {hazard}. Work was stopped immediately.",
            "While inspecting {equipment}, {hazard} was narrowly avoided because {barrier_failure}."
        ],
        "non_sif_templates": [
            "Noticed {equipment} had a minor weep during {activity}. LOTO was applied correctly, but cleanup was delayed.",
            "While preparing for {activity} on {equipment}, found that the lock for LOTO was slightly rusted, making it hard to use.",
            "The tag for {equipment} was faded during {activity}. No {hazard} was present, but tag needs replacement."
        ]
    },
    "Work at Height": {
        "activities": ["Scaffolding", "Painting", "Structural Inspection", "Roof Repair", "Rigging"],
        "hazards": ["Fall from height", "Dropped object", "Unstable platform", "Open grating"],
        "barrier_failures": ["Harness not clipped", "Scaffold tag missing", "No barricade below", "Lanyard defective", "Missing toe boards"],
        "equipment": ["Scaffold Tower B", "Crane C-1", "Pipe Rack Level 3", "Roof Access Ladder", "Cherry Picker"],
        "potential_consequences": ["Fatality from fall", "Serious head injury from dropped object", "Multiple fractures"],
        "precursors": ["working aloft", "unsecured tools", "missing fall protection", "open hole", "edge proximity"],
        "sif_templates": [
            "Worker observed performing {activity} on {equipment} at 15 meters. It was noted that {barrier_failure}. A {hazard} could have resulted in a fatality.",
            "During {activity}, a 5kg wrench was dropped from {equipment}. {barrier_failure} meant the area below was not clear. {hazard} was a near miss.",
            "{activity} was ongoing near {equipment} when grating was removed. {barrier_failure} left the hole unprotected, creating a high risk of {hazard}."
        ],
        "non_sif_templates": [
            "Worker's chin strap on helmet was loose during {activity} on {equipment}.",
            "Scaffolding tubes for {equipment} were stored untidily, causing a minor trip hazard during {activity}.",
            "During {activity}, worker used the correct harness but the inspection log was not filled in for {equipment}."
        ]
    },
    "Line of Fire": {
        "activities": ["Lifting", "Pressure Testing", "Torquing", "Grinding", "Trenching"],
        "hazards": ["Struck by moving object", "High pressure release", "Whip from parted cable", "Caught between objects"],
        "barrier_failures": ["Standing in pinch point", "Exclusion zone breached", "Barricades ignored", "Tension line snapped", "Safety pin removed"],
        "equipment": ["Winch W-2", "Hydraulic Torque Wrench", "Excavator", "High Pressure Hose", "Drill Pipe"],
        "potential_consequences": ["Crush injury", "Amputation", "Fatal impact", "Severe lacerations"],
        "precursors": ["tensioned line", "suspended load", "moving machinery", "pinch point", "pressure test"],
        "sif_templates": [
            "While conducting {activity} with {equipment}, an IP stepped into the danger zone. {barrier_failure} resulted in the worker being in the Line of Fire for {hazard}.",
            "During {activity}, {equipment} failed unexpectedly. Because {barrier_failure}, the worker was exposed to {hazard}. Fortunately, the worker jumped back in time.",
            "{activity} was being performed. {barrier_failure} allowed personnel to walk directly under {equipment}, posing a severe {hazard}."
        ],
        "non_sif_templates": [
            "Worker performing {activity} on {equipment} had their hands slightly close to a rough edge, causing a minor scratch.",
            "During {activity}, a piece of non-tensioned wire from {equipment} was left on the floor.",
            "Barricade tape for {activity} near {equipment} was blown down by the wind. No operations were ongoing."
        ]
    },
    "Confined Space": {
        "activities": ["Tank Cleaning", "Internal Inspection", "Vessel Welding"],
        "hazards": ["Toxic atmosphere", "Oxygen deficiency", "Engulfment", "Heat exhaustion"],
        "barrier_failures": ["Gas test not performed", "Attendant absent", "Ventilation failed", "Rescue plan inadequate"],
        "equipment": ["Storage Tank A", "Separator V-10", "Underground Sump", "Reactor R-5"],
        "potential_consequences": ["Asphyxiation", "Fatality due to toxic gas", "Loss of consciousness"],
        "precursors": ["restricted access", "hazardous atmosphere", "poor ventilation", "deep excavation"],
        "sif_templates": [
            "Prior to {activity} in {equipment}, it was discovered that {barrier_failure}. This could have led to {hazard} and potential fatality.",
            "Worker entered {equipment} for {activity}. Suddenly the alarm sounded. {barrier_failure} meant the worker was exposed to {hazard}.",
            "During {activity} inside {equipment}, {barrier_failure} was noted. This is a critical failure leading to high risk of {hazard}."
        ],
        "non_sif_templates": [
            "The entry log for {equipment} during {activity} was smudged with dirt.",
            "Worker doing {activity} in {equipment} complained about poor lighting, though gas levels were normal.",
            "Tools used for {activity} in {equipment} were left outside the entrance untidily."
        ]
    },
    "Safe Mechanical Lifting": {
        "activities": ["Crane Lifting", "Forklift Operation", "Hoisting", "Rigging"],
        "hazards": ["Dropped suspended load", "Crane tip-over", "Swinging load", "Rigging failure"],
        "barrier_failures": ["Overloaded crane", "Damaged slings used", "Outriggers not extended", "No tag lines used"],
        "equipment": ["Mobile Crane", "Forklift F-3", "Overhead Gantry", "Chain Block"],
        "potential_consequences": ["Crush fatality", "Structural collapse", "Multiple serious injuries"],
        "precursors": ["suspended load", "heavy lifting", "blind lift", "dynamic loading"],
        "sif_templates": [
            "During {activity} using {equipment}, the load began to swing wildly. {barrier_failure} caused a loss of control, resulting in {hazard}.",
            "{activity} was taking place with {equipment}. It was found that {barrier_failure}, which is a severe violation that could result in {hazard}.",
            "A heavy skid was being moved during {activity}. {barrier_failure} led to a near miss involving {hazard} near {equipment}."
        ],
        "non_sif_templates": [
            "The horn on {equipment} sounded a bit quiet during {activity}.",
            "Paint on {equipment} used for {activity} is peeling off.",
            "Operator of {equipment} during {activity} forgot his safety glasses but was in an enclosed cab."
        ]
    },
    "Hot Work": {
        "activities": ["Welding", "Grinding", "Torch Cutting", "Soldering"],
        "hazards": ["Fire", "Explosion", "Severe burns", "Toxic fumes"],
        "barrier_failures": ["Combustibles not cleared", "Fire watcher absent", "Gas test expired", "Sparks falling to lower deck"],
        "equipment": ["Welding Machine W-1", "Angle Grinder", "Oxy-Acetylene Torch", "Pipe Spool P-10"],
        "potential_consequences": ["Major fire", "Facility explosion", "Fatal burns"],
        "precursors": ["ignition source", "flammable atmosphere", "sparks", "hydrocarbon presence"],
        "sif_templates": [
            "{activity} was started on {equipment} near a live process area. {barrier_failure} was observed, posing a massive risk of {hazard}.",
            "During {activity} with {equipment}, {barrier_failure} allowed slag to ignite a small fire. {hazard} potential was extremely high.",
            "A contractor began {activity} using {equipment}. A safety walk discovered {barrier_failure}, immediately stopping work due to {hazard} risk."
        ],
        "non_sif_templates": [
            "Welder doing {activity} with {equipment} had untucked coveralls.",
            "The welding lead for {equipment} was slightly tangled during {activity}.",
            "Fire extinguisher near {equipment} for {activity} was due for monthly inspection next week."
        ]
    },
    "Bypassing Safety Controls": {
        "activities": ["Process Operation", "System Maintenance", "Alarm Testing"],
        "hazards": ["Process safety event", "Overpressure", "Loss of containment", "Equipment destruction"],
        "barrier_failures": ["Interlock bridged", "Alarm disabled", "Safety valve isolated", "Override left active"],
        "equipment": ["DCS System", "Pressure Safety Valve PSV-90", "Fire & Gas Panel", "Shutdown Valve SDV-1"],
        "potential_consequences": ["Major explosion", "Catastrophic release", "Multiple fatalities"],
        "precursors": ["disabled safety device", "bypassed interlock", "ignored alarm", "unauthorized override"],
        "sif_templates": [
            "While investigating {equipment}, it was found that {barrier_failure} during {activity}. This exposed the facility to {hazard}.",
            "Operator engaged in {activity} admitted to bypassing safety logic on {equipment}. {barrier_failure} is a severe breach leading to {hazard}.",
            "Audit of {equipment} post-{activity} revealed {barrier_failure}. Without this protection, {hazard} could have occurred."
        ],
        "non_sif_templates": [
            "Indicator bulb on {equipment} was burnt out during {activity}.",
            "Operator logged {activity} for {equipment} in the wrong logbook.",
            "Screen on {equipment} was dirty, making it hard to read during {activity}."
        ]
    },
    "Work Authorization": {
        "activities": ["General Maintenance", "Contractor Work", "Non-routine Task"],
        "hazards": ["Unidentified hazards", "Simultaneous operations clash", "Uncontrolled work"],
        "barrier_failures": ["PTW not issued", "JSA not discussed", "Wrong permit type", "Expired permit"],
        "equipment": ["Process Module 4", "Workshop", "Utility Area", "Camp Facility"],
        "potential_consequences": ["Serious injury due to lack of hazard awareness", "Process incident"],
        "precursors": ["unauthorized work", "poor communication", "simops", "inadequate planning"],
        "sif_templates": [
            "Crew commenced {activity} at {equipment} but {barrier_failure}. Unaware of SIMOPS, they were exposed to {hazard}.",
            "During a walkaround, found personnel doing {activity} on {equipment}. {barrier_failure} meant they had no controls for {hazard}.",
            "Critical {activity} on {equipment} proceeded despite {barrier_failure}. This breakdown in authorization risked {hazard}."
        ],
        "non_sif_templates": [
            "Permit for {activity} at {equipment} had a minor spelling mistake in the supervisor's name.",
            "JSA for {activity} near {equipment} was printed on crumpled paper.",
            "Worker doing {activity} at {equipment} did not sign the permit exactly on the line."
        ]
    },
    "Driving": {
        "activities": ["Transporting Personnel", "Material Delivery", "Site Patrol", "Heavy Haulage"],
        "hazards": ["Vehicle collision", "Rollover", "Pedestrian impact", "Loss of control"],
        "barrier_failures": ["Speeding", "Seatbelt not worn", "Using mobile phone", "Fatigued driver", "Brakes defective"],
        "equipment": ["Pickup Truck P-12", "Vacuum Truck", "Crew Bus", "Forklift", "Light Vehicle"],
        "potential_consequences": ["Fatal traffic accident", "Severe whiplash", "Crush injuries to pedestrian"],
        "precursors": ["high speed", "distracted driving", "poor road conditions", "heavy vehicle interaction"],
        "sif_templates": [
            "While {activity} with {equipment}, the driver was observed {barrier_failure}. This led to a near miss involving {hazard}.",
            "{equipment} was involved in a near miss during {activity}. Investigation showed {barrier_failure}, drastically increasing the risk of {hazard}.",
            "A safety camera caught {equipment} engaged in {activity}. The footage revealed {barrier_failure}, which is a precursor to {hazard}."
        ],
        "non_sif_templates": [
            "The {equipment} was slightly dirty during {activity}.",
            "Driver doing {activity} in {equipment} parked slightly outside the designated lines.",
            "Radio in {equipment} was left on high volume after {activity}."
        ]
    }
}
