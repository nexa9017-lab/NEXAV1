import pandas as pd
from pathlib import Path
import uuid
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import CHALLENGE_FILE
from src.preprocessing.text_preprocessor import SafetyTextPreprocessor

def generate_challenge_set():
    reports = [
        {"description": "Started cracking open the pump discharge flange after closing the valve, but the line had not been proven dead. Fluid started weeping.", "sif_label": 1, "lsr_tags": "Energy Isolation"},
        {"description": "Welder began work after the gas test validity period had expired.", "sif_label": 1, "lsr_tags": "Hot Work|Work Authorization"},
        {"description": "Technician noticed the toe board was loose during inspection; nobody was working below and no object fell.", "sif_label": 0, "lsr_tags": "Work at Height"},
        {"description": "Found some rust on the side of the container.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "Forklift moving in reverse nearly backed into a crew member. The reversing alarm was disabled.", "sif_label": 1, "lsr_tags": "Driving|Line of Fire"},
        {"description": "Worker walked under a suspended 10-ton load while the rigger was distracted.", "sif_label": 1, "lsr_tags": "Safe Mechanical Lifting|Line of Fire"},
        {"description": "Opened the hatch to the mud pit without sniffing for H2S. Alarms sounded immediately.", "sif_label": 1, "lsr_tags": "Confined Space"},
        {"description": "Bypassed the high level trip on the separator because it was acting up. Vessel almost overflowed.", "sif_label": 1, "lsr_tags": "Bypassing Safety Controls"},
        {"description": "Driving at 90kmh in a 40kmh zone during heavy rain.", "sif_label": 1, "lsr_tags": "Driving"},
        {"description": "Crew began cutting the pipe spool, unaware that SIMOPS was occurring directly above them.", "sif_label": 1, "lsr_tags": "Work Authorization|Hot Work"},
        {"description": "Noticed that the lightbulb in the corridor was flickering.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "IP tripped on an uneven floor mat in the office and twisted an ankle.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "Contractor tried to enter the confined space before the attendant arrived. Stopped by site supervisor.", "sif_label": 1, "lsr_tags": "Confined Space"},
        {"description": "Removed the grating on deck 2 without putting up hard barricades.", "sif_label": 1, "lsr_tags": "Work at Height|Line of Fire"},
        {"description": "Hydraulic torque wrench pressure hose snapped. Worker was out of the line of fire, so nobody was struck.", "sif_label": 0, "lsr_tags": "Line of Fire"}, # Actually, whip from parted cable is a hazard. But nobody was hit, still high potential SIF. Wait, SIF potential? Yes. Let's make it SIF=1.
        {"description": "Hydraulic torque wrench pressure hose snapped. Worker was out of the line of fire, so nobody was struck.", "sif_label": 1, "lsr_tags": "Line of Fire"},
        {"description": "Mechanic used a scaffold pole as a cheater bar on a wrench.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "LOTO padlock was found cut off and lying on the floor. Investigation ongoing.", "sif_label": 1, "lsr_tags": "Energy Isolation"},
        {"description": "Operator disabled the gas detector to avoid nuisance alarms during the turnaround.", "sif_label": 1, "lsr_tags": "Bypassing Safety Controls"},
        {"description": "Scaffold tag was red, but the painters climbed it anyway to finish the final coat.", "sif_label": 1, "lsr_tags": "Work at Height"},
        {"description": "No PTW found for the guys excavating near the live gas line.", "sif_label": 1, "lsr_tags": "Work Authorization"},
        {"description": "Using an uncertified chain block to hoist the engine block.", "sif_label": 1, "lsr_tags": "Safe Mechanical Lifting"},
        {"description": "The crane outriggers sank into the mud because mats were not used. Crane tilted dangerously.", "sif_label": 1, "lsr_tags": "Safe Mechanical Lifting"},
        {"description": "Driver was using his mobile phone and drifted into the oncoming lane.", "sif_label": 1, "lsr_tags": "Driving"},
        {"description": "Smelled gas in the analyzer shelter. The ventilation fan had failed.", "sif_label": 1, "lsr_tags": "Confined Space"},
        {"description": "Dropped a hammer from the 4th deck. It landed near the walkway, missed people by a few feet.", "sif_label": 1, "lsr_tags": "Work at Height|Line of Fire"},
        {"description": "Sparks from grinding fell onto an oily rag, starting a small fire. Extinguished quickly.", "sif_label": 1, "lsr_tags": "Hot Work"},
        {"description": "Opened the bleeder valve, it was plugged with hydrate. Tried to clear it with wire.", "sif_label": 1, "lsr_tags": "Energy Isolation|Line of Fire"},
        {"description": "Minor scratch on the hand while turning a stiff valve.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "Dust blew into IP's eye because safety glasses were resting on the helmet.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "Puddle of water in the locker room.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "Worker found sleeping in the control room during night shift.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "The label on the chemical drum was faded.", "sif_label": 0, "lsr_tags": "None"},
        {"description": "Lanyard was frayed slightly, but still held weight during inspection. Discarded.", "sif_label": 0, "lsr_tags": "Work at Height"},
        {"description": "Tension wire snapped while pulling the stuck pipe. Whip action struck the driller's console.", "sif_label": 1, "lsr_tags": "Line of Fire"},
        {"description": "Confined space entry permit was signed, but the gas tester had left the site.", "sif_label": 1, "lsr_tags": "Confined Space|Work Authorization"},
        {"description": "Bypassed the ESD logic to keep the plant running during a sensor fault.", "sif_label": 1, "lsr_tags": "Bypassing Safety Controls"},
        {"description": "Driver didn't wear a seatbelt while moving the truck in the yard.", "sif_label": 1, "lsr_tags": "Driving"},
        {"description": "Hot work was happening on the grating right above the open hydrocarbon drain.", "sif_label": 1, "lsr_tags": "Hot Work"},
        {"description": "Started the compressor while the mechanic was still bolting the guard back on.", "sif_label": 1, "lsr_tags": "Energy Isolation"}
    ]
    
    # We only need the core columns to test the models, but we'll add dummy values for the rest 
    # so the datasets match the expected schema if passed to older functions.
    
    df = pd.DataFrame(reports)
    df['report_id'] = [f"CHAL-{uuid.uuid4().hex[:6].upper()}" for _ in range(len(df))]
    df['date'] = "2024-01-01"
    df['site'] = "Challenge Site"
    df['location'] = "Any"
    df['report_type'] = "Observation"
    df['activity'] = "Various"
    df['sif_potential'] = df['sif_label'] * 0.9  # Dummy
    df['primary_lsr'] = df['lsr_tags'].apply(lambda x: x.split('|')[0] if x != "None" else "None")
    df['hazard'] = "None"
    df['barrier_failure'] = "None"
    df['equipment'] = "None"
    df['potential_consequence'] = "None"
    df['actual_consequence'] = "None"
    df['precursor_tags'] = ""
    df['severity'] = "Low"
    
    preprocessor = SafetyTextPreprocessor()
    df['description_clean'] = preprocessor.transform_series(df['description'])
    
    df.to_csv(CHALLENGE_FILE, index=False)
    print(f"Generated challenge set at {CHALLENGE_FILE} with {len(df)} reports.")

if __name__ == "__main__":
    generate_challenge_set()
