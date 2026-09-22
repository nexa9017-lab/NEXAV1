# SIF Precursor Analytics Demo Script

**Total Time: ~5 Minutes**

### 0:00–0:30: Problem Statement
"Hello everyone. Today I'm demonstrating our new AI/NLP engine designed to detect Serious Injury and Fatality (SIF) precursors. 
Industrial safety platforms collect vast amounts of free-text reports—Unsafe Acts, Conditions, Near Misses, and Incidents. Reviewing these manually is slow and error-prone, which means we often miss the critical warning signs before a severe incident occurs. 
This prototype converts those raw, unstructured safety observations into structured safety intelligence, automatically detecting potential SIF precursors in real-time."

### 0:30–1:00: Architecture
"Let's briefly look at the architecture. When a report enters the system, it passes through a text preprocessing pipeline. We generate semantic representations using Sentence Embeddings, which are then evaluated by two machine learning models: a SIF Classifier and a Life-Saving Rule Classifier. 
Simultaneously, our hybrid Precursor Extractor pulls out specific activities, equipment, hazards, and barrier failures. This enriched data powers the risk analytics and pattern mining you see on this dashboard, which is served by a robust FastAPI backend."

### 1:00–1:45: Executive Dashboard
"Here on the Executive Overview page, we get an immediate sense of the organization's safety posture. We can see the total number of reports and how many were flagged as SIF precursors. 
The charts give us the SIF density—the ratio of SIF precursors to total reports—broken down by site and by the specific Life-Saving Rules violated. You'll notice that 'Energy Isolation' and 'Line of Fire' are our most frequent critical risks."

### 1:45–2:30: Recurring Pattern Analytics
"Now let's dive into Precursor Patterns. This is where the analytics engine mines historical data to find combinations of factors that frequently lead to high-risk situations. 
We generate a Prototype Prioritization Score for each pattern based on its frequency, its SIF density, and how many different sites it affects. This isn't just counting words; it's identifying systemic vulnerabilities—like 'Working at height without fall protection during scaffolding work'. We can select any pattern to see the exact locations and read the underlying reports driving that score."

### 2:30–3:15: Site & LSR Analysis
"Moving to Site Risk Analysis, we can rank facilities not just by the sheer volume of incidents, but by their SIF Precursor Density. This helps leadership allocate safety resources to the locations with the highest intrinsic risk, rather than just the largest headcount.
Similarly, under Life-Saving Rule Analysis, we can track exactly which critical rules are being compromised and what activities (like Mechanical Lifting or Hot Work) are most associated with those breaches."

### 3:15–4:10: Analyse New Report
"Let's see this in action with a live example. I'll paste a scenario: *'Technician began opening the pump discharge flange before isolation was verified. Residual pressure remained in the line.'*
When I hit 'Analyse', the FastAPI backend processes this in milliseconds. It correctly flags it as a SIF-Potential with a high probability. It identifies the primary Life-Saving Rule as 'Energy Isolation'. 
It also extracts the critical entities: the equipment (flange), the hazard (residual pressure), and tags it as a LOTO violation."

### 4:10–4:40: Explain Model Outputs & Evidence
"Notice the 'Evidence' section below. The system doesn't just give us a black-box prediction. It shows exactly which phrases in the text triggered the extraction. This traceability is critical for building trust with HSE professionals who will use this as a decision-support tool."

### 4:40–5:00: Limitations + Real-Data Migration
"Finally, it's important to note the current limitations. This MVP was trained and evaluated on synthetic data to prove the architecture. The Prototype Prioritization Score is a demonstrative metric, not a validated industry standard.
To move this to production, the next step is migrating to real, anonymized enterprise data. We will need HSE experts to annotate a ground-truth dataset, retrain the embedding classifiers, and calibrate the decision thresholds to match the organization's specific risk tolerance. 
Thank you."
