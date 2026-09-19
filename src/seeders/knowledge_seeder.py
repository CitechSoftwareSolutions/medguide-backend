"""Sample data used when the application starts in development.

Each item in ``SAMPLE_GUIDELINES`` is one clinical guideline document, written
in exactly the JSON shape the ``POST /api/v1/knowledge/import`` endpoint
accepts. To add knowledge, paste another document into the tuple as-is -- no
reshaping needed. Each document is flattened into one knowledge entry per
condition plus one overview entry for its shared workflow, red flags and best
practices.
"""

import logging
from typing import Any

from src.dto import ImportGuidelineRequest
from src.exceptions import DuplicateKnowledgeEntryError
from src.repositories import has_entries
from src.services import create_knowledge_entry
from src.services.guideline_import_service import to_knowledge_requests

logger = logging.getLogger(__name__)

SAMPLE_GUIDELINES: tuple[dict[str, Any], ...] = (
    {
        "document_metadata": {
            "title": "Guideline Based Management of Primary Tremors: A Clinical Flow Chart Guide",
            "category": "Clinical Guidelines",
            "target_audience": "Healthcare Professionals",
        },
        "workflow_steps": [
            "1. Initial Approach: Take detailed history (onset, duration, hand involvement, triggers), Comprehensive examination, Identify red flags",
            "2. Initial Evaluation: Characterize tremor (Rest, Postural, Action, Kinetic), Basic investigations (CBC, RBS, TSH, Renal/Liver/Electrolytes), Toxicology screen if indicated",
            "3. Diagnosis & Etiologic Classification: Determine tremor type, Identify underlying cause, Assess severity/impact (e.g., Fahn-Tolosa-Marin Tremor Rating Scale), Rule out secondary causes",
            "4. Management & Follow-up: Treat underlying cause, Symptomatic management, Non-pharmacologic measures, Follow-up regularly, Refer if refractory/atypical",
        ],
        "conditions_registry": [
            {
                "condition_name": "Essential Tremor (ET)",
                "clinical_presentation": {
                    "key_features": [
                        "Bilateral action tremor (postural > kinetic)",
                        "Usually in hands, may involve head, voice",
                        "Gradual onset",
                        "Family history common",
                        "Improves with small amount of alcohol",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Idiopathic (most common)",
                        "Family history",
                        "Exacerbated by stress, fatigue, caffeine",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Non-pharmacologic: Reassurance, Avoid caffeine & stimulants",
                        "Adequate sleep",
                        "Stress management",
                        "Occupational therapy",
                        "Weighted utensils",
                    ],
                    "specific_advanced": [
                        "First-line (Pharmacologic): Propranolol 40-240 mg/day (start low, titrate) OR Primidone 25-250 mg/day (start low, titrate)",
                        "Combination therapy if inadequate response",
                        "Topiramate, Gabapentin, Pregabalin",
                    ],
                    "follow_up_monitoring": [
                        "Review in 4-6 weeks",
                        "Assess tremor severity, ADL impact & side effects",
                        "Monitor HR & BP (if on beta-blockers)",
                        "Long-term follow-up",
                    ],
                },
            },
            {
                "condition_name": "Parkinsonian Tremor (Resting Tremor)",
                "clinical_presentation": {
                    "key_features": [
                        "Resting tremor (pill-rolling)",
                        "Asymmetric onset",
                        "Decreases with voluntary movement",
                        "Associated bradykinesia, rigidity, postural instability",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Parkinson's disease (common)",
                        "Atypical parkinsonism",
                        "Drugs (e.g., antipsychotics)",
                        "Vascular parkinsonism",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Confirm diagnosis",
                        "Evaluate for PD criteria",
                        "Review medications",
                        "Baseline counseling",
                    ],
                    "specific_advanced": [
                        "Levodopa/carbidopa (start low, titrate)",
                        "Dopamine agonists",
                        "MAO-B inhibitors / COMT inhibitors",
                        "Physiotherapy",
                        "Advanced/refractory: Deep brain stimulation (STN/GPi), Apomorphine / Duodopa",
                    ],
                    "follow_up_monitoring": [
                        "Review in 4-6 weeks",
                        "Monitor motor & non-motor symptoms",
                        "Watch for medication side effects",
                        "Regular neurologist follow-up",
                    ],
                },
            },
            {
                "condition_name": "Physiologic Tremor",
                "clinical_presentation": {
                    "key_features": [
                        "Low amplitude, high frequency",
                        "Present in all individuals",
                        "Enhanced by anxiety, fatigue, medications, caffeine, hyperthyroidism",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Anxiety / Stress",
                        "Fatigue",
                        "Caffeine, nicotine",
                        "Medications (beta-agonists, SSRIs, TCAs, lithium, valproate)",
                        "Hyperthyroidism",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Identify & remove triggers",
                        "Reassurance",
                        "Lifestyle modification",
                        "Treat underlying cause (e.g., hyperthyroidism)",
                    ],
                    "specific_advanced": [
                        "Usually no specific drug therapy required",
                        "If troublesome: Propranolol (10-40 mg PRN) for performance-related situations",
                    ],
                    "follow_up_monitoring": [
                        "Review if persistent or worsening",
                        "Monitor after treating underlying cause",
                    ],
                },
            },
            {
                "condition_name": "Enhanced Physiologic / Situational Tremor",
                "clinical_presentation": {
                    "key_features": [
                        "Action or postural tremor",
                        "Situational (e.g., public speaking, fine tasks)",
                        "Normal neuro exam",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Stress / Performance anxiety",
                        "Stimulants",
                        "Fatigue",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Reassurance",
                        "Relaxation techniques",
                        "Behavioral therapy / CBT",
                        "Skill training",
                    ],
                    "specific_advanced": [
                        "Propranolol 10-40 mg PRN",
                        "Short-term use only",
                    ],
                    "follow_up_monitoring": [
                        "Review as needed",
                        "Ensure no progression of tremor",
                    ],
                },
            },
            {
                "condition_name": "Cerebellar Tremor (Intentional)",
                "clinical_presentation": {
                    "key_features": [
                        "Action tremor (intention)",
                        "Dysmetria",
                        "Associated with ataxia, dysarthria, nystagmus",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Cerebellar disorders (degenerative, inflammatory, vascular, tumor, MS)",
                        "Alcohol",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Neurologic evaluation",
                        "MRI brain (if indicated)",
                        "Identify & treat underlying cause",
                    ],
                    "specific_advanced": [
                        "Treat underlying disorder",
                        "Clonazepam or Primidone may help",
                        "Occupational & physiotherapy",
                    ],
                    "follow_up_monitoring": [
                        "Regular neurologic follow-up",
                        "Monitor progression of ataxia & other signs",
                    ],
                },
            },
            {
                "condition_name": "Dystonic Tremor",
                "clinical_presentation": {
                    "key_features": [
                        "Occurs in body part with dystonia",
                        "Task-specific",
                        "Abnormal postures",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Dystonia (primary or secondary)",
                        "Medications",
                        "Wilson's disease",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Treat underlying dystonia",
                        "Consider botulinum toxin (in focal dystonia)",
                        "Physiotherapy",
                    ],
                    "specific_advanced": [
                        "Anticholinergics",
                        "Benzodiazepines",
                        "Botulinum toxin injections",
                        "Deep brain stimulation (in selected cases)",
                    ],
                    "follow_up_monitoring": [
                        "Review in 4-6 weeks",
                        "Assess response & side effects",
                    ],
                },
            },
            {
                "condition_name": "Drug-Induced Tremor",
                "clinical_presentation": {
                    "key_features": [
                        "Tremor after starting drug",
                        "Often bilateral action tremor",
                        "Timing correlates with drug exposure",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "beta-agonists, steroids",
                        "Lithium, valproate",
                        "SSRIs, SNRIs, TCAs",
                        "Calcineurin inhibitors / immunosuppressants",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Review medication history",
                        "Stop or reduce offending drug if possible",
                        "Consider alternative",
                    ],
                    "specific_advanced": [
                        "Propranolol (if symptomatic)",
                        "Treat underlying condition",
                    ],
                    "follow_up_monitoring": [
                        "Review in 2-4 weeks",
                        "Ensure symptom resolution after drug adjustment",
                    ],
                },
            },
            {
                "condition_name": "Psychogenic (Functional) Tremor",
                "clinical_presentation": {
                    "key_features": [
                        "Variable tremor",
                        "Distractible / entrainable",
                        "Inconsistent findings",
                        "Associated with psychological stress",
                    ]
                },
                "diagnostic_approach": {
                    "common_causes": [
                        "Functional neurologic disorder",
                        "Anxiety, depression, stress",
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Educate & reassure",
                        "Explain diagnosis",
                        "Avoid unnecessary tests",
                        "Address psychological factors",
                    ],
                    "specific_advanced": [
                        "Cognitive behavioral therapy",
                        "Physiotherapy",
                        "Psychiatric evaluation (if needed)",
                    ],
                    "follow_up_monitoring": [
                        "Regular follow-up",
                        "Monitor for improvement",
                        "Encourage active rehabilitation",
                    ],
                },
            },
        ],
        "global_red_flags": [
            "Acute onset tremor",
            "Unilateral resting tremor with rapid progression",
            "Tremor with weakness, numbness, vision change, ataxia",
            "Tremor with altered consciousness",
            "Suspected drug toxicity",
            "Severe disability impacting ADLs",
            "Suspected cerebellar lesion",
            "Tremor in child or adolescent",
            "Signs of Wilson's disease (young age, liver disease)",
        ],
        "clinical_best_practices": [
            "Identify tremor type accurately",
            "Treat underlying cause whenever possible",
            "Start low, go slow with medications",
            "Individualize treatment based on severity & impact",
            "Monitor side effects and adherence",
            "Multidisciplinary approach improves outcomes",
        ],
        "disclaimer": "This chart is a guide only. Clinical judgment is essential. Early recognition and appropriate management improve quality of life and functional outcomes.",
    },
    {
    "document_metadata": {
        "title": "Guideline Based Management of Abdominal Discomfort in Elderly: A Clinical Flow Chart Guide",
        "category": "Clinical Guidelines",
        "target_audience": "Healthcare Professionals"
    },
    "workflow_steps": [
        "1. Initial Approach: Assess ABCs & stability, Detailed history, Medication review, Physical examination, Identify Red Flags",
        "2. Initial Evaluation: Basic labs (CBC, U&E, LFT, CRP), Consider ECG, Abdominal ultrasound, Stool occult blood, Assess hydration/nutrition",
        "3. Diagnosis & Risk Stratification: Use clinical features, tests & imaging; Stratify severity; Identify complications; Consider co-morbidities & frailty",
        "4. Management & Follow-up: Start condition-specific management, Reassess response (24-72 hrs), Adjust therapy, Plan follow-up, Refer if needed"
    ],
    "conditions_registry": [
        {
            "condition_name": "GORD (Gastro-oesophageal Reflux Disease)",
            "clinical_presentation": {
                "key_features": [
                    "Heartburn, regurgitation",
                    "Worse at night or after meals",
                    "Atypical symptoms: cough, hoarseness, asthma",
                    "Consider drug-induced (nitrates, CCBs)",
                    "Alarm symptoms: weight loss, dysphagia"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Clinical diagnosis",
                    "Upper GI endoscopy if alarm symptoms or long-standing",
                    "pH monitoring (if refractory)"
                ]
            },
            "management_plan": {
                "initial_conservative": [
                    "Lifestyle: weight reduction, elevate head end, avoid late meals, reduce trigger foods",
                    "Trial PPI once daily (e.g., Pantoprazole 40 mg OD)",
                    "Antacids / alginates PRN"
                ],
                "specific_advanced": [
                    "Optimize PPI (BD dosing if needed)",
                    "Add H2 blocker at night if nocturnal symptoms",
                    "Evaluate & stop offending drugs",
                    "Endoscopic evaluation if alarm or refractory",
                    "Consider surgery (fundoplication) in fit patients with refractory disease"
                ],
                "follow_up_monitoring": [
                    "Review in 4-8 weeks",
                    "Monitor symptom control",
                    "Watch for B12 deficiency, fractures (long-term PPI)",
                    "Repeat endoscopy if symptoms persist or worsen"
                ]
            }
        },
        {
            "condition_name": "Peptic Ulcer Disease (PUD)",
            "clinical_presentation": {
                "key_features": [
                    "Epigastric pain, burning",
                    "Pain related to meals (DU: pain after, GU: pain with food)",
                    "NSAID use common in elderly",
                    "Alarm: bleeding, anemia, weight loss, vomiting"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Upper GI endoscopy (gold standard)",
                    "H. pylori test (Urea breath test / stool antigen / biopsy)"
                ]
            },
            "management_plan": {
                "initial_conservative": [
                    "Stop NSAIDs",
                    "Start PPI (e.g., Pantoprazole 40 mg BD)",
                    "Test & treat H. pylori if positive",
                    "Sucralfate if needed"
                ],
                "specific_advanced": [
                    "H. pylori eradication: PPI + Clarithromycin + Amoxicillin (or Metronidazole) for 14 days",
                    "Continue PPI 4-8 weeks (DU), 8-12 weeks (GU)",
                    "Endoscopic therapy if bleeding",
                    "Surgery if perforation, obstruction, recurrent bleeding"
                ],
                "follow_up_monitoring": [
                    "Confirm H. pylori eradication",
                    "Monitor for anemia",
                    "Avoid NSAIDs (use alternatives)",
                    "Repeat endoscopy if non-healing ulcer or alarm"
                ]
            }
        },
        {
            "condition_name": "Gall Bladder Diseases",
            "clinical_presentation": {
                "key_features": [
                    "RUQ pain, nausea, bloating",
                    "Pain after fatty meals",
                    "May have fever (cholecystitis)",
                    "Elderly may present atypically",
                    "Consider comorbid cardiac/pulmonary disease"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Ultrasound abdomen (first line)",
                    "LFT, CBC",
                    "Consider HIDA scan if biliary dyskinesia suspected"
                ]
            },
            "management_plan": {
                "initial_conservative": [
                    "Analgesia (Paracetamol)",
                    "Low-fat diet",
                    "Hydration",
                    "Treat infection if present (Antibiotics)"
                ],
                "specific_advanced": [
                    "Cholelithiasis (symptomatic): Laparoscopic cholecystectomy (preferred) if fit",
                    "Acute cholecystitis: IV antibiotics + supportive care",
                    "Percutaneous cholecystostomy if high surgical risk",
                    "ERCP if choledocholithiasis"
                ],
                "follow_up_monitoring": [
                    "Monitor LFT, symptoms",
                    "Post-op follow-up",
                    "Watch for complications (jaundice, pancreatitis)",
                    "Dietary advice"
                ]
            }
        },
        {
            "condition_name": "Gastric Carcinoma",
            "clinical_presentation": {
                "key_features": [
                    "Early satiety, bloating",
                    "Weight loss, anorexia",
                    "Epigastric discomfort",
                    "Anemia, occult bleeding",
                    "More common in >70 yrs",
                    "Alarm symptoms common"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Upper GI endoscopy with biopsy",
                    "CT abdomen (staging)",
                    "CBC, LFT, tumor markers (CEA, CA19-9)"
                ]
            },
            "management_plan": {
                "initial_conservative": [
                    "Nutritional support",
                    "Treat anemia",
                    "Pain control",
                    "PPI for symptom relief"
                ],
                "specific_advanced": [
                    "Multidisciplinary approach",
                    "Surgery (if early & resectable)",
                    "Chemotherapy (e.g., FLOT regimen)",
                    "Radiotherapy (adjuvant / palliative)",
                    "Palliative care for advanced disease",
                    "Endoscopic stenting for obstruction"
                ],
                "follow_up_monitoring": [
                    "Regular oncology follow-up",
                    "Monitor nutrition & weight",
                    "Manage complications",
                    "Psychosocial support"
                ]
            }
        },
            {
                "condition_name": "Bowel Cancer (Colorectal Cancer)",
                "clinical_presentation": {
                    "key_features": [
                        "Change in bowel habits",
                        "Rectal bleeding",
                        "Abdominal discomfort",
                        "Iron deficiency anemia",
                        "Weight loss, fatigue",
                        "High prevalence in elderly"
                    ]
                },
                "diagnostic_approach": {
                    "investigations": [
                        "Colonoscopy with biopsy",
                        "CEA level",
                        "CT abdomen/pelvis (staging)"
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Nutritional assessment",
                        "Treat anemia",
                        "Pain relief",
                        "Bowel regimen"
                    ],
                    "specific_advanced": [
                        "Surgery (curative if early stage)",
                        "Chemotherapy (adjuvant / palliative)",
                        "Radiotherapy (rectal cancer)",
                        "Targeted therapy (selected cases)",
                        "Palliative stoma for obstruction"
                    ],
                    "follow_up_monitoring": [
                        "Surveillance colonoscopy (as per guidelines)",
                        "Monitor CEA",
                        "Manage side effects of treatment",
                        "Supportive care"
                    ]
                }
            },
            {
                "condition_name": "Diverticulitis",
                "clinical_presentation": {
                    "key_features": [
                        "LLQ pain",
                        "Fever, altered bowel habits",
                        "Nausea, bloating",
                        "May present with confusion in elderly",
                        "Consider perforation risk"
                    ]
                },
                "diagnostic_approach": {
                    "investigations": [
                        "Clinical diagnosis",
                        "CT abdomen/pelvis with contrast (gold standard)",
                        "CBC, CRP"
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "Mild (uncomplicated): Oral antibiotics (e.g., Amoxicillin-Clavulanate or Ciprofloxacin + Metronidazole)",
                        "Low residue diet",
                        "Hydration",
                        "Analgesia"
                    ],
                    "specific_advanced": [
                        "Severe / complicated: IV antibiotics",
                        "Abscess: percutaneous drainage",
                        "Peritonitis, fistula, obstruction: Surgical management",
                        "Elective surgery for recurrent episodes"
                    ],
                    "follow_up_monitoring": [
                        "Monitor symptoms",
                        "Colonoscopy after recovery (6-8 weeks)",
                        "High fibre diet after recovery",
                        "Prevent constipation"
                    ]
                }
            },
            {
                "condition_name": "Bowel Obstruction",
                "clinical_presentation": {
                    "key_features": [
                        "Colicky abdominal pain",
                        "Vomiting",
                        "Abdominal distension",
                        "Constipation / obstipation",
                        "High risk in elderly (adhesions, hernia, tumors)"
                    ]
                },
                "diagnostic_approach": {
                    "investigations": [
                        "Clinical diagnosis",
                        "X-ray abdomen (erect)",
                        "CT abdomen (if unclear)",
                        "Labs (U&E, CBC)"
                    ]
                },
                "management_plan": {
                    "initial_conservative": [
                        "NPO (nil by mouth)",
                        "IV fluids & electrolytes",
                        "Nasogastric decompression",
                        "Analgesia",
                        "Treat underlying cause if infection"
                    ],
                    "specific_advanced": [
                        "Conservative management: adhesive obstruction (no strangulation)",
                        "Surgery for bowel strangulation, ischemia, perforation, or failure of conservative therapy",
                        "Endoscopic stenting (selected cases)"
                    ],
                    "follow_up_monitoring": [
                        "Monitor vitals, abdominal signs",
                        "Repeat imaging if no improvement",
                        "Nutritional support",
                        "Prevent recurrence"
                    ]
                }
            }
        ],
        "global_red_flags": [
            "Significant weight loss",
            "Gastrointestinal bleeding (hematemesis / melena / hematochezia)",
            "Progressive dysphagia",
            "Persistent vomiting",
            "Severe abdominal pain with guarding or rigidity",
            "Iron deficiency anemia (unexplained)",
            "Obstructive symptoms",
            "Jaundice",
            "Altered mental status"
        ],
        "clinical_best_practices": [
            "Treat the cause, not just the symptom",
            "Consider geriatric syndromes (frailty, polypharmacy, malnutrition)",
            "Use lowest effective drug dose",
            "Avoid NSAIDs and unnecessary medications",
            "Ensure adequate hydration & nutrition",
            "Family/caregiver education and support",
            "Advance care planning in advanced disease"
        ],
        "disclaimer": "This chart is a guide only. Clinical judgment is essential. Refer to local guidelines and specialist input as needed."
    },
    {
    "document_metadata": {
        "title": "Evaluation of Headache in Adults: A Clinical Flow Chart Guide",
        "category": "Clinical Guidelines",
        "target_audience": "Healthcare Professionals"
    },
    "workflow_steps": [
        "1. Initial Approach: Assess ABCs, history, physical/neuro exam, identify Red Flags",
        "2. Classify Headache: Onset, duration, location, character, symptoms, triggers",
        "3. Targeted Evaluation: History, examination, investigations",
        "4. Diagnosis & Management: Treat underlying cause, provide symptomatic relief"
    ],
    "conditions_registry": [
        {
            "condition_name": "Migraines",
            "clinical_presentation": {
                "key_features": [
                    "Unilateral, throbbing",
                    "Moderate to severe",
                    "Nausea / vomiting",
                    "Photophobia / phonophobia",
                    "Aura (visual, sensory) may precede",
                    "Worse with activity"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Usually clinical",
                    "Neuroimaging if atypical features or red flags"
                ],
                "differential_red_flags": [
                    "Sudden severe onset",
                    "Neurological deficit",
                    "New onset > 50 yrs",
                    "Change in pattern",
                    "Fever / stiff neck"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "acute_abortive": [
                        "Ibuprofen 400-800 mg PO stat, then 400-800 mg q6-8h PRN (max 2400 mg/day) - up to 3-5 days",
                        "Sumatriptan 50-100 mg PO at onset; may repeat 50-100 mg after 2 h (max 200 mg/24 h) - up to 2 days/week",
                        "Metoclopramide 10 mg PO/IV q8h PRN for nausea - up to 3 days"
                    ],
                    "preventive": [
                        "Topiramate 25 mg HS titrate to 50-100 mg/day - >= 3 months",
                        "Propranolol 20 mg BID titrate to 40-80 mg BID - >= 3 months",
                        "Amitriptyline 10-25 mg HS titrate to 25-75 mg HS - >= 3 months"
                    ]
                },
                "non_pharmacological": [
                    "Trigger avoidance",
                    "Regular sleep & meals",
                    "Hydration",
                    "Stress management",
                    "Relaxation therapy",
                    "Biofeedback"
                ]
            }
        },
        {
            "condition_name": "Cluster Headache",
            "clinical_presentation": {
                "key_features": [
                    "Severe, unilateral (around eye)",
                    "Sharp / piercing",
                    "Lasts 15-180 min",
                    "Red eye, tearing, nasal congestion",
                    "Restlessness / agitation",
                    "Occurs in clusters"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Usually clinical",
                    "Imaging to rule out secondary causes if atypical"
                ],
                "differential_red_flags": [
                    "New onset > 40 yrs",
                    "Neurological deficit",
                    "Change in pattern",
                    "Persistent outside cluster periods"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "acute_abortive": [
                        "Sumatriptan 6 mg SC stat; may repeat after 1 h (max 12 mg/24 h) - up to 2 attacks/day",
                        "Zolmitriptan 5 mg nasal spray stat; may repeat after 2 h (max 10 mg/24 h)",
                        "Oxygen 100% via non-rebreather mask 12-15 L/min for 15-20 min - each attack"
                    ],
                    "preventive": [
                        "Verapamil 80 mg TID titrate to 240-960 mg/day - >= 6-8 weeks",
                        "Prednisone 60 mg daily taper over 2-3 weeks (short term bridge)"
                    ]
                },
                "non_pharmacological": [
                    "Avoid alcohol during cluster period",
                    "Regular sleep",
                    "Stress reduction",
                    "Identify & avoid triggers"
                ]
            }
        },
        {
            "condition_name": "Tension Headache",
            "clinical_presentation": {
                "key_features": [
                    "Bilateral, pressing / tightening",
                    "Mild to moderate",
                    "Not aggravated by routine activity",
                    "No nausea / vomiting",
                    "Stress / fatigue related"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Usually clinical",
                    "Imaging if red flags or atypical features"
                ],
                "differential_red_flags": [
                    "Sudden severe onset",
                    "Neurological deficit",
                    "New onset > 50 yrs",
                    "Fever / stiff neck"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "acute_abortive": [
                        "Paracetamol 500-1000 mg PO q6h PRN (max 3 g/day) - up to 7 days",
                        "Ibuprofen 200-400 mg PO q6-8h PRN (max 1200 mg/day) - up to 7 days"
                    ],
                    "preventive": [
                        "Amitriptyline 10-25 mg HS titrate to 25-75 mg HS - >= 2-3 months",
                        "Mirtazapine 15 mg HS - >= 2-3 months"
                    ]
                },
                "non_pharmacological": [
                    "Stress management",
                    "Ergonomic correction",
                    "Regular exercise",
                    "Adequate sleep",
                    "Relaxation therapy"
                ]
            }
        },
        {
            "condition_name": "Brain Tumour",
            "clinical_presentation": {
                "key_features": [
                    "Progressive headache",
                    "Worse in morning",
                    "Vomiting",
                    "Seizures",
                    "Focal neurological deficit",
                    "Personality / cognitive changes"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "MRI brain with contrast",
                    "CT scan (if MRI not available)",
                    "Neurological exam"
                ],
                "differential_red_flags": [
                    "Progressive worsening",
                    "Neurological deficit",
                    "New onset > 50 yrs",
                    "Seizures",
                    "Change in behavior / cognition"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "treatment": [
                        "Dexamethasone 4-16 mg/day PO/IV in divided doses - ongoing",
                        "Paracetamol 500-1000 mg q6h PRN",
                        "Opioids (e.g., Tramadol 50-100 mg q6h or Morphine 5-10 mg q4h PRN) - as needed for severe pain",
                        "Levetiracetam 500 mg BID (seizure prophylaxis) - ongoing if indicated"
                    ]
                },
                "non_pharmacological": [
                    "Supportive care",
                    "Seizure precautions",
                    "Cognitive support",
                    "Psychological support"
                ]
            }
        },
        {
            "condition_name": "Acute Angle Closure Glaucoma",
            "clinical_presentation": {
                "key_features": [
                    "Severe eye pain",
                    "Headache (often unilateral)",
                    "Blurred vision / halos around lights",
                    "Nausea / vomiting",
                    "Red eye"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Intraocular pressure measurement",
                    "Slit lamp exam",
                    "Ophthalmology consult"
                ],
                "differential_red_flags": [
                    "Sudden onset",
                    "Visual disturbance",
                    "Halos around lights",
                    "Nausea / vomiting"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "immediate_intervention": [
                        "Acetazolamide 500 mg IV stat, then 250 mg PO QID - 1-2 days",
                        "Timolol 0.5% eye drops 1 drop stat, then BID - 1-2 days",
                        "Pilocarpine 2% eye drops 1 drop stat, then q15 min for 1 hour, then QID - until controlled",
                        "Mannitol 1-2 g/kg IV over 30-60 min - single dose (if severe)"
                    ]
                },
                "non_pharmacological": [
                    "Avoid dark rooms",
                    "Immediate ophthalmology referral",
                    "Patient education"
                ]
            }
        },
        {
            "condition_name": "Drug Induced Headache",
            "clinical_presentation": {
                "key_features": [
                    "Temporal relation to drug initiation or dose change",
                    "Variable pattern",
                    "May mimic migraines or tension headaches"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Detailed drug history",
                    "Review recent changes",
                    "Imaging if red flags"
                ],
                "differential_red_flags": [
                    "New neurological symptoms",
                    "Severe or worsening pattern"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "treatment": [
                        "Stop / adjust offending drug",
                        "Paracetamol 500-1000 mg q6h PRN for pain - up to 5 days",
                        "If withdrawal headache (e.g., caffeine): Naproxen 250-500 mg BID - 3-5 days"
                    ]
                },
                "non_pharmacological": [
                    "Medication review",
                    "Avoid overuse of analgesics",
                    "Patient counseling"
                ]
            }
        },
        {
            "condition_name": "Thunderclap Headache",
            "clinical_presentation": {
                "key_features": [
                    "Sudden onset",
                    "Severe (worst ever)",
                    "Peaks in seconds",
                    "May be associated with neurological symptoms"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "CT brain (urgent)",
                    "Lumbar puncture",
                    "CT / MR angiography",
                    "Rule out SAH"
                ],
                "differential_red_flags": [
                    "Sudden explosive onset",
                    "Neck stiffness",
                    "Loss of consciousness",
                    "Neurological deficit"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "treatment": [
                        "Treat underlying cause urgently (e.g., aneurysm, bleed, RCVS)",
                        "Paracetamol 500-1000 mg q6h PRN",
                        "Opioids (e.g., Morphine 2-4 mg IV q4h PRN) - as needed"
                    ]
                },
                "non_pharmacological": [
                    "Emergency care",
                    "Monitor vitals",
                    "Neurosurgical / neurology consult"
                ]
            }
        },
        {
            "condition_name": "Chronic Sinusitis",
            "clinical_presentation": {
                "key_features": [
                    "Facial pain / pressure",
                    "Nasal congestion",
                    "Purulent nasal discharge",
                    "Postnasal drip",
                    "Worse when bending forward"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "Clinical exam",
                    "Nasal endoscopy",
                    "CT paranasal sinuses"
                ],
                "differential_red_flags": [
                    "Orbital / neurological complications",
                    "Persistent high fever",
                    "Severe facial swelling"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "treatment": [
                        "Amoxicillin-Clavulanate 625 mg PO TID - 10-14 days (or Doxycycline 100 mg PO BID if allergic)",
                        "Paracetamol 500-1000 mg q6h PRN",
                        "Intranasal corticosteroid (Fluticasone 50 mcg/nostril OD) - >= 4-6 weeks"
                    ]
                },
                "non_pharmacological": [
                    "Steam inhalation",
                    "Hydration",
                    "Allergy management",
                    "Avoid irritants",
                    "Saline nasal irrigation"
                ]
            }
        },
        {
            "condition_name": "Uncontrolled Hypertension",
            "clinical_presentation": {
                "key_features": [
                    "Dull, throbbing headache (often occipital)",
                    "Associated with dizziness",
                    "Blurred vision",
                    "Fatigue",
                    "BP usually markedly elevated"
                ]
            },
            "diagnostic_approach": {
                "investigations": [
                    "BP measurement",
                    "Fundoscopy",
                    "Renal function tests",
                    "ECG",
                    "Urinalysis"
                ],
                "differential_red_flags": [
                    "BP > 180/120 mmHg",
                    "Chest pain",
                    "Breathlessness",
                    "Neurological deficit",
                    "Papilledema"
                ]
            },
            "management_plan": {
                "pharmacological": {
                    "guidelines": "Gradual BP reduction (not > 25% in first hour unless hypertensive emergency)",
                    "treatment": [
                        "Amlodipine 5-10 mg PO OD - ongoing",
                        "Telmisartan 40-80 mg PO OD - ongoing",
                        "Hydrochlorothiazide 12.5-25 mg PO OD - ongoing",
                        "Clonidine 0.1-0.2 mg PO stat; may repeat after 30-60 min - short term",
                        "Paracetamol 500-1000 mg q6h PRN"
                    ]
                },
                "non_pharmacological": [
                    "Lifestyle modification",
                    "Salt restriction",
                    "Regular monitoring",
                    "Medication adherence"
                ]
            }
        }
    ],
    "global_red_flags": [
        "Sudden onset (thunderclap)",
        "Neurological deficit",
        "New onset > 50 years",
        "Fever / stiff neck",
        "Seizures",
        "Progressive worsening",
        "Change in pattern",
        "Persistent vomiting"
    ],
    "clinical_best_practices": [
        "Identify and treat the underlying cause.",
        "Provide symptomatic relief.",
        "Encourage healthy lifestyle (sleep, hydration, exercise).",
        "Use medications judiciously; avoid overuse.",
        "Consider multidisciplinary approach for chronic or complex cases.",
        "Regular follow-up and reassessment.",
        "Address psychosocial factors.",
        "Consider referral when needed."
    ],
    "disclaimer": "This chart is a guide only and does not replace individualized clinical judgment."
}
)


def seed_knowledge_entries() -> None:
    """Add a small, idempotent development dataset."""
    if has_entries():
        return

    for document in SAMPLE_GUIDELINES:
        payload = ImportGuidelineRequest.model_validate(document)
        for request in to_knowledge_requests(payload):
            try:
                create_knowledge_entry(request)
            except DuplicateKnowledgeEntryError:
                # Two pasted guidelines can legitimately share a condition;
                # the first one to define it wins rather than failing startup.
                logger.info("Skipping duplicate seed entry '%s'", request.title)
