import pandas as pd
import json
import time
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from src.extraction.precursor_extractor import SafetyPrecursorExtractor
from src.extraction.normalization import (
    normalize_activity, normalize_hazard, normalize_barrier_failure, 
    normalize_equipment, normalize_precursor_tag, normalize_list, CONSEQUENCE_TERMS_IN_GT
)

def evaluate_metrics(y_true, y_pred, average='micro'):
    # Flatten lists of lists
    all_true_labels = set([label for sublist in y_true for label in sublist])
    all_pred_labels = set([label for sublist in y_pred for label in sublist])
    all_labels = list(all_true_labels.union(all_pred_labels))
    
    if not all_labels:
        return 0.0, 0.0, 0.0, {}

    # Create binary matrices
    y_true_bin = [[1 if label in t else 0 for label in all_labels] for t in y_true]
    y_pred_bin = [[1 if label in p else 0 for label in all_labels] for p in y_pred]
    
    p, r, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average=average, zero_division=0)
    
    # Class-wise metrics
    p_c, r_c, f1_c, support = precision_recall_fscore_support(y_true_bin, y_pred_bin, average=None, zero_division=0)
    
    class_metrics = {}
    for i, label in enumerate(all_labels):
        # Count actual support in ground truth
        label_support = sum(1 for t in y_true if label in t)
        class_metrics[label] = {
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f1_c[i]),
            "support": label_support
        }
        
    return p, r, f1, class_metrics

def _extract_literal_values(dicts_list, key="value"):
    return [str(d.get(key, "")).strip().lower() for d in dicts_list if d.get(key)]

def _extract_literal_list(str_list):
    if pd.isna(str_list): return []
    try:
        if isinstance(str_list, str):
            l = json.loads(str_list.replace("'", '"'))
            return [str(x).strip().lower() for x in l]
        return [str(x).strip().lower() for x in str_list]
    except:
        return [str(str_list).strip().lower()]

def print_errors(name, errors_list, limit=10):
    print(f"\n--- {name.upper()} ERRORS ---")
    for i, err in enumerate(errors_list[:limit]):
        print(f"TEXT: {err['text']}")
        print(f"GT: {err['gt']}")
        print(f"PRED: {err['pred']}")
        if 'gt_norm' in err: print(f"GT NORM: {err['gt_norm']}")
        if 'pred_norm' in err: print(f"PRED NORM: {err['pred_norm']}")
        if 'match_method' in err: print(f"MATCH: {err['match_method']}")
        if 'sim' in err: print(f"SIM: {err['sim']}")
        print("-")

def main():
    print("Loading datasets...")
    test_df = pd.read_csv("data/processed/test_processed.csv")
    
    print("Initializing SafetyPrecursorExtractor (with Semantic caching)...")
    start_time = time.time()
    extractor = SafetyPrecursorExtractor()
    print(f"Initialization took {time.time() - start_time:.2f}s")
    
    sample_text = test_df['description_clean'].iloc[0]
    t0 = time.time()
    _ = extractor.extract(sample_text)
    t1 = time.time()
    print(f"\nTiming - 1 report: {t1 - t0:.4f}s")
    
    t0 = time.time()
    _ = extractor.extract_batch(test_df['description_clean'].head(100).tolist())
    t1 = time.time()
    print(f"Timing - 100 reports: {t1 - t0:.4f}s\n")
    
    print("Extracting entire test set...")
    predictions = extractor.extract_batch(test_df['description_clean'].tolist())
    
    metrics = {
        "literal": {"activity": {}, "equipment": {}, "hazard": {}, "barrier": {}},
        "normalized": {"activity": {}, "equipment": {}, "equipment_type": {}, "hazard": {}, "barrier": {}, "tags": {}}
    }
    
    y_true_lit_act, y_pred_lit_act = [], []
    y_true_norm_act, y_pred_norm_act = [], []
    y_true_lit_eq, y_pred_lit_eq = [], []
    y_true_norm_eq, y_pred_norm_eq = [], []
    y_true_type_eq, y_pred_type_eq = [], []
    y_true_lit_haz, y_pred_lit_haz = [], []
    y_true_norm_haz, y_pred_norm_haz = [], []
    y_true_lit_bar, y_pred_lit_bar = [], []
    y_true_norm_bar, y_pred_norm_bar = [], []
    y_true_norm_tag, y_pred_norm_tag = [], []
    
    hazard_fp, hazard_fn = [], []
    activity_fn, equipment_fn = [], []
    
    for i, row in test_df.iterrows():
        pred = predictions[i]
        text = row['description_clean']
        
        # Ground truths (Literal)
        gt_l_act = _extract_literal_list(row.get('activity', '[]'))
        gt_l_eq = _extract_literal_list(row.get('equipment', '[]'))
        gt_l_haz = _extract_literal_list(row.get('hazard', '[]'))
        gt_l_bar = _extract_literal_list(row.get('barrier_failure', '[]'))
        gt_l_tag = _extract_literal_list(row.get('precursor_tags', '[]'))
        
        # Predictions (Literal)
        pr_l_act = _extract_literal_values(pred['activities'])
        pr_l_eq = _extract_literal_values(pred['equipment'])
        pr_l_haz = _extract_literal_values(pred['hazards'])
        pr_l_bar = _extract_literal_values(pred['barrier_failures'])
        pr_l_tag = pred.get('precursor_tags', [])
        
        y_true_lit_act.append(gt_l_act)
        y_pred_lit_act.append(pr_l_act)
        y_true_lit_eq.append(gt_l_eq)
        y_pred_lit_eq.append(pr_l_eq)
        y_true_lit_haz.append(gt_l_haz)
        y_pred_lit_haz.append(pr_l_haz)
        y_true_lit_bar.append(gt_l_bar)
        y_pred_lit_bar.append(pr_l_bar)
        
        # Normalized
        gt_n_act = normalize_list(gt_l_act, normalize_activity)
        pr_n_act = normalize_list(pr_l_act, normalize_activity)
        y_true_norm_act.append(gt_n_act)
        y_pred_norm_act.append(pr_n_act)
        
        gt_n_eq = normalize_list(gt_l_eq, normalize_equipment)
        pr_n_eq = normalize_list(pr_l_eq, normalize_equipment)
        y_true_norm_eq.append(gt_n_eq)
        y_pred_norm_eq.append(pr_n_eq)
        
        # For equipment type, use the 'type' field from extractor
        pr_t_eq = normalize_list([e.get("type", "") for e in pred['equipment']], normalize_equipment)
        y_true_type_eq.append(gt_n_eq) # Ground truth normalized acts as type
        y_pred_type_eq.append(pr_t_eq)
        
        gt_n_haz = normalize_list(gt_l_haz, normalize_hazard)
        pr_n_haz = normalize_list(pr_l_haz, normalize_hazard)
        y_true_norm_haz.append(gt_n_haz)
        y_pred_norm_haz.append(pr_n_haz)
        
        gt_n_bar = normalize_list(gt_l_bar, normalize_barrier_failure)
        pr_n_bar = normalize_list(pr_l_bar, normalize_barrier_failure)
        y_true_norm_bar.append(gt_n_bar)
        y_pred_norm_bar.append(pr_n_bar)
        
        gt_n_tag = normalize_list(gt_l_tag, normalize_precursor_tag)
        pr_n_tag = normalize_list(pr_l_tag, normalize_precursor_tag)
        y_true_norm_tag.append(gt_n_tag)
        y_pred_norm_tag.append(pr_n_tag)
        
        # Error Collection
        # Hazard FP
        for h in pred['hazards']:
            h_norm = normalize_hazard(h['value'])
            if h_norm not in gt_n_haz and h_norm != "CONSEQUENCE_IGNORE":
                if len(hazard_fp) < 15:
                    hazard_fp.append({
                        "text": text, "gt": gt_l_haz, "pred": [h['value']],
                        "gt_norm": gt_n_haz, "pred_norm": [h_norm],
                        "match_method": h.get("match_method", "unknown"),
                        "sim": h.get("similarity_score", None)
                    })
        # Hazard FN
        for h in gt_n_haz:
            if h not in pr_n_haz and h != "CONSEQUENCE_IGNORE":
                if len(hazard_fn) < 15:
                    hazard_fn.append({
                        "text": text, "gt": gt_l_haz, "pred": [x['value'] for x in pred['hazards']],
                        "gt_norm": [h], "pred_norm": pr_n_haz
                    })
                    
        # Activity FN
        for a in gt_n_act:
            if a not in pr_n_act:
                if len(activity_fn) < 15:
                    activity_fn.append({
                        "text": text, "gt": gt_l_act, "pred": [x['value'] for x in pred['activities']],
                        "gt_norm": [a], "pred_norm": pr_n_act
                    })
                    
        # Equipment FN
        for e in gt_n_eq:
            if e not in pr_n_eq and e not in pr_t_eq:
                if len(equipment_fn) < 15:
                    equipment_fn.append({
                        "text": text, "gt": gt_l_eq, "pred": [x['value'] for x in pred['equipment']],
                        "gt_norm": [e], "pred_norm": pr_n_eq
                    })
        
    # Calculate Literal
    pl_a, rl_a, fl_a, _ = evaluate_metrics(y_true_lit_act, y_pred_lit_act)
    pl_e, rl_e, fl_e, _ = evaluate_metrics(y_true_lit_eq, y_pred_lit_eq)
    pl_h, rl_h, fl_h, _ = evaluate_metrics(y_true_lit_haz, y_pred_lit_haz)
    pl_b, rl_b, fl_b, _ = evaluate_metrics(y_true_lit_bar, y_pred_lit_bar)
    
    # Calculate Normalized
    pn_a, rn_a, fn_a, class_act = evaluate_metrics(y_true_norm_act, y_pred_norm_act)
    pn_e, rn_e, fn_e, _ = evaluate_metrics(y_true_norm_eq, y_pred_norm_eq)
    pt_e, rt_e, ft_e, _ = evaluate_metrics(y_true_type_eq, y_pred_type_eq)
    pn_h, rn_h, fn_h, class_haz = evaluate_metrics(y_true_norm_haz, y_pred_norm_haz)
    pn_b, rn_b, fn_b, _ = evaluate_metrics(y_true_norm_bar, y_pred_norm_bar)
    pn_t, rn_t, fn_t, _ = evaluate_metrics(y_true_norm_tag, y_pred_norm_tag)
    
    print("\n--- LITERAL TEST METRICS ---")
    print(f"ACTIVITY         | P: {pl_a:.3f} | R: {rl_a:.3f} | F1: {fl_a:.3f}")
    print(f"EQUIPMENT EXACT  | P: {pl_e:.3f} | R: {rl_e:.3f} | F1: {fl_e:.3f}")
    print(f"HAZARD           | P: {pl_h:.3f} | R: {rl_h:.3f} | F1: {fl_h:.3f}")
    print(f"BARRIER_FAILURE  | P: {pl_b:.3f} | R: {rl_b:.3f} | F1: {fl_b:.3f}")

    print("\n--- NORMALIZED TEST METRICS ---")
    print(f"ACTIVITY         | P: {pn_a:.3f} | R: {rn_a:.3f} | F1: {fn_a:.3f}")
    print(f"EQUIPMENT EXACT  | P: {pn_e:.3f} | R: {rn_e:.3f} | F1: {fn_e:.3f}")
    print(f"EQUIPMENT TYPE   | P: {pt_e:.3f} | R: {rt_e:.3f} | F1: {ft_e:.3f}")
    print(f"HAZARD           | P: {pn_h:.3f} | R: {rn_h:.3f} | F1: {fn_h:.3f}")
    print(f"BARRIER_FAILURE  | P: {pn_b:.3f} | R: {rn_b:.3f} | F1: {fn_b:.3f}")
    print(f"PRECURSOR_TAGS   | P: {pn_t:.3f} | R: {rn_t:.3f} | F1: {fn_t:.3f}")
    
    print("\n--- PER-CLASS HAZARD METRICS ---")
    for k, v in sorted(class_haz.items(), key=lambda x: x[1]['support'], reverse=True):
        if v['support'] > 0:
            print(f"{k.ljust(25)} | S: {v['support']:3d} | P: {v['precision']:.3f} | R: {v['recall']:.3f} | F1: {v['f1']:.3f}")
            
    print("\n--- PER-CLASS ACTIVITY METRICS ---")
    for k, v in sorted(class_act.items(), key=lambda x: x[1]['support'], reverse=True):
        if v['support'] > 0:
            print(f"{k.ljust(25)} | S: {v['support']:3d} | P: {v['precision']:.3f} | R: {v['recall']:.3f} | F1: {v['f1']:.3f}")

    print_errors("Hazard False Positives", hazard_fp, 10)
    print_errors("Hazard False Negatives", hazard_fn, 10)
    print_errors("Activity False Negatives", activity_fn, 10)
    print_errors("Equipment False Negatives", equipment_fn, 10)

    # Output PHASE_5_READY flag
    is_ready = True
    if fn_a < 0.70: is_ready = False
    if ft_e < 0.70: is_ready = False
    if fn_h < 0.60: is_ready = False
    if fn_b < 0.70: is_ready = False
    
    print(f"\nPHASE_5_READY = {str(is_ready).lower()}")
    
    # Save to disk
    full_metrics = {
        "literal": {
            "activity": {"p": pl_a, "r": rl_a, "f1": fl_a},
            "equipment": {"p": pl_e, "r": rl_e, "f1": fl_e},
            "hazard": {"p": pl_h, "r": rl_h, "f1": fl_h},
            "barrier_failure": {"p": pl_b, "r": rl_b, "f1": fl_b}
        },
        "normalized": {
            "activity": {"p": pn_a, "r": rn_a, "f1": fn_a},
            "equipment_exact": {"p": pn_e, "r": rn_e, "f1": fn_e},
            "equipment_type": {"p": pt_e, "r": rt_e, "f1": ft_e},
            "hazard": {"p": pn_h, "r": rn_h, "f1": fn_h},
            "barrier_failure": {"p": pn_b, "r": rn_b, "f1": fn_b},
            "precursor_tags": {"p": pn_t, "r": rn_t, "f1": fn_t}
        },
        "class_hazard": class_haz,
        "class_activity": class_act,
        "phase_5_ready": is_ready
    }
    with open("artifacts/reports/extraction_metrics_v3.json", "w") as f:
        json.dump(full_metrics, f, indent=4)
        
if __name__ == "__main__":
    main()
