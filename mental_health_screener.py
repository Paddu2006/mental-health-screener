# ============================================
# Mental Health Screener
# Author: Padma Shree
# Phase 3 - Project 5
# Tools: PHQ-9, GAD-7, Stress, ML
# ============================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# QUESTIONNAIRES
# ─────────────────────────────────────────────

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or that you are a failure",
    "Trouble concentrating on things such as reading or watching TV",
    "Moving or speaking slowly, or being fidgety or restless",
    "Thoughts that you would be better off dead or hurting yourself"
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

STRESS_QUESTIONS = [
    "Been upset because of something that happened unexpectedly",
    "Felt unable to control important things in your life",
    "Felt nervous and stressed",
    "Felt confident about your ability to handle problems",
    "Felt that things were going your way",
    "Found that you could not cope with all things you had to do",
    "Been able to control irritations in your life",
    "Felt that you were on top of things",
    "Been angered because of things outside your control",
    "Felt difficulties were piling up so high you could not overcome"
]

RESPONSE_OPTIONS = {
    "phq9"  : ["0 - Not at all", "1 - Several days",
                "2 - More than half the days", "3 - Nearly every day"],
    "gad7"  : ["0 - Not at all", "1 - Several days",
                "2 - More than half the days", "3 - Nearly every day"],
    "stress": ["0 - Never", "1 - Almost never", "2 - Sometimes",
                "3 - Fairly often", "4 - Very often"]
}

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

def score_phq9(responses):
    """Score PHQ-9 depression screening."""
    score = sum(responses)
    if score <= 4:
        severity = "None/Minimal"
        color    = "#4CAF50"
    elif score <= 9:
        severity = "Mild"
        color    = "#8BC34A"
    elif score <= 14:
        severity = "Moderate"
        color    = "#FF9800"
    elif score <= 19:
        severity = "Moderately Severe"
        color    = "#FF5722"
    else:
        severity = "Severe"
        color    = "#E91E63"
    return score, severity, color

def score_gad7(responses):
    """Score GAD-7 anxiety screening."""
    score = sum(responses)
    if score <= 4:
        severity = "None/Minimal"
        color    = "#4CAF50"
    elif score <= 9:
        severity = "Mild"
        color    = "#8BC34A"
    elif score <= 14:
        severity = "Moderate"
        color    = "#FF9800"
    else:
        severity = "Severe"
        color    = "#E91E63"
    return score, severity, color

def score_stress(responses):
    """Score Perceived Stress Scale."""
    # Reverse score positive items (4,5,7,8 — 0-indexed: 3,4,6,7)
    positive_items = [3, 4, 6, 7]
    adjusted = []
    for i, r in enumerate(responses):
        if i in positive_items:
            adjusted.append(4 - r)
        else:
            adjusted.append(r)
    score = sum(adjusted)
    if score <= 13:
        severity = "Low Stress"
        color    = "#4CAF50"
    elif score <= 26:
        severity = "Moderate Stress"
        color    = "#FF9800"
    else:
        severity = "High Stress"
        color    = "#E91E63"
    return score, severity, color

# ─────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────

def simulate_training_data(n=500):
    """Simulate mental health screening dataset."""
    np.random.seed(42)
    data = []
    labels = []
    categories = ["Healthy", "Mild Issues", "Moderate Issues", "Severe Issues"]

    for _ in range(n // 4):
        # Healthy
        phq = [np.random.randint(0, 2) for _ in range(9)]
        gad = [np.random.randint(0, 2) for _ in range(7)]
        stress = [np.random.randint(0, 2) for _ in range(10)]
        data.append(phq + gad + stress)
        labels.append("Healthy")

        # Mild
        phq = [np.random.randint(0, 3) for _ in range(9)]
        gad = [np.random.randint(0, 3) for _ in range(7)]
        stress = [np.random.randint(1, 3) for _ in range(10)]
        data.append(phq + gad + stress)
        labels.append("Mild Issues")

        # Moderate
        phq = [np.random.randint(1, 3) for _ in range(9)]
        gad = [np.random.randint(1, 3) for _ in range(7)]
        stress = [np.random.randint(2, 4) for _ in range(10)]
        data.append(phq + gad + stress)
        labels.append("Moderate Issues")

        # Severe
        phq = [np.random.randint(2, 4) for _ in range(9)]
        gad = [np.random.randint(2, 4) for _ in range(7)]
        stress = [np.random.randint(3, 5) for _ in range(10)]
        data.append(phq + gad + stress)
        labels.append("Severe Issues")

    return np.array(data), np.array(labels)

def train_ml_model():
    """Train ML model for mental health classification."""
    X, y = simulate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model    = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# ─────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────

RECOMMENDATIONS = {
    "None/Minimal": [
        "Continue maintaining your current healthy habits",
        "Regular exercise helps maintain good mental health",
        "Practice mindfulness and gratitude daily",
        "Maintain strong social connections",
        "Ensure 7-8 hours of quality sleep"
    ],
    "Mild": [
        "Consider speaking with a trusted friend or counselor",
        "Practice relaxation techniques like deep breathing",
        "Regular physical activity can help improve mood",
        "Limit alcohol and caffeine consumption",
        "Try journaling to process your thoughts",
        "Monitor your symptoms over the next 2 weeks"
    ],
    "Moderate": [
        "Strongly consider consulting a mental health professional",
        "Talk to your primary care doctor about your symptoms",
        "Cognitive Behavioral Therapy (CBT) can be very effective",
        "Establish a consistent daily routine",
        "Reach out to support groups or helplines",
        "Avoid isolating yourself from friends and family"
    ],
    "Moderately Severe": [
        "Please consult a mental health professional soon",
        "Consider therapy combined with medical evaluation",
        "Contact iCall helpline: 9152987821",
        "Contact Vandrevala Foundation: 1860-2662-345",
        "Inform a trusted family member about how you feel",
        "Avoid making major life decisions while feeling this way"
    ],
    "Severe": [
        "Please seek professional help immediately",
        "Contact iCall: 9152987821 (Mon-Sat 8am-10pm)",
        "Contact NIMHANS helpline: 080-46110007",
        "Contact Vandrevala Foundation 24x7: 1860-2662-345",
        "Go to your nearest hospital if you feel unsafe",
        "Do not be alone — reach out to someone you trust now"
    ],
    "Low Stress": [
        "Great stress management! Keep it up",
        "Continue your current stress management practices",
        "Share your coping strategies with others"
    ],
    "Moderate Stress": [
        "Practice progressive muscle relaxation",
        "Try yoga or meditation",
        "Take regular breaks during work",
        "Prioritize tasks and learn to say no",
        "Spend time in nature"
    ],
    "High Stress": [
        "Consider speaking with a stress management counselor",
        "Practice daily mindfulness meditation",
        "Exercise regularly — even a 20-minute walk helps",
        "Reduce caffeine and screen time",
        "Talk to someone you trust about your stressors"
    ]
}

# ─────────────────────────────────────────────
# ADMINISTER QUESTIONNAIRE
# ─────────────────────────────────────────────

def administer_questionnaire(questions, options, title):
    """Administer a questionnaire and collect responses."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print("  Please answer each question honestly.")
    print("  Your responses are confidential.\n")
    print("  Response options:")
    for opt in options:
        print(f"    {opt}")
    print()

    responses = []
    for i, question in enumerate(questions, 1):
        while True:
            try:
                print(f"  Q{i}. {question}")
                ans = int(input(f"  Your answer (0-{len(options)-1}): ").strip())
                if 0 <= ans <= len(options) - 1:
                    responses.append(ans)
                    break
                else:
                    print(f"  Please enter a number between 0 and {len(options)-1}")
            except ValueError:
                print("  Please enter a valid number!")
    return responses

# ─────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────

def visualize_profile(phq_score, gad_score, stress_score,
                      phq_sev, gad_sev, stress_sev,
                      phq_responses, gad_responses,
                      stress_responses, name="User"):
    """Generate mental health profile visualization."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Mental Health Profile — {name}",
                fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Chart 1 — Overall scores gauge
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ["Depression\n(PHQ-9)", "Anxiety\n(GAD-7)", "Stress\n(PSS-10)"]
    scores     = [phq_score, gad_score, stress_score]
    max_scores = [27, 21, 40]
    pcts       = [s/m*100 for s, m in zip(scores, max_scores)]
    colors1    = ["#E91E63" if p > 60 else "#FF9800" if p > 35 else "#4CAF50"
                 for p in pcts]
    bars = ax1.bar(categories, pcts, color=colors1, edgecolor="black")
    for bar, score, max_s in zip(bars, scores, max_scores):
        ax1.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{score}/{max_s}",
                ha="center", fontweight="bold", fontsize=10)
    ax1.set_title("Overall Scores", fontweight="bold")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 110)
    ax1.axhline(y=60, color="red", linestyle="--", alpha=0.5, label="High risk")
    ax1.axhline(y=35, color="orange", linestyle="--", alpha=0.5, label="Moderate risk")
    ax1.legend(fontsize=7)

    # Chart 2 — PHQ-9 item responses
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = ["#E91E63" if r >= 2 else "#FF9800" if r == 1 else "#4CAF50"
               for r in phq_responses]
    bars2   = ax2.bar([f"Q{i+1}" for i in range(len(phq_responses))],
                     phq_responses, color=colors2, edgecolor="black")
    ax2.set_title(f"PHQ-9 Item Scores (Total: {phq_score})",
                 fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 3.5)

    # Chart 3 — GAD-7 item responses
    ax3 = fig.add_subplot(gs[0, 2])
    colors3 = ["#E91E63" if r >= 2 else "#FF9800" if r == 1 else "#4CAF50"
               for r in gad_responses]
    bars3   = ax3.bar([f"Q{i+1}" for i in range(len(gad_responses))],
                     gad_responses, color=colors3, edgecolor="black")
    ax3.set_title(f"GAD-7 Item Scores (Total: {gad_score})",
                 fontweight="bold")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 3.5)

    # Chart 4 — Severity radar/summary
    ax4 = fig.add_subplot(gs[1, 0])
    sev_data = {
        "Depression"  : phq_score / 27 * 100,
        "Anxiety"     : gad_score / 21 * 100,
        "Stress"      : stress_score / 40 * 100,
        "Overall Risk": (phq_score/27 + gad_score/21 + stress_score/40) / 3 * 100
    }
    bar_colors = ["#E91E63" if v > 60 else "#FF9800" if v > 35 else "#4CAF50"
                 for v in sev_data.values()]
    bars4 = ax4.barh(list(sev_data.keys()), list(sev_data.values()),
                    color=bar_colors, edgecolor="black")
    for bar, val in zip(bars4, sev_data.values()):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax4.axvline(x=60, color="red", linestyle="--", alpha=0.5)
    ax4.set_title("Risk Profile (%)", fontweight="bold")
    ax4.set_xlabel("Risk Level (%)")
    ax4.set_xlim(0, 110)

    # Chart 5 — Stress item responses
    ax5 = fig.add_subplot(gs[1, 1])
    colors5 = ["#E91E63" if r >= 3 else "#FF9800" if r >= 2 else "#4CAF50"
               for r in stress_responses]
    ax5.bar([f"Q{i+1}" for i in range(len(stress_responses))],
           stress_responses, color=colors5, edgecolor="black")
    ax5.set_title(f"Stress Item Scores (Total: {stress_score})",
                 fontweight="bold")
    ax5.set_ylabel("Score")
    ax5.set_ylim(0, 4.5)

    # Chart 6 — Severity summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary_text = f"""
SCREENING SUMMARY

Depression (PHQ-9)
Score    : {phq_score}/27
Severity : {phq_sev}

Anxiety (GAD-7)
Score    : {gad_score}/21
Severity : {gad_sev}

Stress (PSS-10)
Score    : {stress_score}/40
Severity : {stress_sev}

Date: {datetime.now().strftime('%d-%m-%Y')}
"""
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
    ax6.set_title("Summary", fontweight="bold")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"mental_health_profile_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   Profile saved as: {filename}")
    return filename

# ─────────────────────────────────────────────
# SAVE HISTORY
# ─────────────────────────────────────────────

def save_to_history(name, phq_score, phq_sev,
                    gad_score, gad_sev,
                    stress_score, stress_sev):
    """Save screening results to history file."""
    history_file = "screening_history.json"
    history      = []

    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    entry = {
        "name"        : name,
        "date"        : datetime.now().strftime("%d-%m-%Y %H:%M"),
        "phq_score"   : phq_score,
        "phq_severity": phq_sev,
        "gad_score"   : gad_score,
        "gad_severity": gad_sev,
        "stress_score": stress_score,
        "stress_severity": stress_sev
    }
    history.append(entry)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"   Results saved to history!")

def show_history():
    """Show screening history."""
    history_file = "screening_history.json"
    if not os.path.exists(history_file):
        print("   No screening history found!")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    print(f"\n{'='*65}")
    print("  SCREENING HISTORY")
    print(f"{'='*65}")
    print(f"  {'Date':<20} {'PHQ-9':>8} {'GAD-7':>8} {'Stress':>8} {'PHQ Severity'}")
    print("  " + "-"*60)
    for entry in history:
        print(f"  {entry['date']:<20} "
              f"{entry['phq_score']:>8} "
              f"{entry['gad_score']:>8} "
              f"{entry['stress_score']:>8} "
              f"{entry['phq_severity']}")
    print(f"{'='*65}")

# ─────────────────────────────────────────────
# GENERATE REPORT
# ─────────────────────────────────────────────

def generate_report(name, phq_score, phq_sev,
                    gad_score, gad_sev,
                    stress_score, stress_sev,
                    ml_prediction, ml_accuracy):
    """Generate confidential screening report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"mental_health_report_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write("="*60 + "\n")
        f.write("MENTAL HEALTH SCREENING REPORT\n")
        f.write("CONFIDENTIAL\n")
        f.write(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
        f.write("Author: Padma Shree | Phase 3 - Project 5\n")
        f.write("="*60 + "\n\n")

        f.write(f"Name     : {name}\n")
        f.write(f"Date     : {datetime.now().strftime('%d-%m-%Y')}\n\n")

        f.write("SCREENING RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"PHQ-9 Depression Score : {phq_score}/27 — {phq_sev}\n")
        f.write(f"GAD-7 Anxiety Score    : {gad_score}/21 — {gad_sev}\n")
        f.write(f"Stress Scale Score     : {stress_score}/40 — {stress_sev}\n")
        f.write(f"ML Classification      : {ml_prediction}\n")
        f.write(f"ML Model Accuracy      : {ml_accuracy*100:.1f}%\n\n")

        f.write("RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        all_recs = set()
        for sev in [phq_sev, gad_sev, stress_sev]:
            recs = RECOMMENDATIONS.get(sev, [])
            all_recs.update(recs)
        for rec in list(all_recs)[:8]:
            f.write(f"  - {rec}\n")

        f.write("\nIMPORTANT HELPLINES (India)\n")
        f.write("-"*40 + "\n")
        f.write("  iCall              : 9152987821\n")
        f.write("  NIMHANS            : 080-46110007\n")
        f.write("  Vandrevala 24x7    : 1860-2662-345\n")
        f.write("  AASRA              : 9820466627\n\n")

        f.write("="*60 + "\n")
        f.write("DISCLAIMER: This is a screening tool, not a diagnosis.\n")
        f.write("Please consult a qualified mental health professional.\n")
        f.write("="*60 + "\n")

    print(f"   Report saved: {filename}")
    return filename

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────

def display_results(phq_score, phq_sev, gad_score,
                    gad_sev, stress_score, stress_sev,
                    ml_prediction):
    """Display screening results."""
    print("\n" + "="*60)
    print("  YOUR MENTAL HEALTH SCREENING RESULTS")
    print("="*60)
    print(f"\n  Depression (PHQ-9):")
    print(f"    Score    : {phq_score}/27")
    print(f"    Severity : {phq_sev}")
    print(f"\n  Anxiety (GAD-7):")
    print(f"    Score    : {gad_score}/21")
    print(f"    Severity : {gad_sev}")
    print(f"\n  Stress Level:")
    print(f"    Score    : {stress_score}/40")
    print(f"    Severity : {stress_sev}")
    print(f"\n  ML Overall Assessment : {ml_prediction}")

    print("\n  Recommendations:")
    shown = set()
    for sev in [phq_sev, gad_sev, stress_sev]:
        recs = RECOMMENDATIONS.get(sev, [])
        for rec in recs[:2]:
            if rec not in shown:
                print(f"    - {rec}")
                shown.add(rec)

    print("\n  Mental Health Helplines (India):")
    print("    iCall           : 9152987821")
    print("    NIMHANS         : 080-46110007")
    print("    Vandrevala 24x7 : 1860-2662-345")
    print("="*60)
    print("\n  DISCLAIMER: This is a screening tool not a diagnosis.")
    print("  Please consult a qualified mental health professional.")
    print("="*60)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("    Mental Health Screener")
    print("    Author: Padma Shree | Phase 3 - Project 5")
    print("    Tools: PHQ-9 | GAD-7 | PSS-10 | ML")
    print("="*60)
    print("\n  This tool uses validated clinical screening tools.")
    print("  Your responses are confidential.")
    print("  This is NOT a diagnosis — please consult a professional.")

    print("\n  Training ML model...")
    model, accuracy = train_ml_model()
    print(f"  ML Model ready! Accuracy: {accuracy*100:.1f}%")

    while True:
        print("\n  Options:")
        print("  1. Start new screening")
        print("  2. View screening history")
        print("  0. Exit")

        choice = input("\n  Enter choice: ").strip()

        if choice == "1":
            name = input("\n  Enter your name (or leave blank): ").strip()
            if not name:
                name = "Anonymous"

            # PHQ-9
            phq_responses = administer_questionnaire(
                PHQ9_QUESTIONS,
                RESPONSE_OPTIONS["phq9"],
                "PHQ-9: DEPRESSION SCREENING"
            )
            phq_score, phq_sev, phq_color = score_phq9(phq_responses)

            # GAD-7
            gad_responses = administer_questionnaire(
                GAD7_QUESTIONS,
                RESPONSE_OPTIONS["gad7"],
                "GAD-7: ANXIETY SCREENING"
            )
            gad_score, gad_sev, gad_color = score_gad7(gad_responses)

            # Stress
            stress_responses = administer_questionnaire(
                STRESS_QUESTIONS,
                RESPONSE_OPTIONS["stress"],
                "PSS-10: STRESS SCREENING"
            )
            stress_score, stress_sev, stress_color = score_stress(stress_responses)

            # ML prediction
            all_responses = np.array(
                phq_responses + gad_responses + stress_responses
            ).reshape(1, -1)
            ml_prediction = model.predict(all_responses)[0]

            # Display results
            display_results(phq_score, phq_sev,
                           gad_score, gad_sev,
                           stress_score, stress_sev,
                           ml_prediction)

            # Visualize
            print("\n  Generating mental health profile...")
            visualize_profile(
                phq_score, gad_score, stress_score,
                phq_sev, gad_sev, stress_sev,
                phq_responses, gad_responses,
                stress_responses, name
            )

            # Save history
            save_to_history(name, phq_score, phq_sev,
                           gad_score, gad_sev,
                           stress_score, stress_sev)

            # Generate report
            print("  Generating confidential report...")
            generate_report(name, phq_score, phq_sev,
                           gad_score, gad_sev,
                           stress_score, stress_sev,
                           ml_prediction, accuracy)

        elif choice == "2":
            show_history()

        elif choice == "0":
            print("\n  Take care of yourself Paddu!")
            print("  Remember — seeking help is a sign of strength!")
            break
        else:
            print("  Invalid choice!")