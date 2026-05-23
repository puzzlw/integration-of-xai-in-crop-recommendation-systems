from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


def find_repo_root() -> Path:
    resolved_file_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [resolved_file_dir, cwd]
    candidates.extend(resolved_file_dir.parents)
    candidates.extend(cwd.parents)

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if (path / "model_outputs").exists():
            return path

    return cwd


BASE_DIR = find_repo_root()
OUTPUT_DIR = BASE_DIR / "model_outputs"
RESPONSES_PATH = BASE_DIR / "responses_collected.csv"

RECOMMENDATIONS_PATH = OUTPUT_DIR / "user_input_hybrid_recommendations.csv"
XAI_DETAILS_PATH = OUTPUT_DIR / "xai_explanation_details.csv"
XAI_SUMMARY_PATH = OUTPUT_DIR / "xai_best_technique_summary.csv"
REGION_DEFAULTS_PATH = OUTPUT_DIR / "s4a_region_proxy_summary.csv"
MARKET_FEATURES_PATH = OUTPUT_DIR / "market_price_features.csv"
CASE_STUDY_COUNTS_PATH = OUTPUT_DIR / "case_study_record_counts.csv"

CASE_STUDY_CROPS = {
    11: "Maize",
    12: "Paddy",
    31: "Beans",
    54: "Coffee",
    71: "Banana",
}

HYBRID_SUITABILITY_RULES = {
    11: {
        "soil_ph_mean": (5.8, 7.2, 4.8, 8.2),
        "temp_mean": (24.0, 32.0, 20.0, 36.0),
        "rainfall_total_2023_24": (500.0, 1200.0, 300.0, 1800.0),
        "soil_n_mean": (700.0, 1800.0, 300.0, 2600.0),
        "soil_p_mean": (800.0, 2500.0, 300.0, 4000.0),
        "soil_k_mean": (120.0, 700.0, 40.0, 1400.0),
        "soil_bdod_0_5cm_mean": (95.0, 145.0, 70.0, 170.0),
    },
    12: {
        "soil_ph_mean": (5.0, 6.8, 4.2, 7.8),
        "temp_mean": (26.0, 34.0, 22.0, 38.0),
        "rainfall_total_2023_24": (1000.0, 2200.0, 700.0, 3200.0),
        "soil_n_mean": (900.0, 2200.0, 400.0, 3000.0),
        "soil_p_mean": (1000.0, 3500.0, 400.0, 5000.0),
        "soil_k_mean": (120.0, 700.0, 40.0, 1400.0),
        "soil_bdod_0_5cm_mean": (100.0, 150.0, 75.0, 175.0),
    },
    31: {
        "soil_ph_mean": (6.0, 7.5, 5.0, 8.2),
        "temp_mean": (22.0, 30.0, 18.0, 34.0),
        "rainfall_total_2023_24": (600.0, 1200.0, 400.0, 1800.0),
        "soil_n_mean": (700.0, 1800.0, 300.0, 2600.0),
        "soil_p_mean": (900.0, 2500.0, 300.0, 4000.0),
        "soil_k_mean": (120.0, 700.0, 40.0, 1400.0),
        "soil_bdod_0_5cm_mean": (95.0, 145.0, 70.0, 170.0),
    },
    54: {
        "soil_ph_mean": (5.0, 6.5, 4.2, 7.5),
        "temp_mean": (20.0, 28.0, 16.0, 32.0),
        "rainfall_total_2023_24": (1200.0, 2200.0, 800.0, 3200.0),
        "soil_n_mean": (900.0, 2000.0, 400.0, 2800.0),
        "soil_p_mean": (800.0, 2500.0, 300.0, 4000.0),
        "soil_k_mean": (120.0, 700.0, 40.0, 1400.0),
        "soil_bdod_0_5cm_mean": (90.0, 140.0, 65.0, 165.0),
    },
    71: {
        "soil_ph_mean": (5.5, 7.0, 4.5, 8.0),
        "temp_mean": (25.0, 34.0, 21.0, 38.0),
        "rainfall_total_2023_24": (1200.0, 2500.0, 800.0, 3500.0),
        "soil_n_mean": (900.0, 2200.0, 400.0, 3000.0),
        "soil_p_mean": (1000.0, 3500.0, 400.0, 5000.0),
        "soil_k_mean": (150.0, 1000.0, 50.0, 1800.0),
        "soil_bdod_0_5cm_mean": (90.0, 140.0, 65.0, 165.0),
    },
}

HYBRID_ENVIRONMENT_WEIGHTS = {
    "soil_ph_mean": 0.20,
    "temp_mean": 0.20,
    "rainfall_total_2023_24": 0.20,
    "soil_n_mean": 0.12,
    "soil_p_mean": 0.12,
    "soil_k_mean": 0.08,
    "soil_bdod_0_5cm_mean": 0.08,
}

INPUT_SCORE_WEIGHTS = {
    "environment": 0.70,
    "market": 0.15,
    "regional_prior": 0.15,
}

MARKET_PRICE_COLUMNS = {
    11: "maize_recent_market_price_tzs_mean",
    12: "rice_paddy_recent_market_price_tzs_mean",
    31: "beans_recent_market_price_tzs_mean",
}


LIKERT_OPTIONS = {
    "English": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    "Kiswahili": ["Sikubaliani kabisa", "Sikubaliani", "Sina uhakika", "Nakubali", "Nakubali kabisa"],
}
LIKERT_SCORES = {label: index for index, label in enumerate(LIKERT_OPTIONS["English"], start=1)}
LIKERT_SCORES.update({label: index for index, label in enumerate(LIKERT_OPTIONS["Kiswahili"], start=1)})

TEXT = {
    "English": {
        "app_title": "Crop Recommendation User Evaluation",
        "intro": (
            "This is a prototype system that aims to gain your understanding of the outputs "
            "of crop recommendation system. The system contains two parts, each with surveys "
            "at the end, the first one is a regular Crop recommendation system, and the second "
            "one is an advanced system with Explainable AI. Provide your answers based on how "
            "you understand the systems."
        ),
        "language": "Language",
        "participant": "Participant",
        "participant_id": "Participant ID",
        "role": "Role",
        "years_experience": "Years of agricultural experience",
        "region": "Region",
        "saved_caption": "Responses are saved locally in `user_evaluation/responses_collected.csv`.",
        "selected_region": "Selected Region",
        "participant_role": "Participant role",
        "not_provided": "Not provided",
        "experience": "Experience",
        "years": "year(s)",
        "make_recommendation": "Make Your Own Recommendation",
        "input_help": "Enter farm and environmental values to generate a crop recommendation.",
        "soil_ph": "Soil pH",
        "temperature": "Mean temperature",
        "rainfall": "Total rainfall 2023/24",
        "bulk_density": "Topsoil bulk density",
        "nitrogen": "Soil nitrogen",
        "phosphorus": "Soil phosphorus",
        "potassium": "Soil potassium",
        "irrigation": "Irrigation available",
        "no": "No",
        "yes": "Yes",
        "generate": "Generate Recommendation",
        "recommendation": "Recommendation",
        "most_favored_crop": "Most favored crop",
        "first_submitted": "First evaluation submitted. The explanation is now available below.",
        "first_evaluation": "First Evaluation",
        "first_saved": "First evaluation saved. The explanation is now available.",
        "submit_first": "Submit the first evaluation to view the explanation.",
        "recommendation_with_explanation": "Recommendation With Explanation",
        "second_submitted": "Second evaluation submitted. Thank you for completing the assessment.",
        "second_evaluation": "Second Evaluation",
        "second_saved": "Second evaluation saved.",
        "comparative_saved": "Comparative assessment saved.",
        "rank": "Rank",
        "crop": "Crop",
        "input_score": "Input-based score",
        "environment": "Environment",
        "market": "Market",
        "regional_pattern": "Regional pattern",
        "factor": "Factor",
        "score": "Score",
        "comparative_heading": "Comparative Assessment & Insights",
        "preferred_system": "Which system would you prefer to use for daily agricultural decision-making?",
        "system_a": "System A (Direct recommendation only)",
        "system_b": "System B (Recommendation with explanations)",
        "no_preference": "No preference",
        "contradiction": "If System B provided an explanation that contradicted your personal experience, what would you do?",
        "trust_anyway": "Trust the system anyway and plant the recommended crop.",
        "reject": "Reject the recommendation entirely and rely on my own experience.",
        "investigate": "Investigate further, such as testing the soil again or consulting an expert.",
        "helpful_type": "What specific type of explanation do you find most helpful?",
        "text_descriptions": "Text-based descriptions",
        "visual_charts": "Visual charts",
        "contrastive": "Contrastive explanations",
        "additional_info": "What additional information or features should the explanation include to make you trust the system more?",
        "comments": "Comments or suggestions",
        "save": "Save",
        "role_line": "Participant role: **{role}** | Experience: **{years} year(s)**",
    },
    "Kiswahili": {
        "app_title": "Tathmini ya Mfumo wa Kupendekeza Mazao",
        "intro": (
            "Weka taarifa za shamba, angalia pendekezo la kwanza la zao, lifanyie tathmini, "
            "kisha angalia maelezo na ufanye tathmini tena."
        ),
        "language": "Lugha",
        "participant": "Mshiriki",
        "participant_id": "Namba ya mshiriki",
        "role": "Jukumu",
        "years_experience": "Miaka ya uzoefu katika kilimo",
        "region": "Mkoa",
        "saved_caption": "Majibu yanahifadhiwa kwenye `user_evaluation/responses_collected.csv`.",
        "selected_region": "Mkoa Ulioteuliwa",
        "participant_role": "Jukumu la mshiriki",
        "not_provided": "Haijatolewa",
        "experience": "Uzoefu",
        "years": "mwaka/miaka",
        "make_recommendation": "Tengeneza Pendekezo Lako",
        "input_help": "Weka taarifa za shamba na mazingira ili kupata pendekezo la zao.",
        "soil_ph": "pH ya udongo",
        "temperature": "Wastani wa joto",
        "rainfall": "Jumla ya mvua 2023/24",
        "bulk_density": "Msongamano wa udongo wa juu",
        "nitrogen": "Nitrojeni ya udongo",
        "phosphorus": "Fosforasi ya udongo",
        "potassium": "Potasiamu ya udongo",
        "irrigation": "Umwagiliaji upo",
        "no": "Hapana",
        "yes": "Ndiyo",
        "generate": "Tengeneza Pendekezo",
        "recommendation": "Pendekezo",
        "most_favored_crop": "Zao linalopendekezwa zaidi",
        "first_submitted": "Tathmini ya kwanza imehifadhiwa. Maelezo sasa yanapatikana hapa chini.",
        "first_evaluation": "Tathmini ya Kwanza",
        "first_saved": "Tathmini ya kwanza imehifadhiwa. Maelezo sasa yanapatikana.",
        "submit_first": "Wasilisha tathmini ya kwanza ili kuona maelezo.",
        "recommendation_with_explanation": "Pendekezo Lenye Maelezo",
        "second_submitted": "Tathmini ya pili imehifadhiwa. Asante kwa kukamilisha tathmini.",
        "second_evaluation": "Tathmini ya Pili",
        "second_saved": "Tathmini ya pili imehifadhiwa.",
        "comparative_saved": "Tathmini ya kulinganisha imehifadhiwa.",
        "rank": "Nafasi",
        "crop": "Zao",
        "input_score": "Alama kutokana na taarifa",
        "environment": "Mazingira",
        "market": "Soko",
        "regional_pattern": "Mwelekeo wa mkoa",
        "factor": "Kipengele",
        "score": "Alama",
        "comparative_heading": "Ulinganisho na Maoni",
        "preferred_system": "Ni mfumo gani ungependa kutumia kwa maamuzi ya kilimo ya kila siku?",
        "system_a": "Mfumo A (Pendekezo la moja kwa moja tu)",
        "system_b": "Mfumo B (Pendekezo lenye maelezo)",
        "no_preference": "Sina upendeleo",
        "contradiction": "Kama Mfumo B ukitoa maelezo yanayopingana na uzoefu wako, ungefanya nini?",
        "trust_anyway": "Ningeamini mfumo na kupanda zao lililopendekezwa.",
        "reject": "Ningekataa pendekezo na kutegemea uzoefu wangu.",
        "investigate": "Ningefuatilia zaidi, kama kupima udongo tena au kushauriana na mtaalamu.",
        "helpful_type": "Ni aina gani ya maelezo inayokusaidia zaidi?",
        "text_descriptions": "Maelezo ya maandishi",
        "visual_charts": "Chati za kuona",
        "contrastive": "Maelezo ya kulinganisha",
        "additional_info": "Ni taarifa au vipengele gani vingine vinafaa kuongezwa ili uamini mfumo zaidi?",
        "comments": "Maoni au mapendekezo",
        "save": "Hifadhi",
        "role_line": "Jukumu la mshiriki: **{role}** | Uzoefu: **{years} mwaka/miaka**",
    },
}

CROP_LABELS = {
    "English": {
        "Maize": "Maize",
        "Paddy": "Paddy",
        "Beans": "Beans",
        "Coffee": "Coffee",
        "Banana": "Banana",
    },
    "Kiswahili": {
        "Maize": "Mahindi",
        "Paddy": "Mpunga",
        "Beans": "Maharage",
        "Coffee": "Kahawa",
        "Banana": "Ndizi",
    },
}

BLACK_BOX_QUESTIONS = {
    "black_box_clarity_score": "Clarity: The recommendation provided is clear and easy to understand.",
    "black_box_trustworthiness_score": (
        "Trustworthiness: I trust this recommendation enough to invest money and plant this crop."
    ),
    "black_box_actionability_score": (
        "Actionability: The system provides enough information for me to confidently take action."
    ),
    "black_box_safety_risk_score": (
        "Safety/Risk: I feel confident that following this recommendation will not lead to crop failure."
    ),
}

XAI_QUESTIONS = {
    "xai_transparency_score": "Transparency: I understand why the system made this specific crop recommendation.",
    "xai_trustworthiness_score": (
        "Trustworthiness: The explanations provided make the system more trustworthy than a simple recommendation."
    ),
    "xai_agronomic_alignment_score": (
        "Agronomic Alignment: The reasons given by the AI match my practical knowledge and agricultural experience."
    ),
    "xai_decision_support_score": (
        "Decision Support: The explanation helps me understand the risks before planting."
    ),
    "xai_usability_score": (
        "Usability: The format of the explanation is easy to interpret."
    ),
}

BLACK_BOX_QUESTIONS_SW = {
    "black_box_clarity_score": "Uwazi: Pendekezo lililotolewa liko wazi na ni rahisi kuelewa.",
    "black_box_trustworthiness_score": (
        "Uaminifu: Ninaweza kuamini pendekezo hili kiasi cha kutumia fedha na kupanda zao hili."
    ),
    "black_box_actionability_score": (
        "Utekelezaji: Mfumo unatoa taarifa za kutosha kunisaidia kuchukua hatua kwa kujiamini."
    ),
    "black_box_safety_risk_score": (
        "Usalama/Hatari: Ninaamini kuwa kufuata pendekezo hili hakutasababisha kushindwa kwa zao."
    ),
}

XAI_QUESTIONS_SW = {
    "xai_transparency_score": "Uwazi: Ninaelewa kwa nini mfumo umependekeza zao hili.",
    "xai_trustworthiness_score": (
        "Uaminifu: Maelezo yaliyotolewa yanafanya mfumo uaminike zaidi kuliko pendekezo pekee."
    ),
    "xai_agronomic_alignment_score": (
        "Ulinganifu wa kilimo: Sababu zilizotolewa na AI zinaendana na maarifa na uzoefu wangu wa kilimo."
    ),
    "xai_decision_support_score": (
        "Msaada wa maamuzi: Maelezo yananisaidia kuelewa hatari kabla ya kupanda."
    ),
    "xai_usability_score": "Utumiaji: Muundo wa maelezo ni rahisi kuelewa.",
}


def tr(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["English"]).get(key, TEXT["English"].get(key, key))


def crop_label(crop: object, language: str) -> str:
    return CROP_LABELS.get(language, CROP_LABELS["English"]).get(str(crop), str(crop))


def translated_question_set(question_set: dict[str, str], language: str) -> dict[str, str]:
    if language == "Kiswahili" and question_set is BLACK_BOX_QUESTIONS:
        return BLACK_BOX_QUESTIONS_SW
    if language == "Kiswahili" and question_set is XAI_QUESTIONS:
        return XAI_QUESTIONS_SW
    return question_set


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(RECOMMENDATIONS_PATH)


@st.cache_data
def load_xai_details() -> pd.DataFrame:
    return pd.read_csv(XAI_DETAILS_PATH)


@st.cache_data
def load_xai_summary() -> pd.DataFrame:
    return pd.read_csv(XAI_SUMMARY_PATH)


@st.cache_data
def load_region_defaults() -> pd.DataFrame:
    return pd.read_csv(REGION_DEFAULTS_PATH)


@st.cache_data
def load_market_features() -> pd.DataFrame:
    return pd.read_csv(MARKET_FEATURES_PATH)


@st.cache_data
def load_case_study_counts() -> pd.DataFrame:
    return pd.read_csv(CASE_STUDY_COUNTS_PATH)


def score_label(value: float) -> str:
    return f"{value:.3f}"


def clean_feature_name(name: str) -> str:
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    cleaned = cleaned.replace("_", " ")
    return cleaned.strip()


def interpretation_for_factor(column: str, language: str = "English") -> str:
    labels = {
        "English": {
            "model_probability": "Model confidence in the crop class",
            "environment_score": "Environmental suitability from soil, rainfall, and temperature",
            "market_score": "Market price or market attractiveness signal",
            "regional_prior_score": "How common the crop is in similar regional records",
            "soil_ph_mean_score": "Soil pH suitability",
            "temp_mean_score": "Temperature suitability",
            "rainfall_total_2023_24_score": "Rainfall suitability",
            "soil_n_mean_score": "Nitrogen suitability",
            "soil_p_mean_score": "Phosphorus suitability",
            "soil_k_mean_score": "Potassium suitability",
            "soil_bdod_0_5cm_mean_score": "Topsoil bulk-density suitability",
        },
        "Kiswahili": {
            "model_probability": "Uhakika wa modeli kuhusu aina ya zao",
            "environment_score": "Ufaa wa mazingira kulingana na udongo, mvua, na joto",
            "market_score": "Ishara ya bei au mvuto wa soko",
            "regional_prior_score": "Kiasi ambacho zao hulimwa katika rekodi za mkoa",
            "soil_ph_mean_score": "Ufaa wa pH ya udongo",
            "temp_mean_score": "Ufaa wa joto",
            "rainfall_total_2023_24_score": "Ufaa wa mvua",
            "soil_n_mean_score": "Ufaa wa nitrojeni",
            "soil_p_mean_score": "Ufaa wa fosforasi",
            "soil_k_mean_score": "Ufaa wa potasiamu",
            "soil_bdod_0_5cm_mean_score": "Ufaa wa msongamano wa udongo wa juu",
        },
    }
    return labels.get(language, labels["English"]).get(column, clean_feature_name(column))


def bounded_range_score(value: float, ideal_low: float, ideal_high: float, min_value: float, max_value: float) -> float:
    if pd.isna(value):
        return 0.5

    value = float(value)
    if ideal_low <= value <= ideal_high:
        return 1.0
    if min_value < value < ideal_low:
        return max(0.0, (value - min_value) / (ideal_low - min_value))
    if ideal_high < value < max_value:
        return max(0.0, (max_value - value) / (max_value - ideal_high))
    return 0.0


def environment_suitability_score(row: pd.Series, crop_id: int) -> tuple[float, dict[str, float]]:
    rules = HYBRID_SUITABILITY_RULES.get(crop_id, {})
    scores = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for feature, limits in rules.items():
        if feature not in row.index:
            continue
        score = bounded_range_score(row[feature], *limits)
        weight = HYBRID_ENVIRONMENT_WEIGHTS.get(feature, 0.0)
        scores[f"{feature}_score"] = score
        weighted_sum += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.5, scores
    return weighted_sum / weight_sum, scores


def market_score_from_input(row: pd.Series, crop_id: int, market_features: pd.DataFrame) -> float:
    column = MARKET_PRICE_COLUMNS.get(crop_id)
    if column is None or column not in market_features.columns:
        return 0.5

    value = row.get(column)
    values = market_features[column].dropna()
    if pd.isna(value) or values.empty or float(values.max()) == float(values.min()):
        return 0.5

    return max(0.0, min(1.0, (float(value) - float(values.min())) / (float(values.max()) - float(values.min()))))


def regional_prior_lookup(region: str, crop_id: int, counts: pd.DataFrame) -> float:
    region_rows = counts[counts["region_name"] == region]
    if not region_rows.empty:
        total = float(region_rows["records"].sum())
        crop_records = region_rows.loc[region_rows["crop_id"] == crop_id, "records"].sum()
        if total > 0:
            return float(crop_records) / total

    total = float(counts["records"].sum())
    crop_records = counts.loc[counts["crop_id"] == crop_id, "records"].sum()
    if total > 0:
        return float(crop_records) / total
    return 0.2


def recommend_from_inputs(
    input_row: pd.Series,
    market_features: pd.DataFrame,
    case_counts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for crop_id, crop_name in CASE_STUDY_CROPS.items():
        environment_score, environment_parts = environment_suitability_score(input_row, crop_id)
        crop_market_score = market_score_from_input(input_row, crop_id, market_features)
        crop_regional_prior = regional_prior_lookup(str(input_row["region_name"]), crop_id, case_counts)
        input_score = (
            INPUT_SCORE_WEIGHTS["environment"] * environment_score
            + INPUT_SCORE_WEIGHTS["market"] * crop_market_score
            + INPUT_SCORE_WEIGHTS["regional_prior"] * crop_regional_prior
        )
        rows.append(
            {
                "recommended_crop_id": crop_id,
                "recommended_crop": crop_name,
                "input_score": input_score,
                "environment_score": environment_score,
                "market_score": crop_market_score,
                "regional_prior_score": crop_regional_prior,
                **environment_parts,
            }
        )

    result = pd.DataFrame(rows).sort_values("input_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result


def show_input_recommendation_table(rows: pd.DataFrame, language: str) -> None:
    display = rows[
        ["rank", "recommended_crop", "input_score", "environment_score", "market_score", "regional_prior_score"]
    ].copy()
    display["recommended_crop"] = display["recommended_crop"].map(lambda value: crop_label(value, language))
    display = display.rename(
        columns={
            "rank": tr(language, "rank"),
            "recommended_crop": tr(language, "crop"),
            "input_score": tr(language, "input_score"),
            "environment_score": tr(language, "environment"),
            "market_score": tr(language, "market"),
            "regional_prior_score": tr(language, "regional_pattern"),
        }
    )
    st.dataframe(display, hide_index=True, use_container_width=True)


def show_input_factor_breakdown(top_row: pd.Series, language: str) -> None:
    component_columns = [
        "environment_score",
        "market_score",
        "regional_prior_score",
        "soil_ph_mean_score",
        "temp_mean_score",
        "rainfall_total_2023_24_score",
        "soil_n_mean_score",
        "soil_p_mean_score",
        "soil_k_mean_score",
        "soil_bdod_0_5cm_mean_score",
    ]
    component_rows = [
        {
            tr(language, "factor"): interpretation_for_factor(column, language),
            tr(language, "score"): score_label(float(top_row[column])),
        }
        for column in component_columns
        if column in top_row.index
    ]
    st.dataframe(pd.DataFrame(component_rows), hide_index=True, use_container_width=True)


def layman_factor_name(column: str, language: str = "English") -> str:
    labels = {
        "English": {
            "environment_score": "the farm conditions",
            "market_score": "the market price",
            "regional_prior_score": "what farmers commonly grow in this region",
            "soil_ph_mean_score": "soil pH",
            "temp_mean_score": "temperature",
            "rainfall_total_2023_24_score": "rainfall",
            "soil_n_mean_score": "soil nitrogen",
            "soil_p_mean_score": "soil phosphorus",
            "soil_k_mean_score": "soil potassium",
            "soil_bdod_0_5cm_mean_score": "topsoil condition",
        },
        "Kiswahili": {
            "environment_score": "hali ya shamba",
            "market_score": "bei ya soko",
            "regional_prior_score": "mazao ambayo wakulima wa mkoa huu hulima mara nyingi",
            "soil_ph_mean_score": "pH ya udongo",
            "temp_mean_score": "joto",
            "rainfall_total_2023_24_score": "mvua",
            "soil_n_mean_score": "nitrojeni ya udongo",
            "soil_p_mean_score": "fosforasi ya udongo",
            "soil_k_mean_score": "potasiamu ya udongo",
            "soil_bdod_0_5cm_mean_score": "hali ya udongo wa juu",
        },
    }
    return labels.get(language, labels["English"]).get(column, clean_feature_name(column))


def build_layman_explanation(top_row: pd.Series, language: str) -> str:
    crop = crop_label(top_row["recommended_crop"], language)
    main_factors = ["environment_score", "market_score", "regional_prior_score"]
    strongest_main = max(main_factors, key=lambda column: float(top_row.get(column, 0)))

    detail_factors = [
        "soil_ph_mean_score",
        "temp_mean_score",
        "rainfall_total_2023_24_score",
        "soil_n_mean_score",
        "soil_p_mean_score",
        "soil_k_mean_score",
        "soil_bdod_0_5cm_mean_score",
    ]
    strong_details = [
        layman_factor_name(column, language)
        for column in detail_factors
        if column in top_row.index and float(top_row[column]) >= 0.8
    ][:3]
    weak_details = [
        layman_factor_name(column, language)
        for column in detail_factors
        if column in top_row.index and float(top_row[column]) < 0.5
    ][:2]

    if language == "Kiswahili":
        explanation = (
            f"Mfumo unapendelea {crop} hasa kwa sababu {layman_factor_name(strongest_main, language)} "
            "kinatoa alama nzuri. "
        )
        if strong_details:
            explanation += f"Kwa lugha rahisi, {', '.join(strong_details)} vinaonekana kufaa kwa zao hili. "
        if weak_details:
            explanation += f"Vipengele dhaifu ni {', '.join(weak_details)}, hivyo vinapaswa kukaguliwa kabla ya kuamua. "
        explanation += (
            "Alama ya juu haimaanishi mafanikio ya uhakika; inaonyesha taarifa ulizoingiza zinaendana na zao hili zaidi "
            "kuliko mazao mengine yaliyofanyiwa tathmini na programu."
        )
        return explanation

    explanation = (
        f"The system favors {crop} mainly because {layman_factor_name(strongest_main, language)} gives it a strong score. "
    )
    if strong_details:
        explanation += f"In simple terms, {', '.join(strong_details)} look suitable for this crop. "
    if weak_details:
        explanation += f"The weaker parts are {', '.join(weak_details)}, so those should be checked before deciding. "
    explanation += (
        "A higher score does not guarantee success; it means the information entered matches this crop better "
        "than the other crops evaluated by the app."
    )
    return explanation


def region_default_value(defaults: pd.DataFrame, region: str, column: str, fallback: float) -> float:
    if column not in defaults.columns:
        return fallback
    region_rows = defaults[defaults["region_name"] == region]
    if region_rows.empty or pd.isna(region_rows.iloc[0][column]):
        return fallback
    return float(region_rows.iloc[0][column])


def market_default_value(market_features: pd.DataFrame, region: str, column: str, fallback: float) -> float:
    if column not in market_features.columns:
        return fallback
    region_rows = market_features[market_features["region_name"] == region]
    if region_rows.empty or pd.isna(region_rows.iloc[0][column]):
        values = market_features[column].dropna()
        if values.empty:
            return fallback
        return float(values.median())
    return float(region_rows.iloc[0][column])


def show_custom_input_recommender(
    region: str,
    region_defaults: pd.DataFrame,
    market_features: pd.DataFrame,
    case_counts: pd.DataFrame,
    participant_id: str,
    role: str,
    years_experience: int,
    language: str,
) -> None:
    st.subheader(tr(language, "make_recommendation"))
    st.write(tr(language, "input_help"))

    with st.form("custom_recommendation_form"):
        left, right = st.columns(2)
        with left:
            soil_ph = st.number_input(
                tr(language, "soil_ph"),
                min_value=0.0,
                max_value=14.0,
                value=region_default_value(region_defaults, region, "soil_ph_mean", 6.0),
                step=0.1,
            )
            temperature = st.number_input(
                tr(language, "temperature"),
                min_value=0.0,
                max_value=45.0,
                value=region_default_value(region_defaults, region, "temp_mean", 26.0),
                step=0.1,
            )
            rainfall = st.number_input(
                tr(language, "rainfall"),
                min_value=0.0,
                max_value=5000.0,
                value=region_default_value(region_defaults, region, "rainfall_total_2023_24", 1200.0),
                step=10.0,
            )
            bulk_density = st.number_input(
                tr(language, "bulk_density"),
                min_value=0.0,
                max_value=250.0,
                value=region_default_value(region_defaults, region, "soil_bdod_0_5cm_mean", 125.0),
                step=1.0,
            )
        with right:
            nitrogen = st.number_input(
                tr(language, "nitrogen"),
                min_value=0.0,
                max_value=4000.0,
                value=region_default_value(region_defaults, region, "soil_n_mean", 1000.0),
                step=10.0,
            )
            phosphorus = st.number_input(
                tr(language, "phosphorus"),
                min_value=0.0,
                max_value=6000.0,
                value=region_default_value(region_defaults, region, "soil_p_mean", 1200.0),
                step=10.0,
            )
            potassium = st.number_input(
                tr(language, "potassium"),
                min_value=0.0,
                max_value=2500.0,
                value=region_default_value(region_defaults, region, "soil_k_mean", 250.0),
                step=10.0,
            )
            irrigation_display = st.selectbox(tr(language, "irrigation"), [tr(language, "no"), tr(language, "yes")])
            irrigation = "Yes" if irrigation_display == tr(language, "yes") else "No"

        maize_price = market_default_value(
            market_features,
            region,
            "maize_recent_market_price_tzs_mean",
            40000.0,
        )
        rice_price = market_default_value(
            market_features,
            region,
            "rice_paddy_recent_market_price_tzs_mean",
            110000.0,
        )
        beans_price = market_default_value(
            market_features,
            region,
            "beans_recent_market_price_tzs_mean",
            100000.0,
        )

        submitted = st.form_submit_button(tr(language, "generate"))

    if submitted:
        input_row = pd.Series(
            {
                "region_name": region,
                "soil_ph_mean": soil_ph,
                "temp_mean": temperature,
                "rainfall_total_2023_24": rainfall,
                "soil_n_mean": nitrogen,
                "soil_p_mean": phosphorus,
                "soil_k_mean": potassium,
                "soil_bdod_0_5cm_mean": bulk_density,
                "irrigation_available": irrigation,
                "maize_recent_market_price_tzs_mean": maize_price,
                "rice_paddy_recent_market_price_tzs_mean": rice_price,
                "beans_recent_market_price_tzs_mean": beans_price,
            }
        )
        input_recommendations = recommend_from_inputs(input_row, market_features, case_counts)
        st.session_state["input_row"] = input_row.to_dict()
        st.session_state["input_recommendations"] = input_recommendations.to_dict("records")
        st.session_state["first_evaluation_submitted"] = False
        st.session_state["second_evaluation_submitted"] = False

    if "input_recommendations" not in st.session_state:
        return

    input_recommendations = pd.DataFrame(st.session_state["input_recommendations"])
    input_row = pd.Series(st.session_state["input_row"])
    top_row = input_recommendations.iloc[0]

    st.subheader(tr(language, "recommendation"))
    st.success(f"{tr(language, 'most_favored_crop')}: {crop_label(top_row['recommended_crop'], language)}")

    if st.session_state.get("first_evaluation_submitted", False):
        st.markdown(
            "<div style='opacity:0.55; padding:0.75rem 1rem; border:1px solid #ddd; border-radius:0.5rem;'>"
            f"{tr(language, 'first_submitted')}"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        first_saved = show_evaluation_form(
            stage="before_explanation",
            title=tr(language, "first_evaluation"),
            participant_id=participant_id,
            role=role,
            years_experience=years_experience,
            region=region,
            top_row=top_row,
            input_row=input_row,
            question_set=BLACK_BOX_QUESTIONS,
            include_comments=False,
            language=language,
        )
        if first_saved:
            st.session_state["first_evaluation_submitted"] = True
            st.success(tr(language, "first_saved"))
            st.rerun()

    if not st.session_state.get("first_evaluation_submitted", False):
        st.info(tr(language, "submit_first"))
        return

    st.subheader(tr(language, "recommendation_with_explanation"))
    show_input_recommendation_table(input_recommendations, language)
    st.write(build_layman_explanation(top_row, language))
    show_input_factor_breakdown(top_row, language)

    if st.session_state.get("second_evaluation_submitted", False):
        st.markdown(
            "<div style='opacity:0.55; padding:0.75rem 1rem; border:1px solid #ddd; border-radius:0.5rem;'>"
            f"{tr(language, 'second_submitted')}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.subheader(tr(language, "comparative_heading"))
        if st.session_state.get("comparative_assessment_submitted", False):
            st.success(tr(language, "comparative_saved"))
        else:
            comparative_saved = show_comparative_assessment_form(
                stage="comparative_assessment",
                participant_id=participant_id,
                role=role,
                years_experience=years_experience,
                region=region,
                top_row=top_row,
                input_row=input_row,
                language=language,
            )
            if comparative_saved:
                st.session_state["comparative_assessment_submitted"] = True
                st.success(tr(language, "comparative_saved"))
                st.rerun()
    else:
        second_saved = show_evaluation_form(
            stage="after_explanation",
            title=tr(language, "second_evaluation"),
            participant_id=participant_id,
            role=role,
            years_experience=years_experience,
            region=region,
            top_row=top_row,
            input_row=input_row,
            question_set=XAI_QUESTIONS,
            include_comments=False,
            language=language,
        )
        if second_saved:
            st.session_state["second_evaluation_submitted"] = True
            st.success(tr(language, "second_saved"))
            st.rerun()


def show_recommendation_table(rows: pd.DataFrame, include_components: bool) -> None:
    display_columns = ["rank", "recommended_crop", "hybrid_score"]
    if include_components:
        display_columns.extend(
            [
                "model_probability",
                "environment_score",
                "market_score",
                "regional_prior_score",
            ]
        )

    display = rows[display_columns].copy()
    rename_map = {
        "rank": "Rank",
        "recommended_crop": "Crop",
        "hybrid_score": "Final score",
        "model_probability": "Model confidence",
        "environment_score": "Environment",
        "market_score": "Market",
        "regional_prior_score": "Regional pattern",
    }
    display = display.rename(columns=rename_map)
    st.dataframe(display, hide_index=True, use_container_width=True)


def show_xai_explanation(rows: pd.DataFrame, xai_details: pd.DataFrame, xai_summary: pd.DataFrame) -> None:
    top_row = rows.sort_values("rank").iloc[0]
    st.subheader("Why This Crop Was Recommended")
    st.write(f"Top recommendation: **{top_row['recommended_crop']}**")

    component_columns = [
        "model_probability",
        "environment_score",
        "market_score",
        "regional_prior_score",
        "soil_ph_mean_score",
        "temp_mean_score",
        "rainfall_total_2023_24_score",
        "soil_n_mean_score",
        "soil_p_mean_score",
        "soil_k_mean_score",
        "soil_bdod_0_5cm_mean_score",
    ]
    component_rows = []
    for column in component_columns:
        if column in top_row.index:
            component_rows.append(
                {
                    "Factor": interpretation_for_factor(column),
                    "Score": score_label(float(top_row[column])),
                }
            )

    st.dataframe(pd.DataFrame(component_rows), hide_index=True, use_container_width=True)

    best_technique = xai_summary.sort_values("overall_rank").iloc[0]["technique"]
    best_examples = xai_details[xai_details["technique"] == best_technique].copy()
    crop_matches = best_examples[
        best_examples["predicted_crop"].astype(str).str.lower()
        == str(top_row["recommended_crop"]).lower()
    ]
    if not crop_matches.empty:
        best_examples = crop_matches
    example = best_examples.sort_values("local_fidelity", ascending=False).iloc[0]
    features = [clean_feature_name(item) for item in str(example["top_features"]).split(";") if item.strip()]

    st.subheader(f"{best_technique} Explanation Factors")
    st.write(
        "The explanation below shows the most influential features identified by the best-ranked XAI method in this project."
    )
    st.dataframe(
        pd.DataFrame({"Important feature": features[:8]}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Explanation fidelity: {float(example['local_fidelity']):.3f}; "
        f"stability: {float(example['stability_jaccard']):.3f}; "
        f"coverage: {float(example['coverage']):.3f}."
    )


def show_evaluation_form(
    stage: str,
    title: str,
    participant_id: str,
    role: str,
    years_experience: int,
    region: str,
    top_row: pd.Series,
    input_row: pd.Series,
    question_set: dict[str, str],
    include_comments: bool = True,
    language: str = "English",
) -> bool:
    st.subheader(title)

    with st.form(f"{stage}_evaluation_form"):
        scores = {}
        for column, question in translated_question_set(question_set, language).items():
            selected = st.radio(
                question,
                LIKERT_OPTIONS[language],
                index=2,
                horizontal=True,
                key=f"{stage}_{region}_{column}",
            )
            scores[column] = LIKERT_SCORES[selected]
            scores[f"{column}_label"] = selected

        comments = None
        if include_comments:
            comments = st.text_area(tr(language, "comments"), key=f"{stage}_{region}_comments")
        submitted = st.form_submit_button(f"{tr(language, 'save')} {title}")

    if not submitted:
        return False

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "role": role,
        "years_experience": years_experience,
        "language": language,
        "condition": stage,
        "region": region,
        "recommended_crop": top_row["recommended_crop"],
        "recommendation_score": top_row["input_score"],
        "environment_score": top_row["environment_score"],
        "market_score": top_row["market_score"],
        "regional_prior_score": top_row["regional_prior_score"],
        **{f"input_{key}": value for key, value in input_row.to_dict().items()},
        **scores,
        "comments": comments,
    }
    save_response(row)
    return True


def show_comparative_assessment_form(
    stage: str,
    participant_id: str,
    role: str,
    years_experience: int,
    region: str,
    top_row: pd.Series,
    input_row: pd.Series,
    language: str = "English",
) -> bool:
    with st.form(f"{stage}_form"):
        preferred_system = st.radio(
            tr(language, "preferred_system"),
            [
                tr(language, "system_a"),
                tr(language, "system_b"),
                tr(language, "no_preference"),
            ],
            key=f"{stage}_{region}_preferred_system",
        )
        contradiction_response = st.radio(
            tr(language, "contradiction"),
            [
                tr(language, "trust_anyway"),
                tr(language, "reject"),
                tr(language, "investigate"),
            ],
            key=f"{stage}_{region}_contradiction_response",
        )
        helpful_explanation_type = st.radio(
            tr(language, "helpful_type"),
            [
                tr(language, "text_descriptions"),
                tr(language, "visual_charts"),
                tr(language, "contrastive"),
            ],
            key=f"{stage}_{region}_helpful_explanation_type",
        )
        additional_explanation_info = st.text_area(
            tr(language, "additional_info"),
            key=f"{stage}_{region}_additional_explanation_info",
        )
        comments = st.text_area(tr(language, "comments"), key=f"{stage}_{region}_comments")
        submitted = st.form_submit_button(f"{tr(language, 'save')} {tr(language, 'comparative_heading')}")

    if not submitted:
        return False

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "role": role,
        "years_experience": years_experience,
        "language": language,
        "condition": stage,
        "region": region,
        "recommended_crop": top_row["recommended_crop"],
        "recommendation_score": top_row["input_score"],
        "environment_score": top_row["environment_score"],
        "market_score": top_row["market_score"],
        "regional_prior_score": top_row["regional_prior_score"],
        **{f"input_{key}": value for key, value in input_row.to_dict().items()},
        "preferred_system": preferred_system,
        "contradiction_response": contradiction_response,
        "helpful_explanation_type": helpful_explanation_type,
        "additional_explanation_info": additional_explanation_info,
        "comments": comments,
    }
    save_response(row)
    return True


def save_response(row: dict[str, object]) -> None:
    response = pd.DataFrame([row])
    if RESPONSES_PATH.exists():
        existing = pd.read_csv(RESPONSES_PATH)
        response = pd.concat([existing, response], ignore_index=True)
    response.to_csv(RESPONSES_PATH, index=False)


def main() -> None:
    st.set_page_config(page_title="Crop Recommendation XAI Evaluation", layout="wide")

    region_defaults = load_region_defaults()
    market_features = load_market_features()
    case_counts = load_case_study_counts()

    with st.sidebar:
        language = st.selectbox("Language / Lugha", ["English", "Kiswahili"])
        st.header(tr(language, "participant"))
        participant_id = st.text_input(tr(language, "participant_id"), value="P001")
        role = st.text_input(tr(language, "role"), value="")
        years_experience = st.number_input(tr(language, "years_experience"), min_value=0, max_value=80, value=0)
        region = st.selectbox(tr(language, "region"), sorted(region_defaults["region_name"].dropna().unique()))

    st.title(tr(language, "app_title"))
    st.write(tr(language, "intro"))

    st.header(f"{tr(language, 'selected_region')}: {region}")
    st.write(
        tr(language, "role_line").format(
            role=role or tr(language, "not_provided"),
            years=years_experience,
        )
    )

    if st.session_state.get("active_region") != region or st.session_state.get("active_language") != language:
        st.session_state["active_region"] = region
        st.session_state["active_language"] = language
        for key in [
            "input_row",
            "input_recommendations",
            "first_evaluation_submitted",
            "second_evaluation_submitted",
        ]:
            st.session_state.pop(key, None)

    show_custom_input_recommender(
        region,
        region_defaults,
        market_features,
        case_counts,
        participant_id,
        role,
        years_experience,
        language,
    )


if __name__ == "__main__":
    main()
