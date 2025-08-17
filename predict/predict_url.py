import os
import re
import ipaddress
import json
import joblib
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import google.generativeai as genai
from feature_extraction.extract_features import transform_url

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load ML model (IsolationForest or similar)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "phishing_model.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)
ml_model = joblib.load(MODEL_PATH)

# Gemini model
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "models/gemini-1.5-pro-latest")
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)

# Known brands for simple lookalike checks (extend as needed)
KNOWN_BRANDS = {
    "instagram": "instagram.com",
    "facebook": "facebook.com",
    "meta": "meta.com",
    "google": "google.com",
    "gmail": "mail.google.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "office": "office.com",
    "outlook": "outlook.com",
    "paypal": "paypal.com",
    "netflix": "netflix.com",
    "amazon": "amazon.com",
    "whatsapp": "whatsapp.com",
    "linkedin": "linkedin.com",
    "x": "x.com",
    "twitter": "twitter.com",
    "github": "github.com",
}

SUSPICIOUS_TLDS = {
    "zip", "country", "kim", "loan", "mom", "men", "work", "click", "xyz", "link", "gq", "cf", "tk"
}

URL_SHORTENERS = {
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly", "adf.ly", "rebrand.ly", "cutt.ly"
}

# -----------------------------
# Utility helpers
# -----------------------------
def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False

def levenshtein(a: str, b: str) -> int:
    # O(len(a)*len(b)) dynamic programming
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]

def domain_parts(host: str):
    # very lightweight parse; if you have tldextract, prefer it
    parts = host.split(".")
    if len(parts) < 2:
        return host, "", ""
    sld = parts[-2]
    tld = parts[-1]
    sub = ".".join(parts[:-2]) if len(parts) > 2 else ""
    return sub, sld, tld

def local_signal_checks(url: str):
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    sub, sld, tld = domain_parts(host)
    reasons = []
    signals = {
        "scheme": scheme or "",
        "host": host or "",
        "path": path,
        "query": query,
        "is_https": scheme == "https",
        "has_at_symbol": "@" in url,
        "has_ip_host": is_ip(host),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "has_many_subdomains": False,
        "has_hyphen_in_domain": "-" in sld,
        "has_digits_in_domain": bool(re.search(r"\d", sld)),
        "tld": tld,
        "suspicious_tld": tld in SUSPICIOUS_TLDS,
        "is_shortener": host in URL_SHORTENERS,
        "length": len(url),
        "brand_lookalike": None,
        "brand_distance": None,
        "brand_target": None,
    }

    # Many subdomains signal
    signals["has_many_subdomains"] = signals["subdomain_count"] >= 3

    # Brand lookalike (simple)
    best_brand = None
    best_dist = None
    for brand, legit in KNOWN_BRANDS.items():
        legit_host = legit.lower()
        legit_sld = domain_parts(legit_host)[1]
        dist = levenshtein(sld, legit_sld)
        if best_dist is None or dist < best_dist:
            best_brand = (brand, legit_host)
            best_dist = dist

    if best_brand:
        signals["brand_target"] = best_brand[0]
        signals["brand_lookalike"] = best_dist is not None and best_dist <= 2 and sld != domain_parts(best_brand[1])[1]
        signals["brand_distance"] = best_dist

    # Reasons (human-readable) based on signals
    if scheme != "https":
        reasons.append("The URL uses HTTP instead of HTTPS.")
    if signals["has_at_symbol"]:
        reasons.append("The URL contains an '@' symbol, which can hide the real destination.")
    if signals["has_ip_host"]:
        reasons.append("The host is an IP address instead of a domain name.")
    if signals["has_many_subdomains"]:
        reasons.append("The URL has an unusually high number of subdomains.")
    if signals["has_hyphen_in_domain"]:
        reasons.append("The domain contains hyphens, often used in deceptive domains.")
    if signals["has_digits_in_domain"]:
        reasons.append("The domain contains digits, which can indicate impersonation.")
    if signals["suspicious_tld"]:
        reasons.append(f"The top-level domain '.{tld}' is commonly abused in phishing.")
    if signals["is_shortener"]:
        reasons.append("The domain is a known URL shortener, which can hide the final destination.")
    if signals["brand_lookalike"] and signals["brand_target"]:
        legit = KNOWN_BRANDS[signals["brand_target"]]
        reasons.append(
            f"Possible typosquatting: '{sld}' looks like '{legit}' (edit distance {signals['brand_distance']})."
        )

    # Very rough local risk score
    weight = 0
    weight += 2 if scheme != "https" else 0
    weight += 2 if signals["brand_lookalike"] else 0
    weight += 1 if signals["has_ip_host"] else 0
    weight += 1 if signals["has_many_subdomains"] else 0
    weight += 1 if signals["suspicious_tld"] else 0
    weight += 1 if signals["is_shortener"] else 0
    weight += 1 if signals["has_hyphen_in_domain"] else 0
    weight += 1 if signals["has_digits_in_domain"] else 0
    # Clamp to [0,1]
    local_risk = min(1.0, weight / 7.0)

    line_by_line = {
        "scheme": f"{scheme} (secure only if https)",
        "domain": host,
        "subdomains": sub or "(none)",
        "sld": sld or "(n/a)",
        "tld": tld or "(n/a)",
        "path": path or "/",
        "query": query or "(none)"
    }

    return signals, reasons, local_risk, line_by_line

# -----------------------------
# Gemini JSON classifier
# -----------------------------
def gemini_classifier(url: str, observed_signals: dict, local_reasons: list, line_by_line: dict) -> dict:
    """
    Ask Gemini to return STRICT JSON, no markdown, with fields:
    verdict: "phishing" | "safe"
    confidence: float 0..1
    reasons: list[str]
    original_legit_domain: str (the most likely correct safe domain, e.g., 'https://www.instagram.com')
    user_message: str (plain text, single paragraph, no asterisks)
    line_by_line_explanation: dict[str, str]
    """
    try:
        prompt = (
            "You are a cybersecurity analyst. Analyze the given URL for phishing.\n"
            "Return ONLY a JSON object with keys: verdict, confidence, reasons, user_message, line_by_line_explanation, original_legit_domain.\n"
            "Constraints:\n"
            "- verdict must be exactly 'phishing' or 'safe'\n"
            "- confidence is a float between 0 and 1\n"
            "- reasons is an array of short sentences\n"
            "- original_legit_domain should be the most likely correct safe domain (e.g., if given 'http://insta-gram-login.tk', return 'https://www.instagram.com').\n"
            "If the domain already looks legitimate, return it as https:// + hostname.\n"
            "- user_message is a single-paragraph plain sentence summary for end users (no markdown, no asterisks)\n"
            "- line_by_line_explanation maps scheme, domain, path, query to short explanations (no markdown)\n"
            "Do not include any extra text outside the JSON.\n\n"
            f"URL: {url}\n\n"
            f"ObservedSignalsJSON:\n{json.dumps(observed_signals, ensure_ascii=False)}\n\n"
            f"LocalReasonsJSON:\n{json.dumps(local_reasons, ensure_ascii=False)}\n\n"
            f"LineByLineJSON:\n{json.dumps(line_by_line, ensure_ascii=False)}"
        )

        response = gemini_model.generate_content(prompt)
        raw = (response.text or "").strip()

        # Some models may wrap code fences; strip them
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            # Try to remove "json" language tag if present
            raw = re.sub(r"^json\s*", "", raw, flags=re.IGNORECASE).strip()

        data = json.loads(raw)
        verdict = str(data.get("verdict", "")).lower().strip()
        if verdict not in {"phishing", "safe"}:
            verdict = "unknown"

        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        return {
            "label": verdict,
            "confidence": conf,
            "reasons": data.get("reasons", []),
            "user_message": data.get("user_message", ""),
            "line_by_line_explanation": data.get("line_by_line_explanation", {}),
            "original_legit_domain": data.get("original_legit_domain", ""),
            "raw_json": data
        }

    except Exception as e:
        # Fallback to a minimal result using local signals
        fallback_label = "phishing" if observed_signals.get("brand_lookalike") or not observed_signals.get("is_https") else "safe"
        return {
            "label": fallback_label,
            "confidence": 0.6 if fallback_label == "phishing" else 0.5,
            "reasons": local_reasons or ["Model error; using local heuristics."],
            "user_message": "Automated analysis encountered an issue; decision made using local checks.",
            "line_by_line_explanation": line_by_line,
            "original_legit_domain": "",
            "error": str(e),
            "raw_json": {}
        }

# -----------------------------
# ML classifier wrapper
# -----------------------------
def ml_classifier(features_array: np.ndarray) -> dict:
    try:
        pred = ml_model.predict(features_array)[0]
        # Common IsolationForest mapping: -1 = anomaly (phishing), 1 = inlier (safe)
        if pred in (-1, 1):
            label = "phishing" if pred == -1 else "safe"
        else:
            # If your model uses 1==phishing, 0==safe, keep compatibility
            label = "phishing" if pred == 1 else "safe"
        return {"label": label, "raw_pred": int(pred)}
    except Exception as e:
        return {"label": "error", "error": str(e)}

# -----------------------------
# Public API
# -----------------------------
# -----------------------------
# Public API
# -----------------------------
def predict_url(url: str) -> dict:
    try:
        # Local signals
        signals, local_reasons, local_risk, line_by_line = local_signal_checks(url)

        # Feature extraction for ML model
        features = transform_url(url)
        features_array = np.array(features).reshape(1, -1)

        # ML model prediction
        ml_result = ml_classifier(features_array)
        ml_label = ml_result.get("label", "error")

        # Gemini prediction
        gemini_result = gemini_classifier(url, signals, local_reasons, line_by_line)
        gemini_label = gemini_result.get("label", "unknown")
        gemini_conf = gemini_result.get("confidence", 0.0)

        # Decision logic
        if gemini_label in {"phishing", "safe"} and gemini_conf >= 0.70:
            final_verdict = gemini_label
            rationale = gemini_result.get("reasons", [])
            original_legit_domain = gemini_result.get("original_legit_domain", "")
        elif ml_label in {"phishing", "safe"}:
            final_verdict = ml_label
            rationale = local_reasons
            original_legit_domain = ""
        else:
            final_verdict = "phishing" if local_risk >= 0.5 else "safe"
            rationale = local_reasons
            original_legit_domain = ""

        # User message
        user_message = gemini_result.get("user_message") or (
            "This link shows multiple risk indicators. Avoid entering credentials."
            if final_verdict == "phishing"
            else "No strong phishing indicators were found, but always double-check before login."
        )

        return {
            "url": url,
            "final_verdict": final_verdict,
            "reasons": rationale,
            "message": user_message,
            "original_legit_domain": original_legit_domain
        }

    except Exception as e:
        return {
            "url": url,
            "final_verdict": "error",
            "reasons": ["Internal prediction error"],
            "message": str(e),
            "original_legit_domain": ""
        }

# -----------------------------
# Optional: formatter for UI
# -----------------------------
def format_user_report(analysis: dict) -> dict:
    """
    Produces a clean JSON-style report for end users.
    Shows final verdict, reasons (simplified), user-friendly message,
    and the original domain for clarity.
    """

    if "error" in analysis:
        return {
            "url": analysis.get("url", ""),
            "verdict": "error",
            "confidence": 0.0,
            "message": analysis["error"],
            "original_domain": ""
        }

    # Extract domain safely
    original_domain = ""
    try:
        parsed = urlparse(analysis.get("url", ""))
        original_domain = parsed.hostname or ""
    except Exception:
        original_domain = ""

    # Prefer Gemini's original_legit_domain if available
    legit_domain = analysis.get("original_legit_domain", "") or original_domain

    # Get reasons (prefer Gemini, else local)
    reasons = analysis.get("reasons", [])
    if not reasons:
        reasons = ["No detailed reasons available, only automated checks were applied."]

    # Ensure each reason is a clear sentence
    reasons = [str(r).strip().rstrip('.') + '.' for r in reasons]

    return {
        "url": analysis.get("url", ""),
        "final_verdict": analysis.get("final_verdict", "unknown"),
        "reasons": reasons,
        "message": analysis.get("message", ""),
        "original_domain": legit_domain
    }




