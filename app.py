import streamlit as st
import numpy as np
import re
import joblib
from urllib.parse import urlparse

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_models():
    ml_model  = joblib.load('model.pkl')
    scaler    = joblib.load('scaler.pkl')

    with open('best_model_name.txt', 'r') as f:
        model_name = f.read().strip()

    return ml_model, scaler, model_name

ml_model, scaler, model_name = load_models()

def extract_url_features(url):
    url = str(url).strip()
    try:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ''
        path     = parsed.path     or ''
        query    = parsed.query    or ''
    except Exception:
        hostname, path, query = '', '', ''

    tokens = [t for t in re.split(r'\W+', url) if t]

    return {
        'url_length':       len(url),
        'hostname_length':  len(hostname),
        'path_length':      len(path),
        'query_length':     len(query),
        'num_dots':         url.count('.'),
        'num_hyphens':      url.count('-'),
        'num_slashes':      url.count('/'),
        'num_at':           url.count('@'),
        'num_question':     url.count('?'),
        'num_equals':       url.count('='),
        'num_ampersand':    url.count('&'),
        'num_percent':      url.count('%'),
        'num_digits':       sum(c.isdigit() for c in url),
        'num_subdomains':   hostname.count('.'),
        'url_depth':        len([p for p in path.split('/') if p]),
        'digit_ratio':      sum(c.isdigit() for c in url) / (len(url) + 1),
        'letter_ratio':     sum(c.isalpha() for c in url) / (len(url) + 1),
        'has_https':        int(url.lower().startswith('https')),
        'has_ip':           int(bool(re.match(
                                r'^https?://\d{1,3}(\.\d{1,3}){3}', url))),
        'has_double_slash': int('//' in path),
        'has_hex_encoding': int('%' in url and
                                bool(re.search(r'%[0-9a-fA-F]{2}', url))),
        'has_shortener':    int(bool(re.search(
                                r'(bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|'
                                r'is\.gd|buff\.ly|adf\.ly)', url))),
        'longest_word':     max((len(w) for w in tokens), default=0),
    }

def predict(url):
    feats  = extract_url_features(url)
    arr    = np.array([list(feats.values())], dtype=np.float32)
    arr_sc = scaler.transform(arr)
    prob   = float(ml_model.predict_proba(arr_sc)[0][1])
    return prob

# ── UI ────────────────────────────────────────────────────────────
st.title("🔍 Phishing URL Detector")
st.markdown("Paste any URL below to check if it looks suspicious.")

url_input = st.text_input(
    label="Enter URL",
    placeholder="https://example.com",
    label_visibility="collapsed"
)

check_btn = st.button("Check URL", type="primary", use_container_width=True)

if check_btn and url_input.strip():
    with st.spinner("Analysing..."):
        try:
            prob = predict(url_input.strip())

            st.divider()

            if prob > 0.5:
                st.error("⚠️  This URL looks like PHISHING")
            else:
                st.success("✅  This URL looks Legitimate")

            st.markdown("#### Model Confidence")
            st.metric(label=model_name, value=f"{prob:.1%}")
            st.progress(float(prob))

            with st.expander("Show URL feature analysis"):
                import pandas as pd
                feat_df = pd.DataFrame(
                    extract_url_features(url_input.strip()).items(),
                    columns=['Feature', 'Value']
                )
                st.dataframe(
                    feat_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:
            st.error(f"Error: {e}")

elif check_btn and not url_input.strip():
    st.warning("Please enter a URL first.")

st.divider()
st.caption(
    "Trained on PhiUSIIL dataset. "
    "This tool is for educational purposes — always verify with other sources."
)