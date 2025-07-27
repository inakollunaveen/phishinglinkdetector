import re
import tldextract
import numpy as np
from urllib.parse import urlparse

def transform_url(url):
    """
    Extracts 8 numerical features from a given URL for phishing detection.
    Returns a 1D NumPy array of shape (8,)
    """

    # 1. Length of the URL
    url_length = len(url)

    # 2. Presence of '@' symbol (used in phishing to obscure real domain)
    has_at = 1 if "@" in url else 0

    # 3. Count of dots in URL (subdomains, phishing trick)
    dot_count = url.count('.')

    # 4. Count of hyphens (often used to trick users)
    hyphen_count = url.count('-')

    # 5. Presence of IP address instead of domain
    ip_pattern = r"(http|https)://(\d{1,3}\.){3}\d{1,3}"
    has_ip = 1 if re.search(ip_pattern, url) else 0

    # 6. Suspicious TLDs (.xyz, .top, etc.)
    suspicious_tlds = ['.xyz', '.top', '.club', '.online', '.work']
    tld = '.' + tldextract.extract(url).suffix
    is_suspicious_tld = 1 if tld in suspicious_tlds else 0

    # 7. HTTPS used or not
    has_https = 1 if url.startswith("https://") else 0

    # 8. Number of suspicious keywords
    suspicious_keywords = ['login', 'verify', 'update', 'secure', 'bank', 'account']
    keyword_count = sum(1 for word in suspicious_keywords if word in url.lower())

    return np.array([
        url_length,       # Feature 1
        has_at,           # Feature 2
        dot_count,        # Feature 3
        hyphen_count,     # Feature 4
        has_ip,           # Feature 5
        is_suspicious_tld,# Feature 6
        has_https,        # Feature 7
        keyword_count     # Feature 8
    ])
