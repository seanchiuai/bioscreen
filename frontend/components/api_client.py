"""API client helpers for the BioScreen Streamlit frontend.

Centralises all HTTP interactions with the FastAPI backend so that
UI code does not contain raw ``requests`` calls.
"""

from typing import Optional

import requests


# Configuration
API_BASE_URL = "http://localhost:8000/api"

DEMO_SEQUENCES = {
    "-- Select a demo sequence --": "",
    # Dangerous — in toxin DB, scores HIGH
    "Scorpion toxin Aah4 (HIGH — known toxin, 84aa)": "MNYLIMFSLALLLVIGVESGRDGYIVDSKNCTYFCGRNAYCNEECTKLKGESGYCQWASPYGNACYCYKLPDHVRTKGPGRCH",
    "Irditoxin snake venom (HIGH — in DB, 109aa)": "MKTLLLAVAVVAFVCLGSADQLGLGRQQIDWGQGQAVGPPYTLCFECNRMTSSDCSTALRCYRGSCYTLYRPDENCELKWAVKGCAETCPTAGPNERVKCCRSPRCNDD",
    "Spider toxin Beta-diguetoxin (HIGH — structural match, 74aa)": "ACVNDDYRSYYCVRKYMECGAEKSVGCWEYKAYQSCYCRQFAYKGEEGRPCVCRDFDGGQALKLHAGKEDSFH",
    # AI-designed evasion — BLAST misses, BioScreen catches
    "AI-redesigned snake venom (MEDIUM — 39% identity, BLAST misses)": "APGRWRCEVWWSAGRCGNQPDAQMYPEKKKQCESPPLSECHKQWNRFDTEYECTSGCWY",
    "AI-redesigned irditoxin (MEDIUM — 23% identity, BLAST misses)": "AAEAAAAEAAAAAAAAAAEAGTAAAPAPPPAAPAAPAPPPITYCYVCNRSLSSDCSTCQPCINGVCYIRYEKNANGEMVPVERGCSATCPTPGPNEKVICCTSDCCNSE",
    # Benign controls — scores LOW
    "Human lysozyme (LOW — benign enzyme, 130aa)": "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
    "GFP (LOW — jellyfish fluorescent protein, 238aa)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Human insulin B chain (LOW — essential hormone, 30aa)": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
}


def check_api_health() -> dict:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Cannot connect to API: {str(e)}"}


def screen_sequence(
    sequence: str,
    session_id: str,
    sequence_id: Optional[str] = None,
    top_k: int = 5
) -> dict:
    """Submit a sequence for screening via the API."""
    payload = {
        "sequence": sequence,
        "sequence_id": sequence_id,
        "top_k": top_k
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/screen",
            json=payload,
            headers={"X-Session-Id": session_id},
            timeout=120,  # Allow time for structure prediction
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}


def get_session_state(session_id: str) -> dict | None:
    """Fetch session state from the API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/session/{session_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def get_session_alerts(session_id: str) -> dict | None:
    """Fetch anomaly alerts for a session."""
    try:
        resp = requests.get(f"{API_BASE_URL}/session/{session_id}/alerts", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None
