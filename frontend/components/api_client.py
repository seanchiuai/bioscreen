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
    # Dangerous proteins with low BLAST identity to known toxins (demonstrates BioScreen advantage)
    "Trichosanthin (RIP toxin, ~28% identity to ricin, 289aa)": "DVSFRLSGATSSSYGVFISNLRKALPNERKLYDIPLLRSSLPGSQRYALIHLTNYADETISVAIDVTNVYIMGYRAGDTSYFFNEASATEAAKYVFKDAMRKVTLPYSGNYERLQTAAGKIRENIPLGLPALDSAITTLFYYNANSAASALMVLIQSTSEAARYKFIEQQIGKRVDKTFLPSLAIISLENSWSALSKQIQIASTNNGQFESPVVLINAQNQRVTITNVDAGVVTSNIALLLNRNNMAAMDDDVPMTQSFGCGSYAI",
    "Saporin (RIP toxin, ~30% identity to ricin, 253aa)": "DAVTSITLDLVNPTAGQYSSFVDKIRNNVKDPNLKYGGTDIAVIGPPSKEKFLRINFQSSRGTVSLGLKRDNLYVVAYLAMDNTNVNRAYYFRSEITSAESTALFPEATTANQKALEYTEDYQSIEKNAQITQGDQSRKELGLGIDLLSTSMEAVNKKARVVKDEARFLLIAIQMTAEAARFRYIQNLVIKNFPNKFNSENKVIQFEVNWKKISTAIYGDAKNGVFNKDYDFGFGKVRQVKDLQMGLLMYLGKPKSSNEANSTVRHYGPLKPTLLIT",
    "MAP30 (RIP toxin, bitter melon, 263aa)": "DVNFDLSTATAKTYTKFIEDFRATLPFSHKVYDIPLLYSTISDSRRFILLNLTSYAYETISVAIDVTNVYVVAYRTRDVSYFFKESPPEAYNILFKGTRKITLPYTGNYENLQTAAHKIRENIDLGLPALSSAITTLFYYNAQSAPSALLVLIQTTAEAARFKYIERHVAKYVATNFKPNLAIISLENQWSALSKQIFLAQNQGGKFRNPVDLIKPTGERFQVTNVDSDVVKGNIKLLLNSRASTADENFITTMTLLGESVVN",
    "Bouganin (RIP toxin, ~32% identity to ricin, 70aa)": "EFQESVKSQHTERCIDFLTKELKVSNEKEAAERVFFVSARETLQARLEEAKGNPPHLGAIAEGFQIRYFE",
    "Cholix toxin (ADP-ribosyltransferase, <20% identity to DT, 95aa)": "YPTKGRGGKGIKTANITAKNGPLAGLVTVNDDEDIMIITDTGVIIRTSVADISQTGRSAMGVKVMRLDENAKIVTFALVKSEVIEGTSLNNNENE",
    # Benign controls
    "GFP (benign control, 238aa)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Hemoglobin alpha (benign control, 141aa)": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLAS",
}


def check_api_health() -> dict:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
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
