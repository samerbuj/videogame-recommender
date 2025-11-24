import requests

CLIENT_ID = "xvwunh7y0a10ew27jcvji9oicxlkgj"
CLIENT_SECRET = "3t1i06bm27fmg22drjbpddjosfk461"

TOKEN_URL = "https://id.twitch.tv/oauth2/token"

def get_auth_header():
    print("Requesting OAuth token from Twitch...")
    token_resp = requests.post(
        TOKEN_URL,
        params={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
    )
    token_resp.raise_for_status()
    token_data = token_resp.json()
    access_token = token_data["access_token"]
    print("Got access token.")

    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}",
    }

    return headers