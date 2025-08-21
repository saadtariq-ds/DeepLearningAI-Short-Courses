import os
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

def authenticate():
    # Load .env
    # return "DLAI-credentials", "DLAI_PROJECT"

    load_dotenv()

    key_path = 'spherical-jetty-465410-u7-ec994a20970e.json'

    # Create credentials based on key from service account
    # Make sure your account has the roles listed in the Google Cloud Setup section
    credentials = Credentials.from_service_account_file(
        key_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    if credentials.expired:
        credentials.refresh(Request())

    # Set project ID accoridng to environment variable
    PROJECT_ID = os.getenv('PROJECT_ID')

    return credentials, PROJECT_ID