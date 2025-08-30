#!/usr/bin/env python3
"""
Script to convert Firebase JSON credentials to environment variables
"""
import json
import os

def convert_firebase_json_to_env():
    """Convert firebase_key.json to .env file"""
    
    # Check if firebase_key.json exists
    if not os.path.exists("firebase_key.json"):
        print("❌ firebase_key.json not found!")
        print("Please ensure you have your Firebase service account key file.")
        return False
    
    try:
        # Read the JSON file
        with open("firebase_key.json", "r") as f:
            firebase_config = json.load(f)
        
        # Create .env content
        env_content = """# Firebase Configuration
FIREBASE_TYPE={}
FIREBASE_PROJECT_ID={}
FIREBASE_PRIVATE_KEY_ID={}
FIREBASE_PRIVATE_KEY="{}"
FIREBASE_CLIENT_EMAIL={}
FIREBASE_CLIENT_ID={}
FIREBASE_AUTH_URI={}
FIREBASE_TOKEN_URI={}
FIREBASE_AUTH_PROVIDER_X509_CERT_URL={}
FIREBASE_CLIENT_X509_CERT_URL={}
FIREBASE_UNIVERSE_DOMAIN={}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
""".format(
            firebase_config.get("type", ""),
            firebase_config.get("project_id", ""),
            firebase_config.get("private_key_id", ""),
            firebase_config.get("private_key", "").replace("\n", "\\n"),
            firebase_config.get("client_email", ""),
            firebase_config.get("client_id", ""),
            firebase_config.get("auth_uri", ""),
            firebase_config.get("token_uri", ""),
            firebase_config.get("auth_provider_x509_cert_url", ""),
            firebase_config.get("client_x509_cert_url", ""),
            firebase_config.get("universe_domain", "")
        )
        
        # Write .env file
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Successfully created .env file from firebase_key.json")
        print("📝 You can now delete firebase_key.json for security")
        print("🔒 Make sure .env is in your .gitignore (it should be already)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting Firebase config: {e}")
        return False

def main():
    print("🚀 Firebase Environment Setup")
    print("=" * 40)
    
    if convert_firebase_json_to_env():
        print("\n📋 Next steps:")
        print("1. Verify your .env file was created correctly")
        print("2. Delete firebase_key.json (optional but recommended)")
        print("3. Restart your FastAPI server")
        print("4. Test the /health endpoint to verify Firebase connection")
    else:
        print("\n❌ Setup failed. Please check your configuration.")

if __name__ == "__main__":
    main()
