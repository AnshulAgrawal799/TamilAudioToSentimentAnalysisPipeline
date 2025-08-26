#!/usr/bin/env python3
"""
Setup script for Google Cloud Translate API.
This script helps configure the Google Cloud Translate API for better translation quality.
"""

import os
import json
from pathlib import Path

def setup_google_cloud_translate():
    """Setup Google Cloud Translate API configuration."""
    print("Setting up Google Cloud Translate API for better translation quality...")
    print()
    
    # Check if credentials file exists
    credentials_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if credentials_path.exists():
        print(f"✓ Found Google Cloud credentials at: {credentials_path}")
        print("  You can now use Google Cloud Translate API.")
    else:
        print("✗ Google Cloud credentials not found.")
        print()
        print("To use Google Cloud Translate API (recommended for production):")
        print("1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        print("2. Run: gcloud auth application-default login")
        print("3. Enable Cloud Translate API: https://console.cloud.google.com/apis/library/translate.googleapis.com")
        print("4. Set up billing for your Google Cloud project")
        print()
        print("For now, the system will use googletrans fallback (lower quality).")
    
    print()
    print("Alternative: Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    print("export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json")
    
    # Create .env file template
    env_template = """# Google Cloud Translate API Configuration
# Uncomment and set these variables if using service account authentication

# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# GOOGLE_CLOUD_PROJECT=your-project-id

# Translation quality settings
TRANSLATION_USE_GOOGLE_CLOUD=true
TRANSLATION_FALLBACK_TO_GOOGLETRANS=true
TRANSLATION_BATCH_SIZE=10
TRANSLATION_RETRY_ATTEMPTS=3
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_template)
        print(f"✓ Created .env template file: {env_file}")
    
    print()
    print("Setup complete! The system will automatically detect and use the best available translation service.")

if __name__ == "__main__":
    setup_google_cloud_translate()
