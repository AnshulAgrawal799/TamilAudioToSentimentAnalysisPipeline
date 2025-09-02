#!/usr/bin/env python3
"""
Setup script for Google Cloud credentials.
This script helps configure Google Cloud credentials for better translation quality.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_google_cloud_installed():
    """Check if Google Cloud SDK is installed."""
    try:
        result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def setup_google_cloud_credentials():
    """Setup Google Cloud credentials for translation API."""
    print("Setting up Google Cloud credentials for better translation quality...")
    
    # Check if gcloud is installed
    if not check_google_cloud_installed():
        print("❌ Google Cloud SDK not found. Please install it first:")
        print("   https://cloud.google.com/sdk/docs/install")
        return False
    
    # Check if already authenticated
    try:
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], 
                              capture_output=True, text=True)
        if 'ACTIVE' in result.stdout:
            print("✅ Google Cloud authentication already active")
            return True
    except Exception:
        pass
    
    # Authenticate with Google Cloud
    print("🔐 Authenticating with Google Cloud...")
    try:
        subprocess.run(['gcloud', 'auth', 'login'], check=True)
        print("✅ Google Cloud authentication successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Google Cloud authentication failed: {e}")
        return False

def setup_application_default_credentials():
    """Setup Application Default Credentials (ADC)."""
    print("🔑 Setting up Application Default Credentials...")
    
    try:
        subprocess.run(['gcloud', 'auth', 'application-default', 'login'], check=True)
        print("✅ Application Default Credentials configured successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup Application Default Credentials: {e}")
        return False

def create_env_file():
    """Create .env file with Google Cloud configuration."""
    env_content = """# Google Cloud Configuration
# This file contains environment variables for Google Cloud services

# Google Cloud Project ID (replace with your project ID)
GOOGLE_CLOUD_PROJECT=your-project-id-here

# Google Cloud Translate API Configuration
GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json

# Translation settings
USE_GOOGLE_TRANSLATE=true
TRANSLATION_QUALITY=high

# Optional: Service account key file (if using service account)
# GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
"""
    
    env_file = Path('.env')
    if env_file.exists():
        print("⚠️  .env file already exists. Backing up as .env.backup")
        env_file.rename('.env.backup')
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with Google Cloud configuration")
    print("📝 Please edit .env file and set your Google Cloud Project ID")

def test_google_translate():
    """Test Google Cloud Translate API."""
    print("🧪 Testing Google Cloud Translate API...")
    
    try:
        from google.cloud import translate_v2 as translate
        
        # Test translation
        client = translate.Client()
        result = client.translate('வணக்கம்', source_language='ta', target_language='en')
        
        if result and result['translatedText']:
            print(f"✅ Translation test successful: 'வணக்கம்' -> '{result['translatedText']}'")
            return True
        else:
            print("❌ Translation test failed: No result received")
            return False
            
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Google Cloud Credentials Setup")
    print("=" * 50)
    
    # Step 1: Setup authentication
    if not setup_google_cloud_credentials():
        print("❌ Setup failed at authentication step")
        return False
    
    # Step 2: Setup Application Default Credentials
    if not setup_application_default_credentials():
        print("❌ Setup failed at ADC step")
        return False
    
    # Step 3: Create .env file
    create_env_file()
    
    # Step 4: Test translation
    if test_google_translate():
        print("\n🎉 Google Cloud setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and set your Google Cloud Project ID")
        print("2. Enable Google Cloud Translate API in your project")
        print("3. Run the pipeline again for better translation quality")
        return True
    else:
        print("\n⚠️  Setup completed but translation test failed")
        print("Please check your Google Cloud project settings")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
