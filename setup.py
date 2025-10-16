#!/usr/bin/env python3
"""
Setup script for the AI research idea generation system.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "generated_ideas",
        "generated_ideas/raw_responses",
        "generated_ideas/processed_ideas", 
        "generated_ideas/comparisons"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_config_file():
    """Create config file if it doesn't exist"""
    config_file = ".env"
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            f.write("""# AI Model API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Optional: Model preferences
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2000
""")
        print(f"✅ Created config file: {config_file}")
        print("⚠️  Please edit config.env and add your API keys!")
    else:
        print(f"✅ Config file already exists: {config_file}")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Research Idea Generation System")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return
    
    # Create directories
    create_directories()
    
    # Create config file
    create_config_file()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit config.env and add your API keys")
    print("2. Run: python generate_research_ideas.py")
    print("3. Run: python compare_ideas.py")

if __name__ == "__main__":
    main()
