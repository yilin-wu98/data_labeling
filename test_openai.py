import os
import json
import openai
from openai import OpenAI

def load_api_key(provider: str) -> str:
    with open("api_config.json") as f:
        config = json.load(f)
    api_key = config[provider]["api_key"]
    print(f"Loaded API key for {provider}: {api_key[:10]}...")
    return api_key

def test_openai_client():
    try:
        # Method 1: Direct initialization
        print("Testing direct initialization...")
        api_key = load_api_key("openai")
        client1 = OpenAI(api_key=api_key)
        print("✓ Direct initialization successful")
    except Exception as e:
        print(f"✗ Direct initialization failed: {e}")
    
    try:
        # Method 2: Environment variable
        print("Testing environment variable initialization...")
        api_key = load_api_key("openai")
        os.environ["OPENAI_API_KEY"] = api_key
        client2 = OpenAI()
        print("✓ Environment variable initialization successful")
    except Exception as e:
        print(f"✗ Environment variable initialization failed: {e}")
    
    try:
        # Method 3: Using openai.api_key
        print("Testing openai.api_key initialization...")
        api_key = load_api_key("openai")
        openai.api_key = api_key
        client3 = OpenAI()
        print("✓ openai.api_key initialization successful")
    except Exception as e:
        print(f"✗ openai.api_key initialization failed: {e}")

if __name__ == "__main__":
    test_openai_client() 