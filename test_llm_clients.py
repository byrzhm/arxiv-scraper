#!/usr/bin/env python
"""
Test script for verifying LLM clients functionality.
"""

import os
import sys
from dotenv import load_dotenv
from llm_clients import get_llm_client

def test_llm_client(provider_name):
    """Test a specific LLM provider client."""
    try:
        print(f"Testing {provider_name} client...")
        client = get_llm_client(provider_name)
        
        # Simple prompt for testing
        prompt = "Explain the concept of AI infrastructure in one paragraph."
        print("Sending test prompt...")
        response = client.generate_completion(prompt)
        
        print(f"\n--- Response from {provider_name.capitalize()} ---")
        print(response)
        print("-----------------------------------\n")
        return True
    except ImportError as e:
        print(f"Missing dependency for {provider_name}: {e}")
        return False
    except ValueError as e:
        print(f"Configuration error for {provider_name}: {e}")
        return False
    except Exception as e:
        print(f"Error testing {provider_name}: {e}")
        return False

def main():
    """Test all configured LLM providers."""
    load_dotenv()
    
    if len(sys.argv) > 1:
        # Test specific provider
        provider = sys.argv[1].lower()
        test_llm_client(provider)
    else:
        # Test all providers that have API keys configured
        providers = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "qwen": os.getenv("QWEN_API_KEY")
        }
        
        available_providers = [p for p, key in providers.items() if key]
        
        if not available_providers:
            print("No API keys configured in .env file. Please add at least one API key.")
            return
        
        print(f"Found API keys for: {', '.join(available_providers)}")
        
        for provider in available_providers:
            success = test_llm_client(provider)
            if not success:
                print(f"Failed to test {provider}. Check your API key and dependencies.")

if __name__ == "__main__":
    main()
