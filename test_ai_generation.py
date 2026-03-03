"""
Test script specifically for the AI Question Generation feature
"""
import os
import sys
import json
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

def test_ai_generation():
    print("=" * 60)
    print("TEST RUN: AI Question Generation (Google Gemini)")
    print("=" * 60)

    # 1. Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY not found.")
        print("Please ensure you have a .env file with your API key.")
        return

    print(f"\n✅ API Key found: {api_key[:5]}...{api_key[-5:]}")

    # 2. Initialize Client
    try:
        print("Initializing Gemini Client...")
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return

    # 3. Define Mock Data
    position = "Senior Python Developer"
    company = "Tech Innovations Inc."
    resume_snippet = "Experienced in FastAPI, Machine Learning, and Cloud Architecture."
    
    system_prompt = f"""
    You are a Senior Technical Lead and strict interviewer for {company}.
    The candidate is applying for the position of: {position}.
    
    Here is their resume content:
    {resume_snippet}
    
    TASK:
    Generate exactly 3 challenging and in-depth interview questions.
    
    OUTPUT FORMAT:
    Return ONLY a raw JSON list of strings.
    """

    # 4. Run Generation
    print(f"\nGenerating questions for '{position}' at '{company}'...")
    print("Model: gemini-1.5-flash")
    print("-" * 40)

    try:
        # Direct response
        response = client.models.generate_content(
            model='gemini-1.5-flash', 
            contents=system_prompt
        )
        
        print(response.text)
        
        print("-" * 40)
        
        if response.text:
            print("\n✅ Test Passed! Questions generated successfully.")
        else:
            print("\n⚠️ Test Finished, but no text was returned.")

    except Exception as e:
        print(f"\n❌ Generation Error: {e}")
        print("Note: Ensure you are using the correct model name and have internet access.")

if __name__ == "__main__":
    test_ai_generation()