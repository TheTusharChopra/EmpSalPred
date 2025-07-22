import streamlit as st
import subprocess
import time
import threading
from pyngrok import ngrok
import os

# Setup ngrok
def setup_ngrok():
    """Setup ngrok tunnel for Streamlit"""
    # Kill any existing ngrok processes
    try:
        ngrok.kill()
    except:
        pass
    
    # Create tunnel
    public_url = ngrok.connect(8501)
    print(f"🌐 Streamlit app is available at: {public_url}")
    return public_url

# Run this if you want to setup ngrok automatically
if __name__ == "__main__":
    # Setup ngrok tunnel
    url = setup_ngrok()
    print(f"✅ Your app will be accessible at: {url}")
    print("⚠️  Keep this script running to maintain the tunnel")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ngrok tunnel...")
        ngrok.kill()