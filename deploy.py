# deploy.py
import subprocess
import threading
import time
import os
from pyngrok import ngrok
from getpass import getpass

def setup_ngrok(auth_token):
    """Setup ngrok with authentication"""
    try:
        ngrok.set_auth_token(auth_token)
        print("✅ Ngrok authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Ngrok authentication failed: {e}")
        return False

def run_streamlit():
    """Run the Streamlit app on port 8505"""
    try:
        print("🚀 Starting Streamlit app on port 8505...")

        # Run Streamlit
        process = subprocess.Popen([
            "streamlit", "run", "app.py",
            "--server.port", "8505",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ])

        return process
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return None

def check_streamlit_ready(port=8505, timeout=30):
    """Check if Streamlit is ready to accept connections"""
    import requests
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit app is ready!")
                return True
        except:
            pass
        time.sleep(2)

    print("❌ Streamlit app failed to start within timeout period")
    return False

def deploy_app():
    """Main deployment function"""
    print("🎯 AI Financial Advisor Deployment")
    print("=" * 50)

    # Get ngrok auth token
    auth_token = getpass("Enter your ngrok auth token: ")

    if not auth_token or auth_token.strip() == "":
        print("❌ No auth token provided. Exiting.")
        return

    # Setup ngrok
    if not setup_ngrok(auth_token):
        return

    # Start Streamlit
    process = run_streamlit()
    if not process:
        return

    # Wait for app to start
    print("⏳ Waiting for Streamlit app to start...")
    time.sleep(8)

    # Check if app is ready
    if not check_streamlit_ready():
        print("❌ Streamlit app failed to start properly")
        process.terminate()
        return

    try:
        # Create ngrok tunnel
        print("🔗 Creating ngrok tunnel...")
        public_url = ngrok.connect(8505, bind_tls=True)

        print("\n" + "=" * 50)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)
        print(f"🌐 Your Financial Advisor is now live at:")
        print(f"🔗 {public_url}")
        print("\n📱 Share this URL with anyone to access your app!")
        print("=" * 50)

        # Display app information
        print("\n💡 App Information:")
        print(f"   - Port: 8505")
        print(f"   - Local URL: http://localhost:8505")
        print(f"   - Public URL: {public_url}")
        print(f"   - Status: ✅ Running")

        # Keep the app running
        print("\n🔄 App is running... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")

    except Exception as e:
        print(f"❌ Error creating ngrok tunnel: {e}")

    finally:
        # Cleanup
        if process:
            process.terminate()
        ngrok.kill()
        print("✅ App stopped successfully!")

def quick_deploy(auth_token):
    """Quick deployment without user input"""
    print("🚀 Starting quick deployment...")

    if not setup_ngrok(auth_token):
        return False

    process = run_streamlit()
    if not process:
        return False

    time.sleep(8)

    try:
        public_url = ngrok.connect(8505)
        print(f"✅ App deployed at: {public_url}")

        # Display deployment info
        print("\n" + "=" * 50)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)
        print(f"🌐 Your Financial Advisor is now live at:")
        print(f"🔗 {public_url}")
        print("\n📱 Share this URL with anyone to access your app!")
        print("=" * 50)

        # Keep alive
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        process.terminate()
        ngrok.kill()
        print("✅ App stopped")
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        process.terminate()
        return False

if __name__ == "__main__":
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        print("🔍 Detected Google Colab environment")

        # For Colab, use the token you provided
        YOUR_TOKEN = "************************************************"

        if YOUR_TOKEN == "YOUR_NGROK_AUTH_TOKEN_HERE":
            print("❌ Please set your ngrok auth token in the code!")
            print("🔑 Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        else:
            quick_deploy(YOUR_TOKEN)
    else:
        # For local deployment, use interactive mode
        deploy_app()