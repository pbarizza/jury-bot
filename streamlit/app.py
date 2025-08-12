# streamlit/app.py
import streamlit as st
import requests
import os

# -------------------------------
# Configuration
# -------------------------------
# Use Streamlit secrets in production, .env in dev
try:
    # Running in Streamlit Cloud (GitHub Secrets)
    BACKEND_URL = st.secrets["BACKEND_URL"]
except KeyError:
    # Fallback to .env (for local dev)
    from dotenv import load_dotenv
    load_dotenv()
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/query")

# -------------------------------
# Page Config (Dark Mode 🌙)
# -------------------------------
st.set_page_config(
    page_title="🤖 Jury-Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and chat style
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        max-width: 80%;
        line-height: 1.5;
    }
    .user {
        background-color: #1f6feb;
        color: white;
        margin-left: auto;
        margin-right: 0;
    }
    .bot {
        background-color: #2d333e;
        color: #c9d1d9;
        margin-right: auto;
        margin-left: 0;
        border: 1px solid #3c4250;
    }
    .stTextInput > div > div > input {
        background-color: #161b22;
        color: white;
        border: 1px solid #30363d;
    }
    .title {
        color: #58a6ff;
        text-align: center;
        font-size: 2.2rem;
    }
    .subtitle {
        color: #8b949e;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<h1 class="title">🤖 Jury-Bot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-Powered Startup Judge</p>', unsafe_allow_html=True)
st.markdown('<h3 class="debug">{BACKEND_URL}</h3>', unsafe_allow_html=True)

# -------------------------------
# Initialize Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I’m Jury-Bot. Ask me about any startup, their pitch, or ask me to rank them."}
    ]

# -------------------------------
# Display Chat Messages
# -------------------------------
for message in st.session_state.messages:
    css_class = "user" if message["role"] == "user" else "bot"
    st.markdown(f'<div class="chat-message {css_class}">{message["content"]}</div>', unsafe_allow_html=True)

# -------------------------------
# User Input
# -------------------------------
if prompt := st.chat_input("Ask about startups, domains, funding, or rankings..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)

    # Show "thinking" spinner
    with st.spinner("Jury-Bot is evaluating..."):
        try:
            response = requests.post(
                BACKEND_URL,
                json={"question": prompt, "top_k": 3},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    # Extract and format answer
                    answers = []
                    for r in data["results"]:
                        answers.append(
                            f"**{r['startup_name']}** ({r['domain']})\n"
                            f"- Funding: {r['funding_stage']}\n"
                            f"- Team: {r['team_size']} members\n"
                            f"- Summary: {r['description'][:150]}..."
                        )
                    bot_response = "I found:\n\n" + "\n\n".join(answers)
                else:
                    bot_response = "I couldn't find any startups matching your query."
            else:
                bot_response = f"⚠️ API error: {response.status_code}"
        except requests.exceptions.ConnectionError:
            bot_response = "❌ Cannot reach the API. Is your FastAPI server running?"
        except Exception as e:
            bot_response = f"❌ Error: {str(e)}"

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.markdown(f'<div class="chat-message bot">{bot_response}</div>', unsafe_allow_html=True)
