# SpeakGenie - Real-Time AI Voice Tutor

🧞‍♂️ SpeakGenie is a real-time AI-powered voice tutor built as a technical internship task. It helps children aged 6 to 16 practice English through interactive, voice-based learning in a fun, safe environment.

---

## 🌟 Features

- 🎤 **Real-time Voice Interaction** – Speak and get spoken responses from an AI tutor.
- 🔄 **Two Learning Modes**:
  1. **Free-flow AI Chatbot** – Open conversation to encourage spontaneous speaking.
  2. **Interactive Roleplay** – Scenario-based practice (school, store, home).
- 🧒 **Child-Safe AI** – Friendly, encouraging, and age-appropriate responses.
- 🌍 **Native Language Playback** – Translates AI replies into Hindi, Marathi, Gujarati, etc.
- 💻 **Modern UI** – Clean, responsive interface with speech and visual feedback.

---

## 🛠 Technology Stack

| Component  | Tech Used                        |
|------------|----------------------------------|
| Backend    | Python, Flask                    |
| Frontend   | HTML, CSS, JS (Web Speech API)   |
| AI Model   | Gemini API                       |
| Translation| Google Translate API (mocked)    |

---

## ⚙️ Setup Instructions

### 🔗 Prerequisites

- Python 3.8+
- `pip` installed
- API key for **Google Gemini**
- Optional: `.env` file to securely store API keys

---

### 📦 Installation

1. **Clone the project**
```bash
git clone <https://github.com/Raiyan1509/SpeakGenie>
cd speakgenie-project

pip install Flask openai python-dotenv google-generativeai
## To Run
python app.py


speakgenie-project/
│
├── app.py                # Flask backend
├── templates/
│   └── index.html        # UI
├── .env                  # API keys (not committed)
└── README.md             # Project info
