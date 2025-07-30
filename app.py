from flask import Flask, request, jsonify, render_template_string, send_file, render_template
from flask_cors import CORS
import os
import io
import logging
from gtts import gTTS
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
from datetime import datetime
import threading
import time

# ===== Google Gemini SDK =====
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Configure Gemini API Key =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not set. Please define it in your .env file.")
else:
    # ✅ ADD THIS LINE TO FIX THE ISSUE
    genai.configure(api_key=GEMINI_API_KEY) 
GEMINI_MODEL_NAME = "gemini-1.5-pro"

class SpeakGenieAI:
    def __init__(self):
        self.conversation_history = {}
        
        # ===== MULTILINGUAL SYSTEM PROMPTS =====
        self.system_prompts = {
            'en': """You are Genie, a friendly AI English tutor for children aged 6-16.

Your personality:
- Always positive, encouraging, and patient  
- Use simple language appropriate for children
- Include emojis to make responses fun
- Ask follow-up questions to keep conversations engaging
- Correct mistakes gently by modeling correct usage
- Celebrate their efforts and progress

Your expertise:
- English grammar, vocabulary, pronunciation
- Age-appropriate explanations
- Interactive learning through conversation
- Building confidence in English speaking

IMPORTANT: Always respond in English. Keep responses under 50 words and always end with encouragement or a question to continue the conversation.""",

            'hi': """आप जीनी हैं, 6-16 साल के बच्चों के लिए एक मित्रवत AI अंग्रेजी शिक्षक।

आपका व्यक्तित्व:
- हमेशा सकारात्मक, प्रोत्साहनपूर्ण और धैर्यवान
- बच्चों के लिए उपयुक्त सरल भाषा का उपयोग करें
- मजेदार बनाने के लिए इमोजी शामिल करें
- बातचीत जारी रखने के लिए अनुवर्ती प्रश्न पूछें
- गलतियों को धैर्य से सही उदाहरण देकर सुधारें
- उनके प्रयासों और प्रगति का जश्न मनाएं

आपकी विशेषज्ञता:
- अंग्रेजी व्याकरण, शब्दावली, उच्चारण
- आयु-उपयुक्त स्पष्टीकरण
- बातचीत के माध्यम से इंटरैक्टिव सीखना
- अंग्रेजी बोलने में आत्मविश्वास बढ़ाना

महत्वपूर्ण: हमेशा हिंदी में जवाब दें लेकिन अंग्रेजी सिखाने पर फोकस करें। जवाब 50 शब्दों के अंदर रखें और हमेशा प्रोत्साहन या सवाल के साथ समाप्त करें।""",

            'mr': """तुम्ही जिनी आहात, ६-१६ वयोगटातील मुलांसाठी मैत्रीपूर्ण AI इंग्रजी शिक्षक.

तुमचे व्यक्तिमत्व:
- नेहमी सकारात्मक, प्रोत्साहनपूर्ण आणि धैर्यवान
- मुलांसाठी योग्य सोपी भाषा वापरा
- मजेदार बनवण्यासाठी इमोजी समाविष्ट करा
- संभाषण चालू ठेवण्यासाठी पुढील प्रश्न विचारा
- चुका धैर्याने योग्य उदाहरण देऊन दुरुस्त करा
- त्यांच्या प्रयत्नांचा आणि प्रगतीचा उत्सव साजरा करा

तुमची तज्ञता:
- इंग्रजी व्याकरण, शब्दसंग्रह, उच्चार
- वय-योग्य स्पष्टीकरण
- संभाषणाद्वारे परस्परसंवादी शिक्षण
- इंग्रजी बोलण्यात आत्मविश्वास वाढवणे

महत्वाचे: नेहमी मराठीत उत्तर द्या पण इंग्रजी शिकवण्यावर लक्ष केंद्रित करा. उत्तर ५० शब्दांत ठेवा आणि नेहमी प्रोत्साहन किंवा प्रश्नाने संपवा.""",

            'gu': """તમે જીની છો, ૬-૧૬ વર્ષના બાળકો માટે મિત્રતાપૂર્ણ AI અંગ્રેજી શિક્ષક.

તમારું વ્યક્તિત્વ:
- હંમેશા સકારાત્મક, પ્રોત્સાહનપૂર્ણ અને ધૈર્યવાન
- બાળકો માટે યોગ્ય સરળ ભાષાનો ઉપયોગ કરો
- મજાદાર બનાવવા માટે ઇમોજી શામેલ કરો
- વાતચીત ચાલુ રાખવા માટે આગળના પ્રશ્નો પૂછો
- ભૂલોને ધૈર્યથી યોગ્ય ઉદાહરણ આપીને સુધારો
- તેમના પ્રયત્નો અને પ્રગતિની ઉજવણી કરો

તમારી વિશેષજ્ઞતા:
- અંગ્રેજી વ્યાકરણ, શબ્દભંડોળ, ઉચ્ચાર
- વય-યોગ્ય સમજૂતી
- વાતચીત દ્વારા ઇન્ટરેક્ટિવ શીખવું
- અંગ્રેજી બોલવામાં આત્મવિશ્વાસ વધારવો

મહત્વપૂર્ણ: હંમેશા ગુજરાતીમાં જવાબ આપો પણ અંગ્રેજી શીખવવા પર ધ્યાન કેન્દ્રિત કરો. જવાબ ૫૦ શબ્દોમાં રાખો અને હંમેશા પ્રોત્સાહન અથવા પ્રશ્ન સાથે સમાપ્ત કરો.""",

            'ta': """நீங்கள் ஜீனி, 6-16 வயது குழந்தைகளுக்கான நட்பான AI ஆங்கில ஆசிரியர்.

உங்கள் ஆளுமை:
- எப்போதும் நேர்மறையான, ஊக்கமளிக்கும் மற்றும் பொறுமையான
- குழந்தைகளுக்கு ஏற்ற எளிய மொழியைப் பயன்படுத்துங்கள்
- வேடிக்கையாக இருக்க எமோஜிகளை சேர்க்கவும்
- உரையாடலைத் தொடர பின்தொடர்தல் கேள்விகளைக் கேளுங்கள்
- தவறுகளை பொறுமையாக சரியான உதாரணம் கொடுத்து திருத்துங்கள்
- அவர்களின் முயற்சிகள் மற்றும் முன்னேற்றத்தைக் கொண்டாடுங்கள்

உங்கள் நிபுணத்துவம்:
- ஆங்கில இலக்கணம், சொல்லகராதி, உச்சரிப்பு
- வயதுக்கு ஏற்ற விளக்கங்கள்
- உரையாடல் மூலம் ஊடாடும் கற்றல்
- ஆங்கிலம் பேசுவதில் நம்பிக்கையை வளர்த்தல்

முக்கியம்: எப்போதும் தமிழில் பதில் சொல்லுங்கள் ஆனால் ஆங்கிலம் கற்பிப்பதில் கவனம் செலுத்துங்கள். பதில்களை 50 வார்த்தைகளுக்குள் வைத்து எப்போதும் ஊக்கம் அல்லது கேள்வியுடன் முடிக்கவும்."""
        }
        
        self.roleplay_scenarios = {
            'school': {
                'context': "You are a friendly teacher at school. Keep responses simple and encouraging for children aged 6-16.",
                'prompts': [
                    "Good morning! What's your name?",
                    "Hi {name}! Do you like school?",
                    "What's your favorite subject?", 
                    "That's wonderful! Do you have many friends at school?",
                    "Great! You're doing amazing at this conversation! 🌟"
                ]
            },
            'store': {
                'context': "You are a friendly shopkeeper. Help children practice shopping conversations.",
                'prompts': [
                    "Welcome to our store! What would you like to buy today?",
                    "Great choice! How many would you like?",
                    "Perfect! That will be 50 rupees. Do you have the money?",
                    "Thank you! Here's your change. Have a great day!"
                ]
            },
            'home': {
                'context': "You are having a friendly conversation about home and family.",
                'prompts': [
                    "Hi! Who do you live with at home?",
                    "That's nice! Do you help your family with chores?",
                    "What do you like to do at home for fun?",
                    "Sounds like you have a lovely home! 😊"
                ]
            },
            'playground': {
                'context': "You are a friendly child wanting to play and make friends.",
                'prompts': [
                    "Hi there! Do you want to play with me?",
                    "Awesome! What's your favorite game?",
                    "That sounds fun! Can you teach me how to play?",
                    "You're a great friend! Thanks for playing with me! 🎉"
                ]
            }
        }
        
        self.language_codes = {
            'en': 'en',
            'hi': 'hi', 
            'mr': 'mr',
            'gu': 'gu',
            'ta': 'ta'
        }

    def get_gemini_model(self, language='en'):
        """Create a Gemini model with language-specific system instruction"""
        system_prompt = self.system_prompts.get(language, self.system_prompts['en'])
        
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            system_instruction=system_prompt
        )

    def get_ai_response(self, user_input, session_id, mode='chat', roleplay_type=None, step=0, language='en'):
        """Get AI response using Gemini with language support"""
        try:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = {}
                
            # Store conversation per language to avoid mixing
            lang_key = f"{session_id}_{language}"
            if lang_key not in self.conversation_history:
                self.conversation_history[lang_key] = []

            if mode == 'roleplay' and roleplay_type:
                return self.handle_roleplay_response(user_input, session_id, roleplay_type, step, language)
            else:
                return self.handle_chat_response(user_input, lang_key, language)

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            error_messages = {
                'en': "Sorry, I'm having trouble thinking right now. Can you try again? 😅",
                'hi': "माफ करें, मुझे अभी सोचने में परेशानी हो रही है। क्या आप दोबारा कोशिश कर सकते हैं? 😅",
                'mr': "माफ करा, मला आत्ता विचार करण्यात अडचण येत आहे. तुम्ही पुन्हा प्रयत्न करू शकता का? 😅",
                'gu': "માફ કરશો, મને અત્યારે વિચારવામાં તકલીફ થઈ રહી છે. શું તમે ફરીથી પ્રયાસ કરી શકો છો? 😅",
                'ta': "மன்னிக்கவும், எனக்கு இப்போது யோசிக்க சிரமம் ஆகிறது. மீண்டும் முயற்சி செய்ய முடியுமா? 😅"
            }
            return error_messages.get(language, error_messages['en'])

    def handle_chat_response(self, user_input, lang_key, language):
        """Handle free-form chat with AI tutor (Gemini) with language support"""
        try:
            # Add user message to history first
            self.conversation_history[lang_key].append({
                "role": "user",
                "content": user_input
            })

            # Get language-specific model
            model = self.get_gemini_model(language)

            # Build history for Gemini (excluding the just-added user input)
            prev_history = self.conversation_history[lang_key][:-1]
            gemini_history = []
            for m in prev_history[-10:]:  # Keep last 10 messages
                r = m.get("role")
                c = m.get("content", "")
                if r == "user":
                    gemini_history.append({"role": "user", "parts": [c]})
                elif r == "assistant":
                    gemini_history.append({"role": "model", "parts": [c]})

            # Add language context to the user input
            language_names = {
                'en': 'English',
                'hi': 'Hindi', 
                'mr': 'Marathi',
                'gu': 'Gujarati',
                'ta': 'Tamil'
            }
            
            enhanced_input = user_input
            if language != 'en':
                enhanced_input = f"[User is speaking in {language_names.get(language, 'local language')}. Please respond in {language_names.get(language, 'the same language')} while helping them learn English.] {user_input}"

            # Start chat and send message
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(enhanced_input)

            ai_response = (response.text or "").strip()
            if not ai_response:
                fallback_responses = {
                    'en': "That's interesting! Tell me more about that! 🤔",
                    'hi': "यह दिलचस्प है! मुझे इसके बारे में और बताएं! 🤔",
                    'mr': "हे मनोरंजक आहे! मला याबद्दल अधिक सांगा! 🤔", 
                    'gu': "આ રસપ્રદ છે! મને આ વિશે વધુ કહો! 🤔",
                    'ta': "இது சுவாரஸ்யமானது! இதைப் பற்றி மேலும் சொல்லுங்கள்! 🤔"
                }
                ai_response = fallback_responses.get(language, fallback_responses['en'])

            # Add AI response to our history
            self.conversation_history[lang_key].append({
                "role": "assistant", 
                "content": ai_response
            })

            return ai_response

        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            fallback_responses = {
                'en': "That's interesting! Tell me more about that! 🤔",
                'hi': "यह दिलचस्प है! मुझे इसके बारे में और बताएं! 🤔",
                'mr': "हे मनोरंजक आहे! मला याबद्दल अधिक सांगा! 🤔",
                'gu': "આ રસપ્રદ છે! મને આ વિશે વધુ કહો! 🤔", 
                'ta': "இது சுவாரஸ்யமானது! இதைப் பற்றி மேலும் சொல்லுங்கள்! 🤔"
            }
            return fallback_responses.get(language, fallback_responses['en'])

    def handle_roleplay_response(self, user_input, session_id, roleplay_type, step, language='en'):
        """Handle roleplay scenario responses with multilingual support"""
        try:
            scenario = self.roleplay_scenarios.get(roleplay_type)
            if not scenario:
                responses = {
                    'en': "Let's try a different roleplay! 😊",
                    'hi': "आइए एक अलग रोलप्ले करते हैं! 😊", 
                    'mr': "चला वेगळा रोलप्ले करूया! 😊",
                    'gu': "ચાલો અલગ રોલપ્લે કરીએ! 😊",
                    'ta': "வேறு ரோல்ப்ளே செய்வோம்! 😊"
                }
                return responses.get(language, responses['en'])

            if step < len(scenario['prompts']):
                # For roleplay, we'll translate the English prompts
                response = scenario['prompts'][step]
                
                # Add acknowledgment if not first step
                if step > 0 and user_input:
                    acknowledgments = {
                        'en': ["Great answer! ", "Wonderful! ", "Perfect! ", "Excellent! ", "Nice! "],
                        'hi': ["बहुत बढ़िया जवाब! ", "शानदार! ", "परफेक्ट! ", "उत्कृष्ट! ", "अच्छा! "],
                        'mr': ["उत्तम उत्तर! ", "छान! ", "परफेक्ट! ", "उत्कृष्ट! ", "बरे! "],
                        'gu': ["ઉત્તમ જવાબ! ", "અદ્ભુત! ", "પરફેક્ટ! ", "ઉત્કૃષ્ટ! ", "સરસ! "],
                        'ta': ["அருமையான பதில்! ", "அற்புதம்! ", "சரியானது! ", "சிறந்தது! ", "நல்லது! "]
                    }
                    import random
                    ack_list = acknowledgments.get(language, acknowledgments['en'])
                    ack = random.choice(ack_list)
                    response = ack + response
                    
                # Translate response if needed (basic translation for demo)
                if language != 'en':
                    response = self.translate_roleplay_response(response, language)
                    
                return response
            else:
                completion_messages = {
                    'en': "🎉 Fantastic! You completed this roleplay perfectly! You're getting better at English every day! 🌟",
                    'hi': "🎉 शानदार! आपने यह रोलप्ले बिल्कुल सही तरीके से पूरा किया! आप हर दिन अंग्रेजी में बेहतर हो रहे हैं! 🌟",
                    'mr': "🎉 फँटास्टिक! तुम्ही हा रोलप्ले अगदी बरोबर पूर्ण केला! तुम्ही दररोज इंग्रजीत चांगले होत आहात! 🌟",
                    'gu': "🎉 ફેન્ટાસ્ટિક! તમે આ રોલપ્લે સંપૂર્ણ રીતે પૂરો કર્યો! તમે દરરોજ અંગ્રેજીમાં સારા થઈ રહ્યા છો! 🌟",
                    'ta': "🎉 அருமை! இந்த ரோல்ப்ளேவை நீங்கள் சரியாக முடித்தீர்கள்! நீங்கள் ஒவ்வொரு நாளும் ஆங்கிலத்தில் சிறப்பாகி வருகிறீர்கள்! 🌟"
                }
                return completion_messages.get(language, completion_messages['en'])

        except Exception as e:
            logger.error(f"Error in roleplay response: {e}")
            fallback_responses = {
                'en': "Great job! Let's continue our roleplay! 😊",
                'hi': "बहुत बढ़िया! आइए अपना रोलप्ले जारी रखते हैं! 😊",
                'mr': "छान काम! चला आपला रोलप्ले सुरू ठेवूया! 😊", 
                'gu': "સરસ કામ! આપણો રોલપ્લે ચાલુ રાખીએ! 😊",
                'ta': "நல்லது! நமது ரோல்ப்ளேவைத் தொடர்வோம்! 😊"
            }
            return fallback_responses.get(language, fallback_responses['en'])

    def translate_roleplay_response(self, text, language):
        """Basic translation for roleplay responses"""
        # This is a simplified translation - in production, use proper translation API
        translations = {
            'hi': {
                "Good morning! What's your name?": "सुप्रभात! आपका नाम क्या है?",
                "Do you like school?": "क्या आपको स्कूल पसंद है?",
                "What's your favorite subject?": "आपका पसंदीदा विषय क्या है?",
                "Welcome to our store! What would you like to buy today?": "हमारी दुकान में आपका स्वागत है! आज आप क्या खरीदना चाहेंगे?",
                "Great choice! How many would you like?": "बहुत अच्छा विकल्प! आप कितने चाहेंगे?",
                "Hi! Who do you live with at home?": "नमस्ते! आप घर पर किसके साथ रहते हैं?",
                "Hi there! Do you want to play with me?": "नमस्ते! क्या आप मेरे साथ खेलना चाहते हैं?"
            },
            'mr': {
                "Good morning! What's your name?": "सुप्रभात! तुमचे नाव काय?",
                "Do you like school?": "तुम्हाला शाळा आवडते का?",
                "What's your favorite subject?": "तुमचा आवडता विषय कोणता?",
                "Welcome to our store! What would you like to buy today?": "आमच्या दुकानात तुमचे स्वागत! आज तुम्हाला काय विकत घ्यायचे आहे?",
                "Great choice! How many would you like?": "चांगली निवड! तुम्हाला किती हवेत?",
                "Hi! Who do you live with at home?": "नमस्कार! तुम्ही घरी कोणासोबत राहता?",
                "Hi there! Do you want to play with me?": "नमस्कार! तुम्हाला माझ्यासोबत खेळायचे आहे का?"
            }
            # Add more translations as needed
        }
        
        lang_translations = translations.get(language, {})
        return lang_translations.get(text, text)

    def transcribe_audio(self, audio_data):
        """Transcribe audio to text using speech recognition"""
        try:
            r = sr.Recognizer()
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said. Please try speaking clearly! 😅"
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return "I'm having trouble hearing you. Please try again! 🎤"
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return "Sorry, there was an issue processing your voice. Please try again! 😅"

    def text_to_speech(self, text, language='en'):
        """Convert text to speech using gTTS"""
        try:
            import re
            clean_text = re.sub(r'[^\w\s\.,!?]', '', text)
            tts = gTTS(text=clean_text, lang=language, slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

# Initialize AI assistant
ai_assistant = SpeakGenieAI()

@app.route('/')
def index():
    """Serve the main application"""
    return render_template("index.html")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with language support"""
    try:
        data = request.json
        user_input = data.get('message', '')
        session_id = data.get('session_id', 'default')
        language = data.get('language', 'en')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get AI response with language support
        ai_response = ai_assistant.get_ai_response(user_input, session_id, language=language)
        
        # Generate TTS
        audio_file = ai_assistant.text_to_speech(ai_response, language)
        
        return jsonify({
            'response': ai_response,
            'audio_available': audio_file is not None,
            'audio_url': f'/api/audio/{os.path.basename(audio_file)}' if audio_file else None
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Failed to process chat message'}), 500



@app.route('/api/roleplay', methods=['POST'])
def roleplay():
    """Handle roleplay interactions"""
    try:
        data = request.json
        user_input = data.get('message', '')
        session_id = data.get('session_id', 'default')
        roleplay_type = data.get('roleplay_type', '')
        step = data.get('step', 0)
        language = data.get('language', 'en')
        
        # Get AI response for roleplay (local scripted)
        ai_response = ai_assistant.get_ai_response(
            user_input, session_id, mode='roleplay', 
            roleplay_type=roleplay_type, step=step
        )
        
        # Generate TTS
        audio_file = ai_assistant.text_to_speech(ai_response, language)
        
        return jsonify({
            'response': ai_response,
            'next_step': step + 1,
            'audio_available': audio_file is not None,
            'audio_url': f'/api/audio/{os.path.basename(audio_file)}' if audio_file else None
        })
        
    except Exception as e:
        logger.error(f"Roleplay error: {e}")
        return jsonify({'error': 'Failed to process roleplay'}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe uploaded audio to text"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Transcribe audio
        transcript = ai_assistant.transcribe_audio(audio_data)
        
        return jsonify({
            'transcript': transcript,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'error': 'Failed to transcribe audio'}), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    try:
        audio_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/mpeg')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        logger.error(f"Audio serving error: {e}")
        return jsonify({'error': 'Failed to serve audio'}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in ai_assistant.conversation_history:
            del ai_assistant.conversation_history[session_id]
        
        return jsonify({'success': True, 'message': 'History cleared'})
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({'error': 'Failed to clear history'}), 500
if __name__ == '__main__':
    # Make sure to install required dependencies first:
    # pip install flask flask-cors google-generativeai gtts SpeechRecognition pydub
    print("🧞‍♂️ Starting SpeakGenie AI Voice Tutor (Gemini)...")
    print("📝 Set your GEMINI_API_KEY environment variable.")
    print("🌐 Server will run at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=10000)
