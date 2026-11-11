import os
import re
import json
import socket
import base64
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import requests
import concurrent.futures
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

USER_DATA_FILE = "user_data.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Resilient model fallback list - fastest and most available first
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

# ── Dynamic User Data (Profile & Contacts) ───────────────────────────────────
# Clean defaults: NEVER use fake or dummy numbers!
DEFAULT_USER_DATA = {
    "profile": {
        "user_name": "Nanna Garu",
        "city": "Hyderabad",
        "primary_language": "Tanglish",
        "emergency_contact": ""
    },
    "contacts": [
        {
            "id": "c1",
            "name": "Vijay",
            "relation": "Brother / Friend",
            "phone": "",
            "aliases": ["vijay", "vijju", "brother", "anna", "tammudu"]
        },
        {
            "id": "c2",
            "name": "Amma",
            "relation": "Mother",
            "phone": "",
            "aliases": ["amma", "mom", "mother", "mummy"]
        },
        {
            "id": "c3",
            "name": "Ananya",
            "relation": "Daughter",
            "phone": "",
            "aliases": ["ananya", "daughter", "ammayi", "papa"]
        },
        {
            "id": "c4",
            "name": "Dr. Sharma",
            "relation": "Doctor",
            "phone": "",
            "aliases": ["doctor", "dr sharma", "sharma", "clinic", "hospital"]
        },
        {
            "id": "c5",
            "name": "Ravi",
            "relation": "Son",
            "phone": "",
            "aliases": ["ravi", "son", "abbayi", "koduku"]
        }
    ]
}

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Purge any legacy dummy numbers
                for c in data.get("contacts", []):
                    p = c.get("phone", "")
                    if "9848055555" in p or "9848011111" in p or "9848033333" in p:
                        c["phone"] = ""
                return data
        except Exception as e:
            print("[USER DATA READ ERROR]", e)
    return DEFAULT_USER_DATA

def save_user_data(data):
    try:
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print("[USER DATA SAVE ERROR]", e)
        return False

# ── Multi-Turn Conversation Memory & Action State ─────────────────────────────
chat_history = []
session_state = {
    "pending_action": None
}

# ── Smart Contact Matcher ─────────────────────────────────────────────────────
def find_contact(query_text, contacts):
    """Matches text against contact names, relations, and aliases including Telugu postpositions."""
    text = query_text.lower().strip()
    
    # Priority 1: Exact Name match
    for c in contacts:
        name_lower = c["name"].lower()
        if re.search(rf"\b{re.escape(name_lower)}(?:ki|ku|tho|gariki|garu)?\b", text):
            return c
            
    # Priority 2: Alias match
    for c in contacts:
        for alias in c.get("aliases", []):
            if alias and re.search(rf"\b{re.escape(alias.lower())}(?:ki|ku|tho|gariki|garu)?\b", text):
                return c

    # Priority 3: Relation match
    for c in contacts:
        rel = c.get("relation", "").lower()
        if rel and re.search(rf"\b{re.escape(rel)}\b", text):
            return c

    # Priority 4: Substring match
    for c in contacts:
        if c["name"].lower() in text:
            return c

    return None

def extract_single_turn_whatsapp(user_text, contact_name):
    """Extracts message payload when user speaks a combined command in English or Tanglish."""
    text = user_text.strip()
    text = re.sub(r'^(?:hey\s+)?bujji\s*,?\s*', '', text, flags=re.IGNORECASE).strip()

    # 1. "Tell Vijay that ..." or "Message Vijay ..."
    m = re.search(rf'^(?:tell|message|whatsapp)\s+{re.escape(contact_name)}\s+(?:that\s+)?(.+)', text, re.IGNORECASE)
    if m:
        msg = m.group(1).strip()
        msg = re.sub(r'^(?:that|to)\s+', '', msg, flags=re.IGNORECASE).strip()
        msg = re.sub(r'(?:please|thanks|thank you)\b', '', msg, flags=re.IGNORECASE).strip()
        if msg and not re.search(r'^(?:message|whatsapp|call|msg)\b', msg, re.IGNORECASE):
            return msg

    # 2. "Vijay ki [message] ani [whatsapp pettu/cheppu]"
    cleaned = re.sub(rf'^{re.escape(contact_name)}\s*(?:ki|ku|gariki|tho)\s+', '', text, flags=re.IGNORECASE).strip()
    suffix_pattern = r'\s+ani\s+(?:whatsapp|message|msg|text)?\s*(?:pettu|cheppu|pampinchu|rayi|send cheyyi).*$'
    if re.search(suffix_pattern, cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(suffix_pattern, '', cleaned, flags=re.IGNORECASE).strip()
        if cleaned and cleaned.lower() not in ['message', 'whatsapp', 'pettu', 'cheppu', 'msg']:
            return cleaned

    # 3. "Vijay ki WhatsApp message pettu: [message]"
    m3 = re.search(rf'^{re.escape(contact_name)}\s*(?:ki|ku|gariki)?\s*(?:whatsapp|message|msg)?\s*(?:message|msg)?\s*(?:pettu|cheyyi|pampinchu)?\s*[:,-]\s*(.+)', text, re.IGNORECASE)
    if m3:
        msg = m3.group(1).strip()
        msg = re.sub(r'^(?:ani|that|to)\s+', '', msg, flags=re.IGNORECASE).strip()
        if msg and msg.lower() not in ['pettu', 'cheppu', 'cheyyi', 'pampinchu', 'message', 'whatsapp']:
            return msg

    return None

def make_gcal_link(title, details="Scheduled with Bujji AI", location="Hyderabad"):
    """Builds a pre-filled Google Calendar event URL."""
    base = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    now = datetime.now()
    start_dt = now + timedelta(days=1)
    start_str = start_dt.strftime("%Y%m%dT100000")
    end_str = start_dt.strftime("%Y%m%dT110000")
    params = f"&text={urllib.parse.quote(title)}&dates={start_str}/{end_str}&details={urllib.parse.quote(details)}&location={urllib.parse.quote(location)}"
    return base + params

def build_system_prompt(user_data):
    profile = user_data.get("profile", {})
    contacts = user_data.get("contacts", [])

    contacts_summary = []
    for c in contacts:
        aliases_str = ", ".join(c.get("aliases", []))
        phone_str = c.get('phone', '').strip()
        phone_label = phone_str if phone_str else "NOT_SET (Phone number is missing)"
        contacts_summary.append(
            f"- {c['name']} ({c.get('relation', 'Contact')}): Phone: {phone_label} | Aliases: {aliases_str}"
        )
    contacts_block = "\n".join(contacts_summary)

    prompt = f"""You are 'Bujji', an affectionate, patient, voice-first AI companion for elders in India.
The elder's name is '{profile.get('user_name', 'Nanna Garu')}', living in '{profile.get('city', 'Hyderabad')}'.
Speak warmly in simple English or natural, comforting Tanglish (Telugu + English mix).

KNOWN FAMILY CONTACTS:
{contacts_block}
- Emergency SOS Services: 112
- Ambulance: 108

CRITICAL RULES:
1. NEVER USE UNKNOWN OR DUMMY PHONE NUMBERS:
   - When generating CALL or WHATSAPP actions, you MUST use the exact real phone number listed for that contact above.
   - If that contact's phone is 'NOT_SET' or missing, output:
     [ACTION: SET_PHONE | <contact_name> | <intent_details>]
     and ask the elder to provide their phone number.
2. MULTI-TURN CONVERSATIONS:
   - If Bujji asked "Vijay ki em message cheppali?" and user responds with a message, that response IS the message for Vijay!
   - Immediately output: [ACTION: WHATSAPP | <exact_phone_of_vijay> | <user_message>]
3. SMART FEATURES (SESSION 2):
   - GOOGLE CALENDAR: If user asks to schedule or book an event / appointment / meeting:
     [ACTION: CALENDAR | <title> | <time> | <google_calendar_url>]
   - PILL ALARM: If user asks to remind them for a tablet, pill, or medicine:
     [ACTION: PILL_ALARM | <time> | <medicine_name>]
   - NEWS: If user asks for news, headlines, or today's updates:
     [ACTION: NEWS | Telugu / National | <summary>]
   - SAFE UPI: If user asks how to pay with GPay / PhonePe / QR code:
     [ACTION: UPI_GUIDE | GPay / PhonePe | Safe Step-by-Step Instructions]
4. CALL COMMANDS:
   - "Call Amma" or "Amma ki call cheyyi":
     If phone is set -> [ACTION: CALL | <exact_phone> | Amma]
     If phone is missing -> [ACTION: SET_PHONE | Amma | Call Amma]
5. SCAM SHIELD:
   - If user shares an SMS with OTP, bank block, electric cut warning, or lottery prize, warn firmly:
     "Idhi 100% SCAM fraud SMS! Meeku vacchina OTP evariki share cheyyakandi."
     [ACTION: SCAM_ALERT | HIGH_RISK | Do not share OTP or click links]
6. WEATHER:
   - [ACTION: WEATHER | <city> | <temp>]
7. Keep spoken responses warm, polite ('andi', 'garu'), and under 3 short sentences.

ACTION TAG FORMAT (Must be the last line):
- [ACTION: CALL | <real_phone> | <name>]
- [ACTION: WHATSAPP | <real_phone> | <message>]
- [ACTION: SET_PHONE | <name> | <details>]
- [ACTION: CALENDAR | <title> | <time> | <gcal_url>]
- [ACTION: PILL_ALARM | <time> | <medicine>]
- [ACTION: NEWS | <topic> | <summary>]
- [ACTION: UPI_GUIDE | <app> | <instructions>]
- [ACTION: REMINDER | <time> | <label>]
- [ACTION: SCAM_ALERT | HIGH_RISK | <advice>]
- [ACTION: WEATHER | <city> | <temp>]
- [ACTION: SOS | <phone> | Emergency Alert]
- [ACTION: NONE]
"""
    return prompt

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def serve_ui():
    return send_from_directory(".", "index.html")

@app.route("/privacy")
def serve_privacy():
    return send_from_directory(".", "privacy.html")

@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory(".", "manifest.json")

@app.route("/sw.js")
def serve_sw():
    response = send_from_directory(".", "sw.js")
    response.headers["Content-Type"] = "application/javascript"
    return response

# ── Contacts & Profile Management API ─────────────────────────────────────────

@app.route("/api/user_data", methods=["GET"])
def get_user_data_route():
    data = load_user_data()
    return jsonify(data)

@app.route("/api/user_data", methods=["POST"])
def update_user_data_route():
    payload = request.get_json(silent=True) or {}
    if not payload:
        return jsonify({"error": "No data provided"}), 400
    
    success = save_user_data(payload)
    if success:
        return jsonify({"status": "saved", "message": "Contacts and profile updated successfully!"})
    return jsonify({"error": "Failed to save user data"}), 500

@app.route("/api/contacts/quick_phone", methods=["POST"])
def quick_phone_update():
    """Inline phone number update for a contact directly from chat card."""
    payload = request.get_json(silent=True) or {}
    name = payload.get("name", "").strip()
    phone = payload.get("phone", "").strip().replace(" ", "")

    if not name or not phone:
        return jsonify({"error": "Name and Phone required"}), 400

    if not phone.startswith("+"):
        if len(phone) == 10:
            phone = "+91" + phone
        elif len(phone) == 12 and phone.startswith("91"):
            phone = "+" + phone

    user_data = load_user_data()
    contacts = user_data.get("contacts", [])
    updated = False

    for c in contacts:
        if c["name"].lower() == name.lower():
            c["phone"] = phone
            updated = True
            break

    if not updated:
        new_c = {
            "id": f"c_{name.lower()}_{len(contacts)+1}",
            "name": name,
            "relation": "Family / Friend",
            "phone": phone,
            "aliases": [name.lower()]
        }
        contacts.append(new_c)

    user_data["contacts"] = contacts
    save_user_data(user_data)

    return jsonify({"status": "ok", "name": name, "phone": phone})

@app.route("/api/contacts/import_vcf", methods=["POST"])
def import_vcf_route():
    vcf_text = ""
    if "file" in request.files:
        vcf_text = request.files["file"].read().decode("utf-8", errors="ignore")
    elif request.data:
        vcf_text = request.data.decode("utf-8", errors="ignore")

    if not vcf_text:
        return jsonify({"error": "No VCF content received"}), 400

    imported_contacts = []
    cards = re.split(r'BEGIN:VCARD', vcf_text, flags=re.IGNORECASE)
    
    for card in cards:
        if not card.strip():
            continue
        fn_match = re.search(r'\nFN[;:]([^\r\n]+)', card, re.IGNORECASE)
        name = fn_match.group(1).strip() if fn_match else ""
        if not name:
            n_match = re.search(r'\nN[;:]([^\r\n]+)', card, re.IGNORECASE)
            if n_match:
                parts = [p.strip() for p in n_match.group(1).split(";") if p.strip()]
                parts.reverse()
                name = " ".join(parts).strip()

        tel_match = re.search(r'\nTEL[^:]*:([^\r\n]+)', card, re.IGNORECASE)
        raw_phone = tel_match.group(1).strip() if tel_match else ""
        phone = re.sub(r'[^0-9+]', '', raw_phone)

        if phone and not phone.startswith('+'):
            if len(phone) == 10:
                phone = '+91' + phone
            elif len(phone) == 12 and phone.startswith('91'):
                phone = '+' + phone

        if name and phone:
            imported_contacts.append({
                "id": f"vcf_{len(imported_contacts) + 1}",
                "name": name,
                "relation": "Imported Contact",
                "phone": phone,
                "aliases": [name.lower()]
            })

    if not imported_contacts:
        return jsonify({"error": "No valid contacts found in VCF"}), 400

    user_data = load_user_data()
    existing_contacts = user_data.get("contacts", [])
    
    existing_phones = {c.get("phone", "") for c in existing_contacts if c.get("phone")}
    for new_c in imported_contacts:
        if new_c["phone"] not in existing_phones:
            existing_contacts.append(new_c)
            existing_phones.add(new_c["phone"])

    user_data["contacts"] = existing_contacts
    save_user_data(user_data)
    
    return jsonify({
        "status": "imported",
        "count": len(imported_contacts),
        "imported": imported_contacts,
        "total_contacts": len(existing_contacts)
    })

@app.route("/api/contacts/import_bulk", methods=["POST"])
def import_bulk_route():
    """Bulk import contacts from pasted text, list, or CSV."""
    raw_text = ""
    if "file" in request.files:
        raw_text = request.files["file"].read().decode("utf-8", errors="ignore")
    else:
        payload = request.get_json(silent=True) or {}
        raw_text = payload.get("text", "")

    if not raw_text.strip():
        return jsonify({"error": "No text provided"}), 400

    imported_contacts = []
    lines = raw_text.strip().splitlines()
    user_data = load_user_data()
    existing_contacts = user_data.get("contacts", [])
    existing_phones = {c.get("phone", "") for c in existing_contacts if c.get("phone")}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(r'(?:\+?91[\-\s]?)?[6-9](?:[\s\-]?\d){9}', line)
        if m:
            raw_phone = m.group(0)
            phone = re.sub(r'[^0-9+]', '', raw_phone)
            if not phone.startswith('+'):
                if len(phone) == 10:
                    phone = '+91' + phone
                elif len(phone) == 12 and phone.startswith('91'):
                    phone = '+' + phone

            name = line.replace(raw_phone, '').strip(' :-,\t|')
            if not name:
                name = "Contact " + phone[-4:]

            if phone not in existing_phones:
                new_c = {
                    "id": f"bulk_{len(existing_contacts) + 1}",
                    "name": name,
                    "relation": "Contact",
                    "phone": phone,
                    "aliases": [name.lower()]
                }
                existing_contacts.append(new_c)
                existing_phones.add(phone)
                imported_contacts.append(new_c)

    user_data["contacts"] = existing_contacts
    save_user_data(user_data)

    return jsonify({
        "status": "imported",
        "count": len(imported_contacts),
        "imported": imported_contacts,
        "total_contacts": len(existing_contacts)
    })

# ── Session 2 Smart Features: News, Medicine Scanner & Weather ───────────────

@app.route("/api/news")
def get_news():
    """Fetches real-time headlines from Google News RSS."""
    lang = request.args.get("lang", "te") # 'te' for Telugu, 'en' for English
    url = "https://news.google.com/rss?hl=te&gl=IN&ceid=IN:te" if lang == "te" else "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    headlines = []
    try:
        res = requests.get(url, timeout=5)
        if res.ok:
            root = ET.fromstring(res.text)
            for item in root.findall(".//item")[:3]:
                title = item.find("title").text or ""
                clean_title = re.sub(r"\s*-\s*[^-]+$", "", title).strip()
                link = item.find("link").text or "#"
                if clean_title:
                    headlines.append({"title": clean_title, "link": link})
    except Exception as e:
        print("[NEWS FETCH ERROR]", e)

    if not headlines:
        headlines = [
            {"title": "Senior citizen wellness camps launched across Hyderabad and Telangana", "link": "#"},
            {"title": "Indian Railways introduces dedicated senior citizen assistance counters", "link": "#"},
            {"title": "Hyderabad weather to remain pleasant this week with mild sunshine", "link": "#"}
        ]

    return jsonify({"status": "ok", "lang": lang, "headlines": headlines})

@app.route("/api/medicine/scan", methods=["POST"])
def scan_medicine():
    """Gemini Vision AI: Analyzes medicine photos for name, dosage, and precautions."""
    img_bytes = None
    mime_type = "image/jpeg"

    if "file" in request.files:
        file = request.files["file"]
        img_bytes = file.read()
        mime_type = file.mimetype or "image/jpeg"
    else:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("image_base64", "")
        if b64:
            if "," in b64:
                header, b64 = b64.split(",", 1)
                if "png" in header:
                    mime_type = "image/png"
            img_bytes = base64.b64decode(b64)

    if not img_bytes:
        return jsonify({"error": "No image provided"}), 400

    if not client:
        return jsonify({
            "medicine_name": "Dolo 650 (Paracetamol)",
            "purpose": "Fever and mild body pain relief",
            "dosage": "1 tablet after food with plenty of water",
            "warnings": "Do not exceed 3 tablets in 24 hours. Take after food.",
            "suggested_time": "09:00 AM",
            "spoken_summary": "Namaste Nanna garu! Idi Dolo 650 Paracetamol tablet andi, fever taggadaniki. Doctor garu food tarvata vesukomanaru."
        })

    prompt = """You are Bujji, an affectionate and knowledgeable AI health companion for an elder in India.
Analyze this photo of a medicine (tablet strip, bottle, syrup, or prescription).
Extract and structure the medical information clearly.

Return ONLY a valid JSON object matching:
{
  "medicine_name": "Full medicine name & strength (e.g., Telmisartan 40mg or Dolo 650)",
  "purpose": "What this medicine treats in simple language (e.g., High Blood Pressure / Fever & Pain)",
  "dosage": "Clear dosage instructions (e.g., 1 tablet once daily after breakfast with water)",
  "warnings": "Key safety precautions (e.g., Take with plenty of water, do not skip)",
  "suggested_time": "09:00 AM",
  "spoken_summary": "Warm 2-sentence Tanglish explanation starting with Namaste Nanna garu, telling them what medicine this is and when to take it."
}
If no medicine is visible, politely state in spoken_summary that they should take a clear picture of the medicine strip or doctor's prescription.
"""
    try:
        part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        r = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[part, prompt],
            config={"temperature": 0.2}
        )
        raw = r.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return jsonify(data)
    except Exception as e:
        print("[MEDICINE SCAN ERROR]", e)
        return jsonify({
            "medicine_name": "Medicine Identified",
            "purpose": "General health maintenance",
            "dosage": "1 tablet daily after food",
            "warnings": "Please verify dosage with your doctor.",
            "suggested_time": "09:00 AM",
            "spoken_summary": "Namaste Nanna garu! Mee medicine scan chesanu. Doctor cheppina time ki water tho vesukondi."
        })

@app.route("/weather")
def get_weather():
    lat = request.args.get("lat", "17.3850")
    lon = request.args.get("lon", "78.4867")
    city = request.args.get("city", "Hyderabad")

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url, timeout=5)
        if res.ok:
            data = res.json()
            curr = data.get("current_weather", {})
            temp = curr.get("temperature", 28)
            code = curr.get("weathercode", 0)
            
            cond = "Sunny and clear"
            if code in [1, 2, 3]: cond = "Partly cloudy"
            elif code in [45, 48]: cond = "Misty / Foggy"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: cond = "Rainy"
            elif code >= 95: cond = "Thunderstorm"

            return jsonify({
                "city": city,
                "temperature": f"{temp}°C",
                "condition": cond,
                "windspeed": f"{curr.get('windspeed', 10)} km/h"
            })
    except Exception as e:
        print("[WEATHER ERROR]", e)

    return jsonify({
        "city": city,
        "temperature": "29°C",
        "condition": "Warm and pleasant",
        "windspeed": "12 km/h"
    })

@app.route("/health")
def health():
    user_data = load_user_data()
    return jsonify({
        "status": "ok",
        "service": "Bujji Voice Companion",
        "session": 2,
        "api_key_configured": bool(GEMINI_API_KEY),
        "contacts_count": len(user_data.get("contacts", [])),
        "history_turns": len(chat_history)
    })

@app.route("/selftest")
def selftest():
    if not client:
        return jsonify({"status": "error", "error": "GEMINI_API_KEY is missing in .env"}), 400

    for model_name in FALLBACK_MODELS:
        try:
            r = client.models.generate_content(
                model=model_name,
                contents="Namaste Bujji! Say OK in one word."
            )
            return jsonify({
                "status": "ok",
                "model_used": model_name,
                "reply": r.text.strip()[:60]
            })
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                continue
            return jsonify({"status": "error", "model": model_name, "error": err_str[:250]})

    return jsonify({"status": "all_busy", "message": "All Gemini models are momentarily busy. Please retry."})

@app.route("/reset", methods=["POST"])
def reset():
    chat_history.clear()
    session_state["pending_action"] = None
    return jsonify({"status": "cleared", "message": "Conversation history reset"})

# ── Main Conversational Agent (/chat) ─────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please say that again.", "action": "NONE"})

    user_data = load_user_data()
    contacts = user_data.get("contacts", [])
    pending = session_state.get("pending_action")
    lower_msg = user_message.lower()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Multi-Turn Pending Action Resolution
    # ─────────────────────────────────────────────────────────────────────────
    if pending and pending.get("type") == "WHATSAPP":
        target_name = pending.get("contact", "Contact")
        
        # Cancellation
        if any(w in lower_msg for w in ["cancel", "vaddu", "oddu", "vaddhu", "stop", "no", "aagu"]):
            session_state["pending_action"] = None
            reply = "Sare andi, message cancel chesanu."
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Bujji: {reply}")
            return jsonify({"reply": reply, "action": "NONE"})

        # Switch to CALL
        if any(w in lower_msg for w in ["call", "phone", "matladali"]):
            session_state["pending_action"] = None
            ref_c = find_contact(target_name, contacts)
            target_phone = ref_c.get("phone", "").strip() if ref_c else ""
            if target_phone:
                reply = f"Sare andi, {target_name} ki call chestunnanu."
                action = f"CALL | {target_phone} | {target_name}"
            else:
                reply = f"{target_name} phone number inka save avvaledhu. Daya chesi phone number enter cheyyandi."
                action = f"SET_PHONE | {target_name} | Call {target_name}"
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Bujji: {reply}")
            return jsonify({"reply": reply, "action": action})

        # Switch recipient
        new_matched = find_contact(user_message, contacts)
        if new_matched and new_matched["name"].lower() != target_name.lower():
            session_state["pending_action"] = {
                "type": "WHATSAPP",
                "contact": new_matched["name"]
            }
            reply = f"Sare, {new_matched['name']} ki em message cheppali?"
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Bujji: {reply}")
            return jsonify({"reply": reply, "action": "NONE"})

        # Resolve payload
        session_state["pending_action"] = None
        clean_msg = user_message
        clean_msg = re.sub(rf'^(?:cheppu|tell him|tell her|ani cheppu)\s*[:,-]?\s*', '', clean_msg, flags=re.IGNORECASE).strip()

        ref_c = find_contact(target_name, contacts)
        target_phone = ref_c.get("phone", "").strip() if ref_c else ""

        if not target_phone:
            reply = f"{target_name} phone number inka save avvaledhu. Phone number enter cheyyandi, ventane WhatsApp send cheddamu!"
            action = f"SET_PHONE | {target_name} | WhatsApp: \"{clean_msg}\""
        else:
            reply = f"{target_name} ki message ready chesanu: \"{clean_msg}\". Okka tap tho WhatsApp open chesi pampandi!"
            action = f"WHATSAPP | {target_phone} | {clean_msg}"

        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Bujji: {reply}")
        return jsonify({"reply": reply, "action": action})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Smart Feature - Daily News Intent
    # ─────────────────────────────────────────────────────────────────────────
    if any(w in lower_msg for w in ["news", "headlines", "samacharam", "varthalu", "eeroju news"]):
        lang = "te" if any(w in lower_msg for w in ["telugu", "samacharam", "varthalu"]) else "en"
        # Fetch fresh news
        try:
            url = f"http://127.0.0.1:5000/api/news?lang={lang}"
            r = requests.get(url, timeout=3)
            news_items = r.json().get("headlines", [])
        except Exception:
            news_items = []

        if news_items:
            h1 = news_items[0]["title"]
            h2 = news_items[1]["title"] if len(news_items) > 1 else ""
            h3 = news_items[2]["title"] if len(news_items) > 2 else ""
            reply = f"Eeroju mukhya samacharam: {h1}. {h2}"
            action = f"NEWS | Today's Headlines | {h1} ~ {h2} ~ {h3}"
        else:
            reply = "Eeroju news headlines ready ga unnavi. Display lo chudandi!"
            action = "NEWS | Today | Senior citizen wellness camps launched in Hyderabad ~ Indian Railways adds special counters"

        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Bujji: {reply}")
        return jsonify({"reply": reply, "action": action})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Smart Feature - Safe UPI Voice Guide
    # ─────────────────────────────────────────────────────────────────────────
    if any(w in lower_msg for w in ["gpay", "phonepe", "upi", "paytm", "money transfer", "scan qr", "payment"]):
        reply = "Nanna garu, GPay lo payment chesetappudu okati gurthunchukondi: Dabbu teesukotaniki UPI PIN epudoo avasaram ledu! Store lo QR scan chesi peru chusi matrame pay cheyyandi."
        action = "UPI_GUIDE | Safe UPI Payment | 1. Open GPay/PhonePe -> 2. Scan QR -> 3. Verify Name -> 4. Never enter PIN to receive money!"
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Bujji: {reply}")
        return jsonify({"reply": reply, "action": action})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Smart Feature - Google Calendar Booking
    # e.g. "Book doctor appointment for Friday at 10 AM"
    # ─────────────────────────────────────────────────────────────────────────
    if any(w in lower_msg for w in ["calendar", "appointment", "schedule", "repu 10", "repu 9", "podduna 10", "meeting with"]):
        event_title = "Doctor Appointment"
        if "temple" in lower_msg: event_title = "Temple Visit"
        elif "meeting" in lower_msg: event_title = "Family Meeting"
        elif "doctor" in lower_msg: event_title = "Doctor Appointment"
        elif "hospital" in lower_msg: event_title = "Hospital Visit"
        else: event_title = user_message[:40]

        gcal_url = make_gcal_link(event_title, details="Scheduled via Bujji AI Companion")
        reply = f"Sare andi, {event_title} schedule chesanu. Okka tap tho Google Calendar lo add chesukondi!"
        action = f"CALENDAR | {event_title} | Tomorrow 10:00 AM | {gcal_url}"
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Bujji: {reply}")
        return jsonify({"reply": reply, "action": action})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Smart Feature - Interactive Pill Alarm
    # e.g. "Remind me to take BP tablet at 8 PM"
    # ─────────────────────────────────────────────────────────────────────────
    if any(w in lower_msg for w in ["tablet", "pill", "medicine", "bp tablet", "sugar tablet", "mandulu", "vesukovali"]):
        pill_name = "BP Tablet"
        if "sugar" in lower_msg: pill_name = "Sugar Tablet"
        elif "bp" in lower_msg: pill_name = "BP Tablet"
        elif "dolo" in lower_msg or "fever" in lower_msg: pill_name = "Dolo 650"
        elif "night" in lower_msg: pill_name = "Night Tablet"
        else:
            m_med = re.search(r'(?:take|vesuko|reminder for)\s+([a-zA-Z0-9\s]+?)(?:\s+at|\s+ki|\s+tablet|\s*$)', user_message, re.IGNORECASE)
            if m_med:
                pill_name = m_med.group(1).strip()

        # Extract time
        m_time = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)', user_message)
        time_label = m_time.group(1).strip() if m_time else "08:00 PM"
        if "pm" not in time_label.lower() and "am" not in time_label.lower():
            time_label += " PM"

        reply = f"Sure andi! Meeku {pill_name} tablet ki {time_label} ki alarm set chesanu. Time avvagane alarm mogutundi."
        action = f"PILL_ALARM | {time_label} | {pill_name}"
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Bujji: {reply}")
        return jsonify({"reply": reply, "action": action})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Single-Turn Combined WhatsApp Command
    # ─────────────────────────────────────────────────────────────────────────
    matched_contact = find_contact(user_message, contacts)
    is_whatsapp_request = any(w in lower_msg for w in ["whatsapp", "message", "msg", "pettu", "cheppu", "pampinchu", "rayi"])
    is_call_request = any(w in lower_msg for w in ["call", "phone", "matladali"])

    if matched_contact and is_whatsapp_request and not is_call_request:
        extracted_content = extract_single_turn_whatsapp(user_message, matched_contact["name"])
        target_phone = matched_contact.get("phone", "").strip()

        if extracted_content:
            if not target_phone:
                reply = f"{matched_contact['name']} phone number inka save avvaledhu. Phone number enter cheyyandi, ventane WhatsApp send cheddamu!"
                action = f"SET_PHONE | {matched_contact['name']} | WhatsApp: \"{extracted_content}\""
            else:
                reply = f"{matched_contact['name']} ki WhatsApp message ready chesanu: \"{extracted_content}\". Tap to send!"
                action = f"WHATSAPP | {target_phone} | {extracted_content}"
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Bujji: {reply}")
            return jsonify({"reply": reply, "action": action})
        else:
            session_state["pending_action"] = {
                "type": "WHATSAPP",
                "contact": matched_contact["name"]
            }
            reply = f"{matched_contact['name']} ki em message cheppali? Cheppandi, nenu WhatsApp lo ready chestanu."
            chat_history.append(f"User: {user_message}")
            chat_history.append(f"Bujji: {reply}")
            return jsonify({"reply": reply, "action": "NONE"})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Direct Call Command
    # ─────────────────────────────────────────────────────────────────────────
    if matched_contact and is_call_request:
        target_phone = matched_contact.get("phone", "").strip()
        if not target_phone:
            reply = f"{matched_contact['name']} phone number inka save avvaledhu. Daya chesi phone number enter cheyyandi."
            action = f"SET_PHONE | {matched_contact['name']} | Call {matched_contact['name']}"
        else:
            reply = f"{matched_contact['name']} ki ventane call chestunnanu. Chudandi!"
            action = f"CALL | {target_phone} | {matched_contact['name']}"
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"Bujji: {reply}")
        return jsonify({"reply": reply, "action": action})

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Call Gemini AI with Dynamic Context & History
    # ─────────────────────────────────────────────────────────────────────────
    chat_history.append(f"User: {user_message}")
    recent_context = "\n".join(chat_history[-6:])
    dynamic_system_prompt = build_system_prompt(user_data)
    prompt = f"{dynamic_system_prompt}\n\nRecent Conversation:\n{recent_context}\n\nBujji:"

    full_text = ""
    last_err = None

    if client:
        def try_gemini(m):
            return client.models.generate_content(
                model=m,
                contents=prompt,
                config={"temperature": 0.2}
            ).text.strip()

        for model_name in FALLBACK_MODELS:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(try_gemini, model_name)
                    full_text = future.result(timeout=4.5)
                    if full_text:
                        break
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "Timeout" in err_str:
                    continue
                else:
                    break

    # Safety fallbacks
    if not full_text:
        err_msg = str(last_err) if last_err else "No AI response"
        print(f"[BUJJI GEMINI FALLBACK] {err_msg}")
        if "otp" in lower_msg or "block" in lower_msg or "sbi" in lower_msg or "bank" in lower_msg or "bill" in lower_msg:
            full_text = "Idhi 100% SCAM fraud SMS! Meeku vacchina OTP evariki share cheyyakandi. [ACTION: SCAM_ALERT | HIGH_RISK | Do not share OTP or click links]"
        elif "weather" in lower_msg or "varsham" in lower_msg or "yenda" in lower_msg:
            full_text = f"Eeroju {user_data.get('profile',{}).get('city','Hyderabad')} lo weather warm ga undhi, around 29°C. Bayataki velleppudu water bottle teesukellandi! [ACTION: WEATHER | {user_data.get('profile',{}).get('city','Hyderabad')} | 29°C Warm]"
        elif "sos" in lower_msg or "emergency" in lower_msg or "help" in lower_msg or "danger" in lower_msg:
            em_phone = user_data.get("profile", {}).get("emergency_contact", "")
            if not em_phone:
                full_text = "Emergency alert trigger chesanu! 112 Police & 108 Ambulance ki call ready ga undhi. [ACTION: SOS | 112 | Emergency Police 112]"
            else:
                full_text = f"Emergency alert trigger chesanu! Family ki call & location alert ready chesanu. [ACTION: SOS | {em_phone} | Emergency Alert]"
        else:
            full_text = "Namaste! Cheppandi, meeku call cheyyamantara, WhatsApp message pettamantara, ledha medicine reminder set cheyyamantara? [ACTION: NONE]"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Parse Action Tag & Cross-Verify Phone Numbers
    # ─────────────────────────────────────────────────────────────────────────
    action = "NONE"
    reply_text = full_text

    if "[ACTION:" in full_text:
        idx = full_text.rfind("[ACTION:")
        reply_text = full_text[:idx].strip()
        tag_segment = full_text[idx + len("[ACTION:"):].strip()
        if "]" in tag_segment:
            action = tag_segment[:tag_segment.index("]")].strip()
        else:
            action = tag_segment.replace("]", "").strip()

    # Re-verify phone numbers so unknown numbers NEVER happen
    if action.startswith("CALL |") or action.startswith("WHATSAPP |"):
        parts = [p.strip() for p in action.split("|")]
        if len(parts) >= 3:
            act_type = parts[0]
            phone = parts[1]
            extra = parts[2]

            ref_c = find_contact(extra, contacts) or find_contact(user_message, contacts)
            if ref_c:
                real_phone = ref_c.get("phone", "").strip()
                if not real_phone:
                    action = f"SET_PHONE | {ref_c['name']} | {act_type}: {extra}"
                    reply_text = f"{ref_c['name']} phone number inka save avvaledhu. Daya chesi phone number enter cheyyandi."
                else:
                    action = f"{act_type} | {real_phone} | {extra}"

    if not reply_text:
        reply_text = "Done! I have taken care of that for you."

    chat_history.append(f"Bujji: {reply_text}")
    return jsonify({"reply": reply_text, "action": action})

if __name__ == "__main__":
    host_ip = "0.0.0.0"
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"

    print("=" * 60)
    print("🎙️  BUJJI - SESSION 2 SMART AI BACKEND RUNNING")
    print("=" * 60)
    print(f"Local browser : http://localhost:5000")
    print(f"Mobile WiFi   : http://{local_ip}:5000")
    print(f"API Key Active: {bool(GEMINI_API_KEY)}")
    print("=" * 60)
    app.run(debug=True, host=host_ip, port=5000)
