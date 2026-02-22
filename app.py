"""
NarrativeIQ - Complete Flask Backend (Single File)
No module import issues. Just run: python app.py
"""

import os
import sys
import json
import re
import difflib
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, Blueprint, send_file
import PyPDF2
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from io import BytesIO
from flask_cors import CORS
from dotenv import load_dotenv
import bcrypt
import jwt
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "narrativeiq-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file):
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    elif ext in ("txt", "md"):
        return file.read().decode("utf-8").strip()
    return ""

# ============================================================
# GROQ SETUP (free tier - https://console.groq.com)
# ============================================================
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
GROQ_MODEL = "llama-3.3-70b-versatile"  # free, fast, powerful

# ============================================================
# MONGODB SETUP
# ============================================================
_db = None

def get_db():
    global _db
    if _db is not None:
        return _db
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/narrativeiq")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _db = client["narrativeiq"]
    _db.users.create_index("email", unique=True)
    return _db

# ============================================================
# DB HELPERS
# ============================================================

def create_user(email, password, name):
    db = get_db()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = {
        "email": email,
        "password": hashed,
        "name": name,
        "credits": int(os.getenv("NEW_USER_CREDITS", 5)),
        "created_at": datetime.utcnow(),
    }
    result = db.users.insert_one(user)
    user["_id"] = str(result.inserted_id)
    return user

def get_user_by_email(email):
    return get_db().users.find_one({"email": email})

def get_user_by_id(user_id):
    return get_db().users.find_one({"_id": ObjectId(user_id)})

def verify_password(plain, hashed):
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def get_credits(user_id):
    user = get_user_by_id(user_id)
    return user.get("credits", 0) if user else 0

def deduct_credits(user_id, amount):
    user = get_user_by_id(user_id)
    if not user or user.get("credits", 0) < amount:
        return False
    get_db().users.update_one({"_id": ObjectId(user_id)}, {"$inc": {"credits": -amount}})
    return True

def add_credits(user_id, amount):
    get_db().users.update_one({"_id": ObjectId(user_id)}, {"$inc": {"credits": amount}})

def save_document(user_id, title, content):
    db = get_db()
    result = db.documents.insert_one({
        "user_id": user_id, "title": title, "content": content,
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    })
    return str(result.inserted_id)

def get_documents(user_id):
    docs = get_db().documents.find({"user_id": user_id}).sort("created_at", -1).limit(20)
    return [{**d, "_id": str(d["_id"])} for d in docs]

def get_document(doc_id):
    d = get_db().documents.find_one({"_id": ObjectId(doc_id)})
    if d:
        d["_id"] = str(d["_id"])
    return d

def save_enhancement(user_id, doc_id, operation, input_text, output_text, persona, changes, credits_used):
    get_db().enhancements.insert_one({
        "user_id": user_id, "doc_id": doc_id, "operation": operation,
        "persona": persona, "input_text": input_text, "output_text": output_text,
        "changes": changes, "credits_used": credits_used, "created_at": datetime.utcnow(),
    })

def get_enhancement_history(user_id):
    entries = get_db().enhancements.find({"user_id": user_id}).sort("created_at", -1).limit(50)
    result = []
    for e in entries:
        e["_id"] = str(e["_id"])
        if "created_at" in e:
            e["created_at"] = e["created_at"].isoformat()
        result.append(e)
    return result

# ============================================================
# JWT HELPERS
# ============================================================
SECRET_KEY = os.getenv("SECRET_KEY", "narrativeiq-secret-key")

def generate_token(user_id, email):
    payload = {
        "user_id": user_id, "email": email,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token):
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_token(token)
            kwargs["user_id"] = payload["user_id"]
            kwargs["user_email"] = payload["email"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ============================================================
# DIFF UTILS
# ============================================================

def compute_diff(original, enhanced):
    orig_words = original.split()
    enh_words = enhanced.split()
    matcher = difflib.SequenceMatcher(None, orig_words, enh_words)
    result = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            result.append({"type": "equal", "text": " ".join(orig_words[i1:i2])})
        elif op == "insert":
            result.append({"type": "insert", "text": " ".join(enh_words[j1:j2])})
        elif op == "delete":
            result.append({"type": "delete", "text": " ".join(orig_words[i1:i2])})
        elif op == "replace":
            result.append({"type": "delete", "text": " ".join(orig_words[i1:i2])})
            result.append({"type": "insert", "text": " ".join(enh_words[j1:j2])})
    return result

def similarity_score(original, enhanced):
    return round(difflib.SequenceMatcher(None, original, enhanced).ratio() * 100, 1)



PERSONAS = {
    "technical": {
        "label": "Technical",
        "description": "Precise, structured, jargon-aware writing",
        "system": "You are a technical writing expert. Rewrite the content to be precise, well-structured, and technically accurate. Use clear terminology, logical flow, and avoid ambiguity.",
    },
    "business": {
        "label": "Business",
        "description": "Professional & persuasive",
        "system": "You are a senior business communication expert. Rewrite the content to be professional, persuasive, and results-oriented. Use active voice and business-appropriate vocabulary.",
    },
    "finance": {
        "label": "Finance",
        "description": "Analytical & data-oriented tone",
        "system": "You are a financial analyst and writer. Rewrite the content with an analytical, data-driven tone. Use precise quantitative language and structured arguments.",
    },
    "simplified": {
        "label": "Simplified",
        "description": "Easy-to-read, beginner-friendly",
        "system": "You are a plain-language expert. Rewrite the content so it is crystal-clear for a beginner. Use simple words, short sentences, and avoid jargon.",
    },
    "comedian": {
        "label": "Comedian",
        "description": "Light, witty style",
        "system": "You are a witty comedy writer. Rewrite the content with a light, humorous tone. Add clever wordplay and amusing observations while keeping the core message intact.",
    },
    "poet": {
        "label": "Poet",
        "description": "Creative & expressive tone",
        "system": "You are a creative literary writer. Rewrite the content with vivid, expressive, and poetic language. Use metaphors and imagery while preserving the original meaning.",
    },
}

def _call_gemini(prompt):
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()

def _parse_json(text):
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(clean)

def enhance_with_persona(text, persona_key):
    persona = PERSONAS.get(persona_key, PERSONAS["simplified"])
    prompt = f"""{persona['system']}

TASK: Enhance the following text. Then provide a JSON explanation of key changes.

INPUT TEXT:
\"\"\"{text}\"\"\"

Respond ONLY with a valid JSON object (no markdown):
{{
  "enhanced_text": "<your enhanced version>",
  "changes": [
    {{"original": "<original phrase>", "enhanced": "<new version>", "reason": "<why this improves the text>"}}
  ]
}}
Include 3-8 of the most significant changes."""
    raw = _call_gemini(prompt)
    return _parse_json(raw)

def analyze_consistency(text):
    prompt = f"""You are a narrative consistency expert. Analyze the following text for inconsistencies.
Look for: character inconsistencies, timeline issues, factual contradictions, tone shifts, plot holes, setting inconsistencies.

TEXT:
\"\"\"{text}\"\"\"

Respond ONLY with valid JSON (no markdown):
{{
  "issues": [{{"type": "character|timeline|factual|tone|logic|setting", "description": "<issue>", "excerpt": "<relevant text>", "severity": "low|medium|high"}}],
  "overall_consistency_score": <0-100>,
  "summary": "<2-3 sentence assessment>"
}}"""
    return _parse_json(_call_gemini(prompt))

def suggest_structure(text):
    prompt = f"""You are an expert editor. Analyze the structure and clarity of this text.

TEXT:
\"\"\"{text}\"\"\"

Respond ONLY with valid JSON (no markdown):
{{
  "structure_score": <0-100>,
  "clarity_score": <0-100>,
  "flow_score": <0-100>,
  "suggestions": [{{"category": "structure|clarity|flow|redundancy|voice", "issue": "<problem>", "suggestion": "<fix>", "priority": "high|medium|low"}}],
  "strengths": ["<strength>"],
  "overall_feedback": "<2-3 sentence summary>"
}}"""
    return _parse_json(_call_gemini(prompt))

def track_character_evolution(text, character_name):
    prompt = f"""Track the emotional and behavioral evolution of "{character_name}" throughout this text.

TEXT:
\"\"\"{text}\"\"\"

Respond ONLY with valid JSON (no markdown):
{{
  "character": "{character_name}",
  "evolution_stages": [{{"stage": <int>, "label": "<short label>", "emotional_state": "<state>", "key_trait": "<trait>", "trigger": "<cause>", "excerpt": "<supporting text>"}}],
  "arc_type": "<e.g. Hero's Journey, Redemption Arc, Static>",
  "overall_development": "<2-3 sentence analysis>"
}}"""
    return _parse_json(_call_gemini(prompt))

def generate_mindmap(text):
    # Step 1: Extract entities
    entity_prompt = f"""Extract all key entities from this text.

TEXT:
\"\"\"{text}\"\"\"

Respond ONLY with valid JSON (no markdown):
{{
  "characters": [{{"name": "<name>", "role": "<role>", "traits": ["<trait>"], "mentions": <int>}}],
  "locations": [{{"name": "<name>", "description": "<desc>"}}],
  "organizations": [{{"name": "<name>", "description": "<desc>"}}],
  "themes": ["<theme>"],
  "time_periods": ["<period>"]
}}"""
    entities = _parse_json(_call_gemini(entity_prompt))

    # Step 2: Relationship mapping
    char_names = [c["name"] for c in entities.get("characters", [])]
    rel_prompt = f"""Given characters: {char_names}

TEXT:
\"\"\"{text[:3000]}\"\"\"

Identify relationships between entities. Respond ONLY with valid JSON (no markdown):
{{
  "relationships": [{{"from": "<name>", "to": "<name>", "type": "<Friend|Enemy|Mentor|etc>", "description": "<brief desc>"}}]
}}"""
    relationships = _parse_json(_call_gemini(rel_prompt))

    # Step 3: Build React Flow graph
    nodes, edges, node_id_map = [], [], {}

    def add_node(name, node_type, extra=None):
        if name in node_id_map:
            return node_id_map[name]
        nid = f"node_{len(nodes)}"
        node_id_map[name] = nid
        nodes.append({"id": nid, "data": {"label": name, "type": node_type, **(extra or {})},
                      "type": node_type, "position": {"x": 0, "y": 0}})
        return nid

    for c in entities.get("characters", []):
        add_node(c["name"], "character", {"role": c.get("role",""), "traits": c.get("traits",[]), "mentions": c.get("mentions",1)})
    for l in entities.get("locations", []):
        add_node(l["name"], "location", {"description": l.get("description","")})
    for o in entities.get("organizations", []):
        add_node(o["name"], "organization", {"description": o.get("description","")})
    for t in entities.get("themes", []):
        add_node(t, "theme", {})

    for i, rel in enumerate(relationships.get("relationships", [])):
        src = node_id_map.get(rel["from"])
        tgt = node_id_map.get(rel["to"])
        if src and tgt:
            edges.append({"id": f"edge_{i}", "source": src, "target": tgt,
                          "label": rel["type"], "data": {"description": rel.get("description","")}})

    return {
        "nodes": nodes, "edges": edges, "entities": entities,
        "relationships": relationships.get("relationships", []),
        "summary": {
            "character_count": len(entities.get("characters", [])),
            "location_count": len(entities.get("locations", [])),
            "theme_count": len(entities.get("themes", [])),
            "relationship_count": len(relationships.get("relationships", [])),
        },
    }

def deep_consistency_scan(text):
    consistency = analyze_consistency(text)
    structure = suggest_structure(text)
    return {
        "consistency": consistency,
        "structure": structure,
        "combined_score": (
            consistency.get("overall_consistency_score", 50) +
            structure.get("structure_score", 50) +
            structure.get("clarity_score", 50)
        ) // 3,
    }

# ============================================================
# FEATURE PRICING
# ============================================================
FEATURE_PRICING = {
    "script_enhancement":     {"cost": 1, "label": "Script / Text Enhancement"},
    "style_transformation":   {"cost": 1, "label": "Persona Style Transformation"},
    "consistency_check":      {"cost": 1, "label": "Narrative Consistency Check"},
    "structure_analysis":     {"cost": 1, "label": "Structure & Clarity Analysis"},
    "character_evolution":    {"cost": 1, "label": "Character Evolution Tracking"},
    "mindmap_generation":     {"cost": 2, "label": "Narrative Memory Graph (Mindmap)"},
    "deep_consistency_scan":  {"cost": 2, "label": "Deep Consistency Scan"},
}


# ============================================================
# STORY COMPLETION AI FUNCTION
# ============================================================

def complete_story(partial_text, genre="general", style="narrative", length="medium"):
    length_map = {"short": "500-800", "medium": "1000-1500", "long": "2000-3000"}
    word_count = length_map.get(length, "1000-1500")

    prompt = (
        "You are a master storyteller and screenwriter. "
        "A user has given you a partial script or story idea. "
        "Complete it into a full, compelling, well-structured story.\n\n"
        f"Genre: {genre}\n"
        f"Style: {style}\n"
        f"Target length: {word_count} words\n\n"
        "PARTIAL SCRIPT / IDEA:\n"
        f"{partial_text}\n\n"
        "Instructions:\n"
        "- Continue naturally from where the user left off (don't restart)\n"
        "- Maintain the same characters, tone, and setting\n"
        "- Add a proper story arc: rising action, climax, resolution\n"
        "- Make dialogue feel natural and authentic\n"
        "- End with a satisfying conclusion\n\n"
        "Respond ONLY with valid JSON (no markdown):\n"
        "{\n"
        '  \"completed_story\": \"<the full completed story text>\",\n'
        '  \"title\": \"<a compelling title>\",\n'
        '  \"summary\": \"<2-3 sentence summary>\",\n'
        '  \"characters\": [\"<main characters>\"],\n'
        '  \"genre_detected\": \"<genre>\",\n'
        '  \"word_count\": 1200,\n'
        '  \"story_structure\": {\n'
        '    \"setup\": \"<setup>\",\n'
        '    \"conflict\": \"<conflict>\",\n'
        '    \"climax\": \"<climax>\",\n'
        '    \"resolution\": \"<resolution>\"\n'
        "  }\n"
        "}"
    )
    return _parse_json(_call_gemini(prompt))

# ============================================================
# ROUTES - AUTH
# ============================================================

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"error": "name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    try:
        user = create_user(email, password, name)
    except DuplicateKeyError:
        return jsonify({"error": "Email already registered"}), 409
    token = generate_token(str(user["_id"]), email)
    return jsonify({"token": token, "user": {"id": str(user["_id"]), "name": user["name"], "email": user["email"], "credits": user["credits"]}}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    token = generate_token(str(user["_id"]), email)
    return jsonify({"token": token, "user": {"id": str(user["_id"]), "name": user["name"], "email": user["email"], "credits": user["credits"]}})

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me(user_id, user_email):
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"id": str(user["_id"]), "name": user["name"], "email": user["email"], "credits": user["credits"]})


@app.route("/api/auth/update-profile", methods=["PUT"])
@require_auth
def update_profile(user_id, user_email):
    data = request.get_json()
    name  = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()

    if not name and not email:
        return jsonify({"error": "Provide at least name or email to update"}), 400

    update_fields = {}
    if name:
        update_fields["name"] = name
    if email:
        # Check if email already taken by another user
        existing = get_user_by_email(email)
        if existing and str(existing["_id"]) != user_id:
            return jsonify({"error": "Email already in use"}), 409
        update_fields["email"] = email

    get_db().users.update_one({"_id": ObjectId(user_id)}, {"$set": update_fields})
    user = get_user_by_id(user_id)

    # If email changed, issue a new token
    new_email = user["email"]
    token = generate_token(user_id, new_email)

    return jsonify({
        "message": "Profile updated successfully",
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "credits": user["credits"],
        }
    })


@app.route("/api/auth/change-password", methods=["PUT"])
@require_auth
def change_password(user_id, user_email):
    data         = request.get_json()
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not old_password or not new_password:
        return jsonify({"error": "old_password and new_password are required"}), 400
    if len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400

    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    if not verify_password(old_password, user["password"]):
        return jsonify({"error": "Current password is incorrect"}), 401

    new_hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    get_db().users.update_one({"_id": ObjectId(user_id)}, {"$set": {"password": new_hashed}})

    return jsonify({"message": "Password changed successfully"})

# ============================================================
# ROUTES - ENHANCE
# ============================================================

@app.route("/api/enhance/personas", methods=["GET"])
def list_personas():
    return jsonify({k: {"label": v["label"], "description": v["description"]} for k, v in PERSONAS.items()})

@app.route("/api/enhance/persona", methods=["POST"])
@require_auth
def persona_enhance(user_id, user_email):
    data       = request.get_json()
    text       = data.get("text", "").strip()
    persona_key= data.get("persona", "simplified")
    doc_id     = data.get("doc_id", "")
    title      = data.get("title", "Untitled")
    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > 50000:
        return jsonify({"error": "Text too long (max 50,000 characters)"}), 400
    if persona_key not in PERSONAS:
        return jsonify({"error": f"Unknown persona. Valid: {list(PERSONAS.keys())}"}), 400
    if not deduct_credits(user_id, 1):
        return jsonify({"error": "Insufficient credits", "credits_required": 1}), 402
    try:
        result = enhance_with_persona(text, persona_key)
    except Exception as e:
        add_credits(user_id, 1)
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    enhanced_text = result.get("enhanced_text", "")
    changes       = result.get("changes", [])
    if not doc_id:
        doc_id = save_document(user_id, title, text)
    save_enhancement(user_id, doc_id, "persona_enhance", text, enhanced_text, persona_key, changes, 1)
    return jsonify({
        "enhanced_text": enhanced_text,
        "changes": changes,
        "diff": compute_diff(text, enhanced_text),
        "similarity_score": similarity_score(text, enhanced_text),
        "persona": PERSONAS[persona_key]["label"],
        "doc_id": doc_id,
        "credits_used": 1,
    })

@app.route("/api/enhance/history", methods=["GET"])
@require_auth
def history(user_id, user_email):
    return jsonify({"history": get_enhancement_history(user_id)})

@app.route("/api/enhance/save", methods=["POST"])
@require_auth
def save_doc(user_id, user_email):
    data    = request.get_json()
    title   = data.get("title", "Untitled").strip()
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400
    doc_id = save_document(user_id, title, content)
    return jsonify({"doc_id": doc_id, "message": "Document saved"}), 201

@app.route("/api/enhance/documents", methods=["GET"])
@require_auth
def list_docs(user_id, user_email):
    docs = get_documents(user_id)
    for d in docs:
        if "created_at" in d: d["created_at"] = d["created_at"].isoformat()
        if "updated_at" in d: d["updated_at"] = d["updated_at"].isoformat()
        d["preview"] = d.get("content", "")[:200]
        d.pop("content", None)
    return jsonify({"documents": docs})

@app.route("/api/enhance/documents/<doc_id>", methods=["GET"])
@require_auth
def get_doc(user_id, user_email, doc_id):
    doc = get_document(doc_id)
    if not doc or doc.get("user_id") != user_id:
        return jsonify({"error": "Document not found"}), 404
    if "created_at" in doc: doc["created_at"] = doc["created_at"].isoformat()
    return jsonify(doc)

# ============================================================
# ROUTES - ANALYZE
# ============================================================

@app.route("/api/analyze/consistency", methods=["POST"])
@require_auth
def consistency_check(user_id, user_email):
    data   = request.get_json()
    text   = data.get("text", "").strip()
    doc_id = data.get("doc_id", "")
    if not text:
        return jsonify({"error": "text is required"}), 400
    if not deduct_credits(user_id, 1):
        return jsonify({"error": "Insufficient credits", "credits_required": 1}), 402
    try:
        result = analyze_consistency(text)
    except Exception as e:
        add_credits(user_id, 1)
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    save_enhancement(user_id, doc_id, "consistency_check", text, "", None, [], 1)
    return jsonify({"consistency_analysis": result, "credits_used": 1})

@app.route("/api/analyze/structure", methods=["POST"])
@require_auth
def structure_check(user_id, user_email):
    data   = request.get_json()
    text   = data.get("text", "").strip()
    doc_id = data.get("doc_id", "")
    if not text:
        return jsonify({"error": "text is required"}), 400
    if not deduct_credits(user_id, 1):
        return jsonify({"error": "Insufficient credits", "credits_required": 1}), 402
    try:
        result = suggest_structure(text)
    except Exception as e:
        add_credits(user_id, 1)
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    save_enhancement(user_id, doc_id, "structure_check", text, "", None, [], 1)
    return jsonify({"structure_analysis": result, "credits_used": 1})

@app.route("/api/analyze/character", methods=["POST"])
@require_auth
def character_evolution(user_id, user_email):
    data           = request.get_json()
    text           = data.get("text", "").strip()
    character_name = data.get("character_name", "").strip()
    doc_id         = data.get("doc_id", "")
    if not text or not character_name:
        return jsonify({"error": "text and character_name are required"}), 400
    if not deduct_credits(user_id, 1):
        return jsonify({"error": "Insufficient credits", "credits_required": 1}), 402
    try:
        result = track_character_evolution(text, character_name)
    except Exception as e:
        add_credits(user_id, 1)
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    save_enhancement(user_id, doc_id, "character_evolution", text, "", None, [], 1)
    return jsonify({"character_evolution": result, "credits_used": 1})

@app.route("/api/analyze/deep-scan", methods=["POST"])
@require_auth
def deep_scan(user_id, user_email):
    data   = request.get_json()
    text   = data.get("text", "").strip()
    doc_id = data.get("doc_id", "")
    if not text:
        return jsonify({"error": "text is required"}), 400
    if not deduct_credits(user_id, 2):
        return jsonify({"error": "Insufficient credits", "credits_required": 2}), 402
    try:
        result = deep_consistency_scan(text)
    except Exception as e:
        add_credits(user_id, 2)
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    save_enhancement(user_id, doc_id, "deep_scan", text, "", None, [], 2)
    return jsonify({"deep_scan": result, "credits_used": 2})


# ============================================================
# MINDMAP IMAGE GENERATOR
# ============================================================

def generate_mindmap_image(nodes, edges, title="Narrative Mindmap"):
    COLOR_MAP = {
        "character":    "#4F46E5",
        "location":     "#059669",
        "organization": "#D97706",
        "theme":        "#DC2626",
        "default":      "#6B7280",
    }

    G = nx.DiGraph()
    node_labels, node_colors, node_sizes = {}, [], []

    for node in nodes:
        nid   = node["id"]
        label = node["data"]["label"]
        ntype = node["data"].get("type", "default")
        G.add_node(nid)
        node_labels[nid] = label
        node_colors.append(COLOR_MAP.get(ntype, COLOR_MAP["default"]))
        mentions = node["data"].get("mentions", 1)
        node_sizes.append(2000 + mentions * 300)

    edge_labels = {}
    for edge in edges:
        G.add_edge(edge["source"], edge["target"])
        edge_labels[(edge["source"], edge["target"])] = edge.get("label", "")

    try:
        pos = nx.spring_layout(G, k=3, seed=42, iterations=100)
    except Exception:
        pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    if len(G.nodes) > 0:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                                font_color="white", font_size=9, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#94A3B8",
                               arrows=True, arrowsize=20, width=1.5,
                               alpha=0.7, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                     font_color="#CBD5E1", font_size=7, alpha=0.9)

    legend_items = [
        mpatches.Patch(color="#4F46E5", label="Character"),
        mpatches.Patch(color="#059669", label="Location"),
        mpatches.Patch(color="#D97706", label="Organization"),
        mpatches.Patch(color="#DC2626", label="Theme"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              facecolor="#1E293B", labelcolor="white", fontsize=9, framealpha=0.8)
    ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf

# ============================================================
# ROUTES - MINDMAP
# ============================================================

@app.route("/api/mindmap/generate", methods=["POST"])
@require_auth
def generate_mindmap_route(user_id, user_email):
    data   = request.get_json()
    text   = data.get("text", "").strip()
    doc_id = data.get("doc_id", "")
    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) < 100:
        return jsonify({"error": "Text too short (min 100 characters)"}), 400
    if len(text) > 30000:
        return jsonify({"error": "Text too long (max 30,000 characters)"}), 400
    if not deduct_credits(user_id, 2):
        return jsonify({"error": "Insufficient credits", "credits_required": 2}), 402
    try:
        result = generate_mindmap(text)
    except Exception as e:
        add_credits(user_id, 2)
        return jsonify({"error": f"AI error: {str(e)}"}), 500
    save_enhancement(user_id, doc_id, "mindmap_generate", text, "", None, [], 2)
    return jsonify({"mindmap": result, "credits_used": 2})

# ============================================================
# ROUTES - CREDITS
# ============================================================

@app.route("/api/credits/balance", methods=["GET"])
@require_auth
def balance(user_id, user_email):
    return jsonify({"credits": get_credits(user_id), "user_id": user_id})

@app.route("/api/credits/pricing", methods=["GET"])
def pricing():
    return jsonify({"features": FEATURE_PRICING})

@app.route("/api/credits/add", methods=["POST"])
@require_auth
def add_credits_route(user_id, user_email):
    data   = request.get_json()
    amount = int(data.get("amount", 5))
    if amount < 1 or amount > 100:
        return jsonify({"error": "amount must be between 1 and 100"}), 400
    add_credits(user_id, amount)
    return jsonify({"message": f"Added {amount} credits", "new_balance": get_credits(user_id)})


@app.route("/api/mindmap/image", methods=["POST"])
@require_auth
def mindmap_image_route(user_id, user_email):
    data   = request.get_json()
    text   = data.get("text", "").strip()
    doc_id = data.get("doc_id", "")
    title  = data.get("title", "Narrative Mindmap")

    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) < 100:
        return jsonify({"error": "Text too short (min 100 characters)"}), 400
    if len(text) > 30000:
        return jsonify({"error": "Text too long (max 30,000 characters)"}), 400

    if not deduct_credits(user_id, 2):
        return jsonify({"error": "Insufficient credits", "credits_required": 2}), 402

    try:
        # Generate graph data via AI
        graph_data = generate_mindmap(text)
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        # Generate image
        img_buf = generate_mindmap_image(nodes, edges, title)
        save_enhancement(user_id, doc_id, "mindmap_image", text, "", None, [], 2)

        return send_file(img_buf, mimetype="image/png",
                         as_attachment=False,
                         download_name="mindmap.png")
    except Exception as e:
        add_credits(user_id, 2)
        return jsonify({"error": f"Error: {str(e)}"}), 500



# ============================================================
# ROUTES - FILE UPLOAD (PDF / TXT / MD)
# ============================================================

@app.route("/api/upload/extract", methods=["POST"])
@require_auth
def extract_text(user_id, user_email):
    """Upload a PDF/TXT/MD file and extract its text content."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key: file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: pdf, txt, md"}), 400
    try:
        text = extract_text_from_file(file)
        if not text:
            return jsonify({"error": "Could not extract text from file"}), 400
        word_count = len(text.split())
        char_count = len(text)
        return jsonify({
            "text": text,
            "word_count": word_count,
            "char_count": char_count,
            "filename": secure_filename(file.filename),
            "message": f"Successfully extracted {word_count} words"
        })
    except Exception as e:
        return jsonify({"error": f"File processing error: {str(e)}"}), 500


# ============================================================
# ROUTES - STORY COMPLETION
# ============================================================

@app.route("/api/story/complete", methods=["POST"])
@require_auth
def story_complete(user_id, user_email):
    """Complete a partial script or story idea into a full story. (2 credits)"""
    COST = 2
    data         = request.get_json()
    partial_text = data.get("text", "").strip()
    genre        = data.get("genre", "general")
    style        = data.get("style", "narrative")
    length       = data.get("length", "medium")
    doc_id       = data.get("doc_id", "")
    title        = data.get("title", "My Story")

    if not partial_text:
        return jsonify({"error": "text is required"}), 400
    if len(partial_text) < 20:
        return jsonify({"error": "Please provide at least a sentence or idea to complete"}), 400

    if not deduct_credits(user_id, COST):
        return jsonify({"error": "Insufficient credits", "credits_required": COST}), 402

    try:
        result = complete_story(partial_text, genre, style, length)
    except Exception as e:
        add_credits(user_id, COST)
        return jsonify({"error": f"AI error: {str(e)}"}), 500

    # Auto save the completed story as a document
    completed_text = result.get("completed_story", "")
    saved_title    = result.get("title", title)
    if not doc_id:
        doc_id = save_document(user_id, saved_title, completed_text)

    save_enhancement(user_id, doc_id, "story_complete", partial_text, completed_text, None, [], COST)

    return jsonify({
        "completed_story": completed_text,
        "title": saved_title,
        "summary": result.get("summary", ""),
        "characters": result.get("characters", []),
        "genre_detected": result.get("genre_detected", genre),
        "word_count": result.get("word_count", 0),
        "story_structure": result.get("story_structure", {}),
        "doc_id": doc_id,
        "credits_used": COST,
    })


@app.route("/api/story/complete-from-file", methods=["POST"])
@require_auth
def story_complete_from_file(user_id, user_email):
    """Upload a PDF/TXT file with partial story, then complete it. (2 credits)"""
    COST = 2

    # Get file
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key: file"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: pdf, txt, md"}), 400

    genre  = request.form.get("genre", "general")
    style  = request.form.get("style", "narrative")
    length = request.form.get("length", "medium")
    title  = request.form.get("title", "My Story")

    try:
        partial_text = extract_text_from_file(file)
        if not partial_text:
            return jsonify({"error": "Could not extract text from file"}), 400
    except Exception as e:
        return jsonify({"error": f"File error: {str(e)}"}), 500

    if not deduct_credits(user_id, COST):
        return jsonify({"error": "Insufficient credits", "credits_required": COST}), 402

    try:
        result = complete_story(partial_text, genre, style, length)
    except Exception as e:
        add_credits(user_id, COST)
        return jsonify({"error": f"AI error: {str(e)}"}), 500

    completed_text = result.get("completed_story", "")
    saved_title    = result.get("title", title)
    doc_id         = save_document(user_id, saved_title, completed_text)
    save_enhancement(user_id, doc_id, "story_complete_file", partial_text, completed_text, None, [], COST)

    return jsonify({
        "completed_story": completed_text,
        "title": saved_title,
        "summary": result.get("summary", ""),
        "characters": result.get("characters", []),
        "genre_detected": result.get("genre_detected", genre),
        "word_count": result.get("word_count", 0),
        "story_structure": result.get("story_structure", {}),
        "original_text": partial_text,
        "doc_id": doc_id,
        "credits_used": COST,
    })

# ============================================================
# DEBUG - list available models for your API key
# ============================================================
@app.route("/api/debug/models")
def list_models():
    return jsonify({"model": GROQ_MODEL, "provider": "Groq", "status": "active"})

# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "NarrativeIQ"})

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)