import os
import re
import json
import pickle
from flask import Flask, request, jsonify
from datetime import datetime
from flask_cors import CORS
from chatbot_embeddings import (
    get_most_relevant_chunk,
    summarize_candidate_topic,
    detect_topic_from_query,
    summarize_topic_with_gpt,
    summarize_topic_by_candidate,
    classify_policy_stance,
    aliases,
    df
)

app = Flask(__name__)
CORS(app)

# Load topic chunks
with open("topic_chunks.json", "r") as f:
    topic_chunks = json.load(f)

# Load stance cache
with open("stance_cache_gst.json", "r") as f:
    gst_stance_cache = json.load(f)

# General topic cache
cache_file = "topic_response_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        topic_response_cache = json.load(f)
else:
    topic_response_cache = {}

# Log queries
def log_query_console(query, response, matched_topic=None, response_type="info"):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "response": str(response)[:300],
        "topic": matched_topic or "unknown",
        "type": response_type
    }

    # Always show in Render logs
    print(json.dumps(log_entry))

    # Attempt to store locally (may not persist on Render)
    try:
        logs = []
        if os.path.exists("query_log.json"):
            try:
                with open("query_log.json", "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ query_log.json is corrupt. Starting fresh.")
                logs = []

        logs.insert(0, log_entry)

        with open("query_log.json", "w") as f:
            json.dump(logs, f, indent=2)

    except Exception as e:
        print(f"⚠️ Logging to file failed: {e}")

# Save updated topic cache
def save_topic_cache():
    with open(cache_file, "w") as f:
        json.dump(topic_response_cache, f, indent=2)

# candidate URLS

def candidate_url(name):
    """Generate a candidate profile URL slug from name"""
    slug = name.strip().lower().replace(" ", "-")
    return f"https://election2025.gg/candidates/{slug}"

# Topic normalized

def normalize_topic(topic):
    topic = topic.lower().strip()
    return topic[4:] if topic.startswith("the ") else topic

# Regex for stance-type queries
stance_pattern = re.compile(
    r"\b(who|which candidates)\b\s+(support(?:s)?|oppose(?:s)?|want(?:s)?|favour(?:s)?|reject(?:s)?)\s+(.*)",
    re.IGNORECASE
)

print("🚀 Server is starting and logging works.")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        query = data.get("query", "")
        print(f"Received query: {query}")

        # Normalize special characters and formatting issues
        cleaned_query = query.lower()
        cleaned_query = cleaned_query.replace("’", "'")  # curly apostrophe
        cleaned_query = cleaned_query.replace("‘", "'")  # opening curly apostrophe
        cleaned_query = cleaned_query.replace("“", '"').replace("”", '"')  # curly quotes
        cleaned_query = cleaned_query.replace("–", "-").replace("—", "-")  # en and em dashes
        cleaned_query = re.sub(r"[^\w\s'\-]", "", cleaned_query)  # remove other non-word characters, keep hyphens and apostrophes
        cleaned_query = cleaned_query.replace(" the ", " ")  # normalize 'the'

        print(f"🔧 Cleaned query: {cleaned_query}")

        stance_match = stance_pattern.search(cleaned_query)

        if stance_match:
            topic = detect_topic_from_query(stance_match.group(3), aliases)
            if topic.startswith("the "):  # 🆕 strip leading 'the' from topic
                topic = topic[4:]
            print(f"📚 Detected general topic: {topic}")

            stance_keyword = stance_match.group(2).lower()

            if topic == "gst":
                primary_group = [c for c in gst_stance_cache if c["stance"] == "SUPPORT"] if "support" in stance_keyword else [c for c in gst_stance_cache if c["stance"] == "OPPOSE"]
                alternate_group = [c for c in gst_stance_cache if c["stance"] == "OPPOSE"] if "support" in stance_keyword else [c for c in gst_stance_cache if c["stance"] == "SUPPORT"]

                if not primary_group:
                    return jsonify({"response": "No clear stances found on GST."})

                def format_candidates(group):
                    return [{
                        "name": c["name"],
                        "summary": f"{c['stance']} - {c['reason']}",
                        "source_url": c.get("url", "")
                    } for c in group]

                response = {
                    "primary": format_candidates(primary_group),
                    "alternate": format_candidates(alternate_group)
                }

                log_query_console(query, response, matched_topic=topic, response_type="stance_gst")
                return jsonify({"response": response})

        # --- General topic summary ---
        summary_keywords = [
            "what do candidates say", "how do candidates view",
            "what are the candidates", "what is said about",
            "what are the views on", "tell me about", "views on", "tell me candidates' thoughts",
            "summary of", "what do they think", "what do they believe", "what are the candidates' plans",
            "what is their position", "what do they say", "how do they feel about", "what are candidates' ideas"
        ]

        if any(phrase in cleaned_query for phrase in summary_keywords):
            topic = detect_topic_from_query(cleaned_query, aliases)
            if topic:
                topic = normalize_topic(topic)
            print(f"📚 Detected general topic: {topic}")

            if topic in topic_response_cache:
                print("⚡ Using cached response")
                response_data = topic_response_cache[topic]

                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except json.JSONDecodeError:
                        print("⚠️ Could not parse cached response as JSON")
                        response_data = {"message": "⚠️ Corrupted cached response."}

                if isinstance(response_data, list):
                    response_data = {"candidates": response_data}                 

                log_query_console(query, response_data, matched_topic=topic, response_type="cached_topic_summary")
                return jsonify({"response": response_data})

            chunks = topic_chunks.get(topic, [])
            if not chunks:
                return jsonify({"response": f"No information found on {topic}."})

            response = {"candidates": summarize_topic_by_candidate(topic, chunks)}
            topic_response_cache[topic] = response
            save_topic_cache()
            log_query_console(query, response, matched_topic=topic, response_type="candidate_chunk_match")
            return jsonify({"response": response})

        # --- Fallback: single candidate on topic ---
        if "what does" in cleaned_query and "say about" in cleaned_query:
            parts = cleaned_query.split("say about")
            candidate_name = parts[0].replace("what does", "").strip()
            topic = parts[1].strip()
            if topic.startswith("the "):  # 🆕 normalize
                topic = topic[4:]

            print(f"🔁 Fallback to summarize_candidate_topic: '{candidate_name}' on '{topic}'")
            summary_text = summarize_candidate_topic(candidate_name, topic, df)
            if not isinstance(summary_text, str):
                summary_text = "No relevant content found."

            response_data = {
                "candidates": [{
                    "name": candidate_name,
                    "url": candidate_url(candidate_name),
                    "summary": summary_text
                }],
                "topic": topic
            }

            log_query_console(query, response_data, matched_topic=topic, response_type="generated_topic_summary")
            return jsonify({"response": response_data})

        # --- Fallback: more complex phrasing ---
        candidate_topic_match = re.search(
            r"(?:what does|where does|tell me what)\s+([\w\s\-']+?)\s+(?:say|think|stand).*?\b(on|about)?\b\s+([\w\s\-']+)",
            cleaned_query
        )

        if candidate_topic_match:
            candidate_name = candidate_topic_match.group(1).strip()
            topic = detect_topic_from_query(candidate_topic_match.group(3).strip(), aliases)
            if topic.startswith("the "):  # 🆕 normalize
                topic = topic[4:]

            print(f"🧑‍💼 Candidate detected: {candidate_name} | 🧠 Topic detected: {topic}")
            chunks = topic_chunks.get(topic, [])
            for chunk in chunks:
                if chunk["name"].lower() == candidate_name.lower():
                    response = {
                        "candidates": [{
                            "name": chunk['name'],
                            "summary": chunk['text'],
                            "url": candidate_url(chunk['name'])
                        }]
                    }
                    log_query_console(query, response, matched_topic=topic, response_type="fallback_direct_match")
                    return jsonify({"response": response})

            log_query_console(query, f"No specific statement found for {candidate_name} on {topic}.", matched_topic=topic, response_type="no_candidate_match")
            return jsonify({"response": f"No specific statement found for {candidate_name} on {topic}."})

        # --- Final fallback ---
        fallback_message = "I'm really sorry, I can't answer that question. Please try again by referring to a candidate and topic area."
        log_query_console(query, fallback_message, response_type="unrecognized_format")
        return jsonify({"response": fallback_message})

    except Exception as e:
        error_message = f"An error occurred: {e}"
        log_query_console(query, error_message, response_type="exception")
        return jsonify({"response": error_message}), 500
       

    

















