import os
import re
import json
import pickle
from collections import defaultdict
import openai
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

# Keyword extractor
def extract_keywords(query):
    stopwords = set([
        "what", "does", "do", "say", "think", "about", "on", "the", "is",
        "candidates", "candidate", "view", "views", "opinions", "are", "their", "position",
        "they"  # ✅ Now included
    ])
    tokens = re.findall(r"\b\w+\b", query.lower())
    return [word for word in tokens if word not in stopwords]

# GPT-powered summarizer
def gpt_summarize_candidate(candidate_name, text, query):
    prompt = f"""The following is campaign content from a candidate in an election. Based on the content and the question, summarize the candidate's position in a clear and concise way suitable for a general audience.

Candidate: {candidate_name}
User question: {query}
Content:
\"\"\"
{text.strip()}
\"\"\"

Summary:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ GPT fallback error: {e}")
        return text.strip().split("\n")[0][:400] + "..."

# candidate URLS

def candidate_url(name):
    """Generate a candidate profile URL slug from name"""
    slug = name.strip().lower().replace(" ", "-")
    return f"https://election2025.gg/candidates/{slug}"

# Last-resort keyword matcher using embeddings.csv
def last_resort_keyword_summary(query, df, fallback_topic=None, top_n=3):
    if fallback_topic is None:
        fallback_topic = detect_topic_from_query(query, aliases)
    if fallback_topic and fallback_topic in aliases:
        query_keywords = [kw.lower() for kw in aliases[fallback_topic]]
    else:
        query_keywords = extract_keywords(query)
    print(f"🔍 Fallback keywords: {query_keywords}")

    matches = defaultdict(list)

    for _, row in df.iterrows():
        candidate = row.get("name", "").strip()
        text = str(row.get("Text", "")).lower()

        if any(keyword in text for keyword in query_keywords):
            matches[candidate].append(row)

    if not matches:
        return {
            "candidates": [{
                "name": "Info",
                "summary": "Sorry, I couldn't find relevant information on that topic.",
                "source_url": ""
            }]
        }

    results = []
    for candidate, entries in matches.items():
        combined_text = " ".join(str(e["Text"]) for e in entries[:top_n])
        summary = gpt_summarize_candidate(candidate, combined_text, query)
        results.append({
            "name": candidate,
            "summary": summary,
            "source_url": candidate_url(candidate)
        })

    return {"candidates": results}

def candidates_with_little_on_topic(df, topic, aliases, min_mentions=1):
    topic_keywords = aliases.get(topic, [topic])
    topic_keywords = [kw.lower() for kw in topic_keywords]

    counts = defaultdict(int)
    for _, row in df.iterrows():
        name = row.get("name", "").strip()
        text = str(row.get("Text", "")).lower()
        if any(keyword in text for keyword in topic_keywords):
            counts[name] += 1

    all_candidates = set(df["name"].dropna().unique())
    low_mention_candidates = [
        {
            "name": name,
            "summary": f"No substantial mention of {topic}.",
            "source_url": candidate_url(name)
        }
        for name in all_candidates if counts[name] < min_mentions
    ]

    return {"candidates": low_mention_candidates}

# Save updated topic cache
def save_topic_cache():
    with open(cache_file, "w") as f:
        json.dump(topic_response_cache, f, indent=2)



# Topic normalized

def normalize_topic(topic):
    topic = topic.lower().strip()
    return topic[4:] if topic.startswith("the ") else topic

# Regex for stance-type queries
stance_pattern = re.compile(
    r"\b(who|which candidates)\b\s+("
    r"supports?|opposes?|opposed\s+to|in\s+favour\s+of|backs?|rejects?|wants?|favours?|"
    r"are\s+against|are\s+for|is\s+against|is\s+for|stands\s+(?:against|for)|"
    r"don't\s+support|do\s+not\s+support|disagree\s+with|doesn't\s+agree\s+with"
    r")\s+(.*)",
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
                return jsonify({
                    "response": response,
                    "type": "stance_gst"
                })

        # --- Detect "who talks little about..." questions ---
        low_mention_match = re.search(
            r"(which|who)\s+(candidates\s+)?(don'?t|do not|rarely|barely|seldom).*?(talk|mention|say).*?\b(about|on)?\b\s+(.+)",
            cleaned_query
        )

        if low_mention_match:
            raw_topic = low_mention_match.group(6).strip()
            raw_topic = raw_topic.strip().lower()
            topic = detect_topic_from_query(raw_topic, aliases)

            if topic:
                topic = normalize_topic(topic)
            else:
                topic = raw_topic  # fallback to raw topic if detection fails
            print(f"🔎 Detected low-mention topic: {topic}")

            response = candidates_with_little_on_topic(df, topic, aliases)
            log_query_console(query, response, matched_topic=topic, response_type="low_mention_query")
            return jsonify({
                "response": response,
                "type": "low_mention_query"
            })

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
                return jsonify({
                    "response": response_data,
                    "type": "cached_topic_summary"
                })

            chunks = topic_chunks.get(topic, [])
            if len(chunks) > 40:
                # Attempt sub-filtering based on the user's original query
                keywords = extract_keywords(cleaned_query)
                filtered_chunks = [chunk for chunk in chunks if any(k in chunk["text"].lower() for k in keywords)]

                if filtered_chunks:
                    try:
                        gpt_summary = summarize_topic_with_gpt(topic, filtered_chunks)
                        response_data = {
                            "candidates": gpt_summary,
                            "topic": topic
                        }
                        log_query_console(query, response_data, matched_topic=topic, response_type="gpt_filtered_summary")
                        return jsonify({
                            "response": response_data,
                            "type": "gpt_filtered_summary"
                        })
                    except Exception as e:
                        log_query_console(query, f"⚠️ GPT filtered summary failed: {e}", matched_topic=fallback_topic, response_type="gpt_error")

                # Fallback if no relevant chunks
                warning = {
                    "candidates": [{
                        "name": "Note",
                        "summary": f"The topic '{fallback_topic}' includes too many sources to summarize. Try a more specific question (e.g., 'active travel in transport').",
                        "source_url": ""
                    }]
                }
                log_query_console(query, warning, matched_topic=fallback_topic, response_type="fallback_topic_too_large")
                return jsonify({"response": warning})


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
                    "source_url": candidate_url(candidate_name),
                    "summary": summary_text
                }],
                "topic": topic
            }

            log_query_console(query, response_data, matched_topic=topic, response_type="generated_topic_summary")
            return jsonify({
                "response": response_data,
                "type": "generated_topic_summary"
            })

        # --- Fallback: "[Candidate] on [Topic]" style query ---
        short_form_match = re.match(r"^([\w\s\-']+?)\s+on\s+([\w\s\-']+)$", cleaned_query)
        if short_form_match:
            candidate_name = short_form_match.group(1).strip()
            topic = detect_topic_from_query(short_form_match.group(2).strip(), aliases)
            if topic.startswith("the "):  # normalize
                topic = topic[4:]

            print(f"📌 Short form detected: {candidate_name} on {topic}")
            chunks = topic_chunks.get(topic, [])
            for chunk in chunks:
                if chunk["name"].lower() == candidate_name.lower():
                    response = {
                        "candidates": [{
                            "name": chunk['name'],
                            "summary": chunk.get('summary') or chunk.get('text', ''),
                            "source_url": candidate_url(chunk['name'])
                        }]
                    }
                    log_query_console(query, response, matched_topic=topic, response_type="fallback_short_form_match")
                    return jsonify({
                        "response": response,
                        "type": "fallback_short_form_match"
                    })

            log_query_console(query, f"No specific statement found for {candidate_name} on {topic}.", matched_topic=topic, response_type="no_short_match")
            return jsonify({
                "response": {
                    "candidates": [{
                        "name": candidate_name,
                        "summary": f"No specific statement found on {topic}.",
                        "source_url": candidate_url(candidate_name)
                    }]
                },
                "type": "no_short_match"
            })

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
                            "summary": chunk.get('summary') or chunk.get('text', ''),
                            "source_url": candidate_url(chunk['name'])
                        }]
                    }
                    log_query_console(query, response, matched_topic=topic, response_type="fallback_direct_match")
                    return jsonify({
                        "response": response,
                        "type": "fallback_direct_match"
                    })

            log_query_console(query, f"No specific statement found for {candidate_name} on {topic}.", matched_topic=topic, response_type="no_candidate_match")
            return jsonify({
                "response": {
                    "candidates": [{
                        "name": candidate_name,
                        "summary": f"No specific statement found on {topic}.",
                        "source_url": candidate_url(candidate_name)
                    }]
                },
                "type": "no_candidate_match"
            })

       
        # --- Last-resort: keyword-based GPT summary ---
        fallback_topic = detect_topic_from_query(cleaned_query, aliases)
        if fallback_topic:
            print(f"🆘 Last-resort GPT fallback: detected topic '{fallback_topic}'")

            chunks = topic_chunks.get(fallback_topic, [])

            if chunks:
                if len(chunks) > 40:
                    warning = {
                        "candidates": [{
                            "name": "Note",
                            "summary": f"The topic '{fallback_topic}' includes too many sources to summarize directly. Please try a more specific question (e.g., 'special needs in schools').",
                            "source_url": ""
                        }]
                    }
                    log_query_console(query, warning, matched_topic=fallback_topic, response_type="fallback_topic_too_large")
                    return jsonify({
                        "response": warning,
                        "type": "fallback_topic_too_large"
                    })

                try:
                    gpt_summary = summarize_topic_with_gpt(fallback_topic, chunks)
                    response_data = {
                        "candidates": gpt_summary,
                        "topic": fallback_topic
                    }
                    log_query_console(query, response_data, matched_topic=fallback_topic, response_type="gpt_fallback_summary")
                    return jsonify({
                        "response": response_data,
                        "type": "gpt_fallback_summary"
                    })
                except Exception as e:
                    log_query_console(query, f"⚠️ GPT fallback failed: {e}", matched_topic=fallback_topic, response_type="gpt_error")
            else:
                # No topic chunks found, use keyword matcher with GPT summaries over df
                keyword_summary = last_resort_keyword_summary(query, df, fallback_topic=fallback_topic)
                log_query_console(query, keyword_summary, matched_topic=fallback_topic, response_type="keyword_gpt_summary")
                return jsonify({"response": keyword_summary})
        
        # --- Final fallback: use keyword matcher across all embeddings if no topic matched ---
        print("🧭 No alias-based topic detected. Using full-text fallback.")
        keyword_summary = last_resort_keyword_summary(query, df)
        log_query_console(query, keyword_summary, matched_topic="unknown", response_type="keyword_fulltext_summary")
        return jsonify({"response": keyword_summary})

    

    except Exception as e:
        error_message = f"An error occurred: {e}"
        log_query_console(query, error_message, response_type="exception")
        return jsonify({
            "response": error_message,
            "type": "exception"
        }), 500
       

    

















