import re
import os
import json
import pickle
import difflib
import nltk
from nltk.tokenize import sent_tokenize
from topics import aliases
from openai import OpenAI

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

client = OpenAI()

# Load embeddings data
with open("embeddings.pkl", "rb") as f:
    df = pickle.load(f)

# Load cache for per-topic summaries
SUMMARY_CACHE_PATH = "summary_cache.json"
if os.path.exists(SUMMARY_CACHE_PATH):
    with open(SUMMARY_CACHE_PATH, "r") as f:
        topic_summary_cache = json.load(f)
else:
    topic_summary_cache = {}

# Standardize column names
if "Candidate Name" in df.columns:
    df.rename(columns={
        "Candidate Name": "name",
        "Text": "text",
        "URL": "source_url"
    }, inplace=True)
    print("📊 Columns in DF:", df.columns.tolist())
    print("🔎 Sample row:", df.iloc[0])

# Load precomputed topic chunks
with open("topic_chunks.json", "r") as f:
    topic_chunks = json.load(f)
print(f"✅ Loaded {len(topic_chunks)} topics from topic_chunks.json")

# Load summary and stance caches if available
try:
    with open("topic_summary_cache.pkl", "rb") as f:
        topic_summary_cache = pickle.load(f)
except FileNotFoundError:
    topic_summary_cache = {}

try:
    with open("stance_cache.pkl", "rb") as f:
        stance_cache = pickle.load(f)
except FileNotFoundError:
    stance_cache = {}

# Default to GPT-4 if available, else GPT-3.5
MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

def detect_topic_from_query(query, aliases):
    """
    Attempts to match a query to one of the known topic aliases.
    """
    query_lower = query.lower()
    for topic, alias_list in aliases.items():
        if topic.lower() in query_lower:
            return topic
        for alias in alias_list:
            if alias.lower() in query_lower:
                return topic
    return None

def get_model():
    try:
        openai.Model.retrieve("gpt-4")
        return "gpt-4"
    except:
        return "gpt-3.5-turbo"

def classify_policy_stance(topic, df, position_keywords, batch_size=5):


    results = []
    candidates = []

    # Gather all candidates with relevant topic content
    for _, row in df.iterrows():
        name = row.get("name", "")
        text = row.get("text", "")
        if not isinstance(text, str):
            continue

        keywords = aliases.get(topic.lower(), [topic.lower()])
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        relevant = [
            p for p in paragraphs
            if any(re.search(rf"\b{re.escape(kw)}\b", p, re.IGNORECASE) for kw in keywords)
        ]
        if not relevant:
            continue

        combined = " ".join(relevant)
        candidates.append((name, combined))

    if not candidates:
        return "No relevant candidate positions found on this topic."

    # Batch process
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]

        batch_text = "\n\n".join(
            f"{name}:\n{statement}" for name, statement in batch
        )

        prompt = f"""
Below are statements from political candidates about '{topic}'.

For each candidate, determine if they SUPPORT, OPPOSE, or express NO CLEAR STANCE on the topic. For each, reply in this format:

Candidate Name: [Stance] - [Brief explanation]

Statements:
{batch_text}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You analyze political candidate positions and determine their stance on a given topic."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0,
                max_tokens=800,
            )
            result_text = response.choices[0].message.content.strip()
            results.append(result_text)

        except Exception as e:
            results.append(f"❌ Error processing batch: {str(e)}")

    return "\n\n".join(results)

def summarize_topic_by_candidate(topic, chunks, batch_size=5):
    
    if not chunks:
        return [{"message": f"No candidate statements found on {topic}."}]

    batch_summaries = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        combined_text = "\n\n".join(
            [f"{chunk['name']}: {chunk['text']}" for chunk in batch if chunk.get("text")]
        )

        prompt = f"""
        Please summarize each of the following political candidate's positions on '{topic}' individually and clearly:

        {combined_text}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You summarize candidate views by individual."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.5,
            )
            batch_summary_text = response.choices[0].message.content.strip()

            # Heuristically split summaries
            for summary in batch_summary_text.split("\n\n"):
                if ":" in summary:
                    name, text = summary.split(":", 1)
                    batch_summaries.append({
                        "name": name.strip(),
                        "summary": text.strip()
                    })

        except Exception as e:
            batch_summaries.append({
                "name": f"Batch {i // batch_size + 1}",
                "summary": f"\u274c GPT error: {str(e)}"
            })

           


    return batch_summaries


def summarize_topic_with_gpt(topic, chunks):
    if not chunks:
        return f"No candidate statements found on {topic}."

    summaries = []

    for chunk in chunks:
        candidate_name = chunk["name"]
        candidate_text = chunk["text"]
        source_url = chunk.get("source_url", "No link")

        # Compose a concise instruction per candidate
        user_prompt = (
            f"This is a candidate's statement on the topic of {topic}:\n\n"
            f"{candidate_text}\n\n"
            "Summarise their stance clearly in 1-2 sentences. Be factual. Avoid repetition. Mention the candidate's name at the start."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a political assistant summarising candidate views."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
            )
            summary = response.choices[0].message.content.strip()
            summaries.append(f"- [{candidate_name}]({source_url}): {summary}")
        except Exception as e:
            summaries.append(f"- {candidate_name}: ❌ Error summarising statement. ({e})")

    return "\n\n".join(summaries)
    

def get_most_relevant_chunk(topic, topic_chunks):
    topic = topic.lower()
    best_match = None
    best_score = 0

    for t, chunks in topic_chunks.items():
        score = difflib.SequenceMatcher(None, topic, t.lower()).ratio()
        if score > best_score:
            best_match = chunks
            best_score = score

    return best_match

def summarize_candidate_topic(candidate_name, topic, df):


    for _, row in df.iterrows():
        if row.get("name", "").lower() == candidate_name.lower():
            text = row.get("text", "")
            if not isinstance(text, str):
                return f"No available text for {candidate_name}."

            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            keywords = aliases.get(topic.lower(), [topic.lower()])

            # Match paragraphs that mention any keyword for the topic
            relevant_paragraphs = [
                p for p in paragraphs
                if any(re.search(rf"\b{re.escape(kw)}\b", p, re.IGNORECASE) for kw in keywords)
            ]

            if not relevant_paragraphs:
                return f"No relevant content found for {candidate_name} on {topic}."

            summary_input = "\n".join(relevant_paragraphs)
            prompt = f"""
            Please summarize {candidate_name}'s views on '{topic}' based on the following text:

            {summary_input}
            """

            messages = [
                {"role": "system", "content": "You summarize political candidate views on a topic."},
                {"role": "user", "content": prompt.strip()}
            ]

            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"❌ GPT error: {str(e)}"

    return f"No relevant response found for {candidate_name} on {topic}."

def summarize_topic(topic):
    if topic in topic_summary_cache:
        return topic_summary_cache[topic]

    matched_chunks = []
    for t_key, t_chunks in topic_chunks.items():
        if t_key == topic or topic in aliases.get(t_key, []):
            matched_chunks.extend(t_chunks)

    if not matched_chunks:
        return f"No information available for topic '{topic}'."

    summary = summarize_topic_with_gpt(topic, "\n".join(matched_chunks))
    topic_summary_cache[topic] = summary
    with open("topic_summary_cache.pkl", "wb") as f:
        pickle.dump(topic_summary_cache, f)

    return summary

