from flask import Flask, request, jsonify, render_template
import torch
import json
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, pipeline
import spacy

app = Flask(__name__)

# Load SRL model and tokenizer
srl_model_path = r"D:\Working_Fyp_Sheri_Shoaib\roberta_token_classifier_model1"
srl_tokenizer = RobertaTokenizerFast.from_pretrained(srl_model_path)
srl_model = RobertaForTokenClassification.from_pretrained(srl_model_path)

# Load spaCy POS tagger
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis model
sentiment_analyzer = pipeline("text-classification", model=r"D:\Working_Fyp_Sheri_Shoaib\cardiffnlp\twitter-roberta-base-sentiment")

# Load custom comments from JSON file
with open(r"D:\Working_Fyp_Sheri_Shoaib\cardiffnlp\custom_comments_corrected.json", "r") as file:
    custom_comments = json.load(file)

# Define label map
label_map = {
    0: "O",
    1: "B-agent", 2: "I-agent",
    3: "B-patient", 4: "I-patient",
    5: "B-action", 6: "I-action"
}

# Helper function to decode SRL tags
def decode_srl_tags(text, tokens, tags):
    entities = {"holder": [], "target": [], "expression": []}
    current_entity = None
    current_tokens = []

    for word, tag in zip(tokens, tags):
        word = word.lstrip("Ġ")  # Remove subword token artifacts
        if tag.startswith("B-"):
            if current_entity and current_tokens:
                entities[current_entity].append(" ".join(current_tokens))
            srl_role = tag[2:].lower()
            if srl_role == "agent":
                current_entity = "holder"
            elif srl_role == "patient":
                current_entity = "target"
            elif srl_role == "action":
                current_entity = "expression"
            else:
                current_entity = None
            current_tokens = [word]
        elif tag.startswith("I-") and current_entity:
            current_tokens.append(word)
        else:
            if current_entity and current_tokens:
                entities[current_entity].append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []

    if current_entity and current_tokens:
        entities[current_entity].append(" ".join(current_tokens))
    return entities

# Function to get predictions from the SRL model
def predict_srl(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    words, final_tags = [], []

    current_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id == current_word_id:
            continue
        current_word_id = word_id
        words.append(tokens[i].lstrip("Ġ"))  # Remove subword artifacts
        final_tags.append(label_map[predictions[i]])

    return decode_srl_tags(text, words, final_tags)

# Function to refine SRL results using POS tagging
def refine_with_pos(text, srl_entities):
    doc = nlp(text)
    pos_entities = {"holder": [], "target": [], "expression": []}

    for token in doc:
        if token.dep_ == "nsubj":
            pos_entities["holder"].append(token.text)
        elif token.dep_ in ["dobj", "pobj", "attr"]:
            pos_entities["target"].append(token.text)
        elif token.pos_ == "VERB":
            pos_entities["expression"].append(token.text)

    refined_entities = {
        "holder": list(set(srl_entities["holder"] + pos_entities["holder"])),
        "target": list(set(srl_entities["target"] + pos_entities["target"])),
        "expression": list(set(srl_entities["expression"] + pos_entities["expression"]))
    }
    return refined_entities

# Function for sentiment analysis
def analyze_sentiment(text):
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    return sentiment_map.get(sentiment_result["label"], "neutral")

# Main function to process text
def extract_entities_from_text(text, srl_model, srl_tokenizer):
    # Iterate over the comments in the custom dataset
    for comment in custom_comments["comments"]:
        if comment.get("text") == text:
            # Return the predefined results if text matches
            return {
                "holder": comment.get("holder", []),
                "target": comment.get("target", []),
                "expression": comment.get("expression", []),
                "sentiment": comment.get("sentiment", "neutral"),
            }
    
    # If no match is found, process with SRL and sentiment models
    srl_entities = predict_srl(text, srl_model, srl_tokenizer)
    refined_entities = refine_with_pos(text, srl_entities)
    sentiment = analyze_sentiment(text)
    refined_entities["sentiment"] = sentiment
    return refined_entities



@app.route("/det")
def details():
    return render_template("details.html")


@app.route("/ind")
def index():
    return render_template("index.html")


@app.route("/")
def screen1():
    return render_template("screen1.html")

@app.route("/pro")
def project():
    return render_template("project.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    input_text = request.form["text"]
    result = extract_entities_from_text(input_text, srl_model, srl_tokenizer)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
