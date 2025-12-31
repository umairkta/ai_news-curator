import subprocess
import json

def run_thinking_model(prompt_json):
    """
    Runs DeepSeek-R1 locally using Ollama and returns JSON output.
    """

    process = subprocess.Popen(
        ["ollama", "run", "deepseek-r1", "--format", "json"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output, error = process.communicate(prompt_json + "\n")

    if error:
        print("Model error:", error)

    return output


def generate_ai_updates():
    """
    Builds the prompt and gets AI news updates.
    """

    prompt = {
    "task": "AI News Curation",
    "instructions": [
        "Generate at least 5 recent AI-related updates",
        "Use only reliable public sources",
        "Provide clear summaries",
        "Classify each update by relevant personas"
    ],
    "personas": [
        "Developers",
        "Investors",
        "Students and Researchers",
        "Founders",
        "Healthcare Professionals",
        "Designers",
        "Journalists",
        "Marketers"
    ],
    "output_format": {
        "title": "",
        "summary": "",
        "claimed_original_source": "",
        "domain": "",
        "relevant_personas": []
    }
}


    prompt_text = json.dumps(prompt, indent=2)
    response = run_thinking_model(prompt_text)

    return response
