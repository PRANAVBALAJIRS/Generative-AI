import openai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        language = request.form["language"]
        file = request.files["file"]
        if file:
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.translate("whisper-1", audio_file)

            response = openai.ChatCompletion.create(
                model = "gpt-4",
                messages = [
                    {"role": "system", "content": f"You will be provided with a content in english and your task is to translate it into {language}"},
                    {"role": "user", "content": transcript.text}
                ],
                temperature = 0,
                max_tokens = 512
            )
            return render_template("index.html",
                                   original=transcript["text"],
                                   translated=response["choices"][0]["message"]["content"])
        else:
            return render_template("index.html", error="No file uploaded")
    
    return render_template("index.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True, port = 8080)