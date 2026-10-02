from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

app = Flask(__name__)

MODELS = {}
LABELS = {}


def load_model(game):
    if game in MODELS:
        return

    model_path = f"models/{game}/model.tflite"
    label_path = f"models/{game}/labels.txt"

    if not os.path.exists(model_path):
        raise Exception(f"Model not found: {game}")

    interpreter = tf.lite.Interpreter(
        model_path=model_path
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(label_path, "r", encoding="utf-8") as f:
        labels = [
            line.strip()
            for line in f.readlines()
        ]

    MODELS[game] = {
        "interpreter": interpreter,
        "input": input_details,
        "output": output_details
    }

    LABELS[game] = labels


@app.post("/predict")
def predict():

    game = request.args.get("game")

    if not game:
        return jsonify({
            "error": "game is required"
        }), 400

    try:
        load_model(game)
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    model = MODELS[game]

    image = Image.open(
        io.BytesIO(request.data)
    ).convert("RGB")

    input_detail = model["input"][0]

    height = input_detail["shape"][1]
    width = input_detail["shape"][2]

    image = image.resize(
        (width, height)
    )

    image = np.array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    # 일반적인 float32 모델
    image = image.astype(np.float32) / 255.0

    interpreter = model["interpreter"]

    interpreter.set_tensor(
        input_detail["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        model["output"][0]["index"]
    )[0]

    labels = LABELS[game]

    results = []

    for i, score in enumerate(output):
        results.append({
            "label": labels[i],
            "confidence": float(score)
        })

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return jsonify({
        "game": game,
        "result": results[0],
        "predictions": results
    })


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


app.run(
    host="0.0.0.0",
    port=8000
)