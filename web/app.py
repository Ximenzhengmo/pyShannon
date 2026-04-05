import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.calculator import CalculatorError, compute_from_payload, get_exam_sample_payload


app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/example/problem-xyz")
def problem_xyz():
    return jsonify({"ok": True, "payload": get_exam_sample_payload()})


@app.post("/api/calculate")
def calculate():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"ok": False, "error": "Request body must be JSON."}), 400

    try:
        result = compute_from_payload(payload)
    except CalculatorError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": "Unexpected server error.",
                "detail": str(exc),
            }
        ), 500

    return jsonify({"ok": True, "result": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
