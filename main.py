import os
import glob
import base64
import tempfile
import json
import uuid
import numpy as np
import cv2
from flask import Flask, request, redirect, jsonify, send_file
import torch
import torch.nn as nn

app = Flask(__name__)

# 1. Definición de clases reducida solo a la A, B, C, D y E
UNOWN_CLASSES = ['A', 'B', 'C', 'D', 'E']

# Mapeo a símbolos y etiquetas legibles
CLASS_SYMBOLS = {c: c for c in UNOWN_CLASSES}
CLASS_DISPLAY_NAMES = {c: f"Letra '{c}'" for c in UNOWN_CLASSES}

# 2. Arquitectura de la Red Neuronal Convolucional (num_classes=5)
class UnownCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Cargar modelo entrenado en memoria
device = torch.device("cpu")
model = None
model_classes = UNOWN_CLASSES
model_path = os.path.join(os.path.dirname(__file__), "unown_model.pth")

def load_trained_model():
    global model, model_classes
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model_classes = checkpoint.get('class_names', UNOWN_CLASSES)
            loaded_model = UnownCNN(num_classes=len(model_classes))
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model.eval()
            model = loaded_model
            acc = checkpoint.get('accuracy', 'N/A')
            print(f"[INFO] Modelo Unown cargado exitosamente desde '{model_path}'. Precisión: {acc:.2f}%")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo cargar el modelo: {e}")
    else:
        print(f"[INFO] No se encontró el archivo '{model_path}'. La funcionalidad de predicción estará disponible tras entrenar.")

load_trained_model()

# 3. Interfaz HTML modificada con opción de guardado
main_html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Clasificador de Unown (A-E)</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f0f2f5;
      margin: 0;
      padding: 20px;
      color: #333;
    }
    .container {
      max-width: 600px;
      margin: 0 auto;
      background: #ffffff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.1);
      text-align: center;
    }
    h1 {
      margin-top: 0;
      color: #1a1a2e;
    }
    .subtitle {
      color: #666;
      margin-bottom: 25px;
    }
    .workspace {
      display: flex;
      justify-content: center;
      align-items: center;
      margin-bottom: 20px;
    }
    canvas {
      border: 3px solid #222;
      border-radius: 8px;
      background: #fff;
      cursor: crosshair;
      touch-action: none;
    }
    .button-group {
      display: flex;
      justify-content: center;
      gap: 15px;
      margin-bottom: 15px;
    }
    button {
      padding: 12px 20px;
      font-size: 16px;
      font-weight: 600;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-clear { background: #ef4444; color: white; }
    .btn-clear:hover { background: #dc2626; }
    .btn-predict { background: #8b5cf6; color: white; width: 100%; max-width: 250px;}
    .btn-predict:hover { background: #7c3aed; }
    
    .save-group {
      background: #eef2ff;
      border: 1px dashed #4f46e5;
      padding: 15px;
      border-radius: 8px;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
      margin-bottom: 20px;
    }
    .save-group label { font-weight: bold; color: #312e81; }
    .save-group select { padding: 8px 12px; font-size: 16px; border-radius: 4px; border: 1px solid #ccc; }
    .btn-save { background: #10b981; color: white; }
    .btn-save:hover { background: #059669; }

    #pred-result {
      display: none;
      background: #f8fafc;
      border: 2px solid #e2e8f0;
      border-radius: 8px;
      padding: 20px;
      margin-top: 15px;
      text-align: left;
    }
    .pred-highlight {
      font-size: 22px;
      font-weight: bold;
      color: #2563eb;
      margin-bottom: 15px;
      text-align: center;
    }
    .top-list {
      margin-top: 10px;
      list-style: none;
      padding-left: 0;
    }
    .top-list li {
      padding: 8px 10px;
      border-bottom: 1px solid #edf2f7;
      font-size: 18px;
      display: flex;
      justify-content: space-between;
    }
    .top-list li:first-child {
      background-color: #dbeafe;
      border-radius: 5px;
      font-weight: bold;
    }
  </style>
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
</head>
<body onload="InitThis();">
  <div class="container">
    <h1>Clasificador Unown (A-E)</h1>
    <p class="subtitle">Dibuja un símbolo Unown (A, B, C, D o E) en el lienzo.</p>

    <div class="workspace">
      <canvas id="myCanvas" width="200" height="200"></canvas>
    </div>

    <div class="button-group">
      <button type="button" class="btn-clear" onclick="clearArea();">Borrar</button>
      <button type="button" class="btn-predict" onclick="predictDrawing();">Clasificar con IA</button>
    </div>

    <!-- Nueva sección para guardar en el dataset -->
    <div class="save-group">
      <label for="save-class">Clase real:</label>
      <select id="save-class">
        <option value="A">A</option>
        <option value="B">B</option>
        <option value="C">C</option>
        <option value="D">D</option>
        <option value="E">E</option>
      </select>
      <button type="button" class="btn-save" onclick="saveDrawing();">Guardar Muestra</button>
    </div>

    <div id="pred-result">
      <div class="pred-highlight">Resultados Top 3:</div>
      <ul class="top-list" id="pred-top3"></ul>
    </div>
  </div>

  <script>
    var mousePressed = false;
    var lastX, lastY;
    var ctx;

    function InitThis() {
        ctx = document.getElementById('myCanvas').getContext("2d");

        $('#myCanvas').mousedown(function (e) {
            mousePressed = true;
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
        });

        $('#myCanvas').mousemove(function (e) {
            if (mousePressed) {
                Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
            }
        });

        $('#myCanvas').mouseup(function (e) {
            mousePressed = false;
        });
        $('#myCanvas').mouseleave(function (e) {
            mousePressed = false;
        });

        var canvas = document.getElementById('myCanvas');
        canvas.addEventListener('touchstart', function(e) {
            e.preventDefault();
            var rect = canvas.getBoundingClientRect();
            var touch = e.touches[0];
            mousePressed = true;
            Draw(touch.clientX - rect.left, touch.clientY - rect.top, false);
        }, false);

        canvas.addEventListener('touchmove', function(e) {
            e.preventDefault();
            if (mousePressed) {
                var rect = canvas.getBoundingClientRect();
                var touch = e.touches[0];
                Draw(touch.clientX - rect.left, touch.clientY - rect.top, true);
            }
        }, false);

        canvas.addEventListener('touchend', function(e) {
            mousePressed = false;
        }, false);
    }

    function Draw(x, y, isDown) {
        if (isDown) {
            ctx.beginPath();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 14;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.closePath();
            ctx.stroke();
        }
        lastX = x; lastY = y;
    }

    function clearArea() {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        document.getElementById('pred-result').style.display = 'none';
    }

    // Nueva función JavaScript para enviar la imagen y guardarla
    function saveDrawing() {
        var canvas = document.getElementById('myCanvas');
        var imgData = canvas.toDataURL('image/png');
        var selectedClass = document.getElementById('save-class').value;

        $.ajax({
            type: "POST",
            url: "/save",
            data: JSON.stringify({ image: imgData, label: selectedClass }),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(resp) {
                if (resp.error) {
                    alert("Error: " + resp.error);
                } else {
                    // Muestra alerta de éxito y limpia el canvas para el siguiente trazo
                    alert(resp.message);
                    clearArea(); 
                }
            },
            error: function(err) {
                alert("Error al comunicarse con el servidor para guardar.");
            }
        });
    }

    function predictDrawing() {
        var canvas = document.getElementById('myCanvas');
        var imgData = canvas.toDataURL('image/png');

        $.ajax({
            type: "POST",
            url: "/predict",
            data: JSON.stringify({ image: imgData }),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(resp) {
                if (resp.error) {
                    alert("Error: " + resp.error);
                    return;
                }
                var resDiv = document.getElementById('pred-result');
                resDiv.style.display = 'block';

                var listHtml = "";
                resp.top3.forEach(function(item, idx) {
                    listHtml += "<li><span>#" + (idx + 1) + " Unown " + item.display + "</span> <span>" + item.prob + "%</span></li>";
                });
                document.getElementById('pred-top3').innerHTML = listHtml;
            },
            error: function(err) {
                alert("Error al comunicarse con el servidor.");
            }
        });
    }
  </script>
</body>
</html>
"""

@app.route("/")
def main():
    return main_html

# Nueva ruta de Flask para procesar y guardar la imagen localmente
@app.route('/save', methods=['POST'])
def save():
    try:
        data = request.get_json(force=True)
        raw_img = data.get('image', '').replace("data:image/png;base64,", "")
        label = data.get('label', 'A')

        if label not in UNOWN_CLASSES:
            return jsonify({"error": "Clase no permitida."}), 400

        # Crear la ruta del directorio: /dataset/A (o B, C, D, E)
        base_dir = os.path.dirname(__file__)
        target_dir = os.path.join(base_dir, "dataset", label)
        os.makedirs(target_dir, exist_ok=True)

        # Generar un nombre de archivo único usando UUID corto
        filename = f"{label}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(target_dir, filename)

        # Decodificar y escribir el archivo PNG
        with open(file_path, "wb") as fh:
            fh.write(base64.b64decode(raw_img))
        
        print(f"[INFO] Nueva muestra guardada: {file_path}")
        return jsonify({"message": f"¡Imagen guardada con éxito en /dataset/{label}!"})
    
    except Exception as err:
        return jsonify({"error": str(err)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, model_classes
    if model is None:
        load_trained_model()
    if model is None:
        return jsonify({"error": "Modelo no cargado. Por favor entrene la red primero."}), 503

    try:
        data = request.get_json(force=True)
        raw_img = data.get('image', '').replace("data:image/png;base64,", "")
        img_bytes = base64.b64decode(raw_img)
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        if img_np is None:
            return jsonify({"error": "No se pudo decodificar la imagen."}), 400

        if img_np.ndim == 3 and img_np.shape[2] == 4:
            processed = img_np[:, :, 3]
        elif img_np.ndim == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            processed = 255 - gray
        else:
            processed = img_np

        processed = cv2.resize(processed, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:3]
        top3 = []
        for idx in top_indices:
            if idx < len(model_classes):
                cls_name = model_classes[idx]
                top3.append({
                    "class": cls_name,
                    "symbol": CLASS_SYMBOLS.get(cls_name, cls_name),
                    "display": CLASS_DISPLAY_NAMES.get(cls_name, cls_name),
                    "prob": round(float(probs[idx]) * 100, 2)
                })

        best_cls = top3[0]["class"] if top3 else "N/A"
        return jsonify({
            "prediction": best_cls,
            "top3": top3
        })
    except Exception as err:
        return jsonify({"error": str(err)}), 500

if __name__ == "__main__":
    app.run()