import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import cv2
import face_recognition

# Configuration de l'application
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"}

# Fonction pour vérifier les extensions de fichiers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Charger les visages enregistrés et leurs noms
def charger_visages_enregistres(dossier_visages):
    noms = []
    empreintes = []

    for fichier in os.listdir(dossier_visages):
        chemin = os.path.join(dossier_visages, fichier)
        image = face_recognition.load_image_file(chemin)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            empreintes.append(encodings[0])
            noms.append(os.path.splitext(fichier)[0])  # Récupérer le nom sans extension
    return noms, empreintes

# Route principale - Page d'accueil avec deux boutons
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# Route pour la reconnaissance d'image
@app.route("/reconnaissance_image", methods=["GET", "POST"])
def reconnaissance_image():
    uploaded_file_url = None
    resultats = []  # Liste pour stocker les résultats des visages

    if request.method == "POST":
        # Vérifier si un fichier est téléchargé
        if "file" not in request.files:
            return "Aucun fichier sélectionné !", 400
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Charger les visages enregistrés
            path = "C:/Users/brice/OneDrive/Bureau/INDIA 2A/Traitement d'image/Traitement d'image/visage_enregistrés"
            noms_visages, empreintes_visages = charger_visages_enregistres(path)

            # Charger l'image téléchargée
            image = face_recognition.load_image_file(filepath)
            locations_visages = face_recognition.face_locations(image)
            encodings_visages = face_recognition.face_encodings(image, locations_visages)

            # Convertir l'image en format BGR pour OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for (top, right, bottom, left), enc in zip(locations_visages, encodings_visages):
                correspondances = face_recognition.compare_faces(empreintes_visages, enc)
                distances = face_recognition.face_distance(empreintes_visages, enc)

                if len(distances) > 0:
                    meilleur_match_index = np.argmin(distances)
                    if correspondances[meilleur_match_index]:
                        nom = noms_visages[meilleur_match_index]
                    else:
                        nom = "Inconnu"
                else:
                    nom = "Inconnu"

                resultats.append(nom)

                # Dessiner un rectangle rouge autour du visage
                cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
                # Ajouter le nom sous le rectangle
                cv2.putText(image_bgr, nom, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Enregistrer l'image avec les rectangles
            processed_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + filename)
            cv2.imwrite(processed_image_path, image_bgr)

            # URL de l'image avec les rectangles
            uploaded_file_url = url_for("static", filename="uploads/" + "processed_" + filename)

    return render_template("reconnaissance_image.html", file_url=uploaded_file_url, resultats=resultats)

# Route pour la reconnaissance vidéo
@app.route("/reconnaissance_video", methods=["GET", "POST"])
def reconnaissance_video():
    video_file_url = None
    video_results = []  # Liste pour stocker les frames traitées

    if request.method == "POST":
        # Vérifier si un fichier vidéo est téléchargé
        if "file" not in request.files:
            return "Aucun fichier sélectionné !", 400
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Charger les visages enregistrés
            path = "C:/Users/brice/OneDrive/Bureau/INDIA 2A/Traitement d'image/Traitement d'image/visage_enregistrés"
            noms_visages, empreintes_visages = charger_visages_enregistres(path)

            # Lire la vidéo
            cap = cv2.VideoCapture(filepath)
            frame_count = 0
            video_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 == 0:  # Extraire une frame tous les 10 frames pour la reconnaissance
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locations_visages = face_recognition.face_locations(rgb_frame)
                    encodings_visages = face_recognition.face_encodings(rgb_frame, locations_visages)

                    for (top, right, bottom, left), enc in zip(locations_visages, encodings_visages):
                        correspondances = face_recognition.compare_faces(empreintes_visages, enc)
                        distances = face_recognition.face_distance(empreintes_visages, enc)

                        if len(distances) > 0:
                            meilleur_match_index = np.argmin(distances)
                            if correspondances[meilleur_match_index]:
                                nom = noms_visages[meilleur_match_index]
                            else:
                                nom = "Inconnu"
                        else:
                            nom = "Inconnu"

                        # Dessiner un rectangle rouge autour du visage
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Ajouter le nom sous le rectangle
                        cv2.putText(frame, nom, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Sauvegarder la frame traitée
                    frame_filename = f"frame_{frame_count}.jpg"
                    frame_path = os.path.join(app.config["UPLOAD_FOLDER"], frame_filename)
                    cv2.imwrite(frame_path, frame)

                    # Ajouter l'URL de la frame à la liste
                    video_frames.append(url_for("static", filename="uploads/" + frame_filename))

            cap.release()
            video_file_url = video_frames

    return render_template("reconnaissance_video.html", video_frames=video_file_url)

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
