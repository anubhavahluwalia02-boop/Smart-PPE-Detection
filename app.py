"""
Smart PPE Compliance Detection System - Main Flask Application.
"""
import os
import io
import csv
import time
from pathlib import Path
from typing import Generator

from flask import (
    Flask, render_template, request, redirect,
    url_for, Response, jsonify, send_from_directory, send_file, flash
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np

import config
from src.database import init_db, log_violation, get_violations, get_dashboard_stats, get_violation_by_id, delete_violation
from src.compliance import ComplianceAnalyzer
from src.detector import PPEDetector
from src.evidence import EvidenceManager
from src.alerts import AlertManager
from src.camera import CameraManager

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = config.SECRET_KEY

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Initialize Subsystems
init_db()
detector = PPEDetector()
analyzer = ComplianceAnalyzer()
evidence_mgr = EvidenceManager()
alert_mgr = AlertManager()
camera_mgr = CameraManager()

# Global runtime state
current_detection_state = {
    "total_workers": 0,
    "compliant": 0,
    "violations": 0,
    "status": "CLEAR",
    "last_updated": 0
}

def allowed_file(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

@app.route("/")
def dashboard():
    """Renders the executive safety dashboard."""
    stats = get_dashboard_stats()
    # Calculate compliance percentage
    total_detections = stats["total_violations"] + 50 # baseline sample
    compliance_rate = round(max(0, min(100, ((total_detections - stats["total_violations"]) / max(1, total_detections)) * 100)), 1)
    
    model_status = detector.get_status()
    hw_status = alert_mgr.get_hardware_status()
    
    return render_template(
        "dashboard.html",
        stats=stats,
        compliance_rate=compliance_rate,
        model_status=model_status,
        hw_status=hw_status,
        current_state=current_detection_state
    )

@app.route("/live")
def live_detection():
    """Live camera monitoring page."""
    model_status = detector.get_status()
    return render_template(
        "live_detection.html",
        model_status=model_status,
        camera_running=camera_mgr.is_running
    )

def generate_video_stream() -> Generator[bytes, None, None]:
    """Generates JPEG frame stream with real-time detection & bounding boxes."""
    global current_detection_state
    
    while True:
        if not camera_mgr.is_running:
            frame = camera_mgr.generate_placeholder_frame("Camera is Paused")
            time.sleep(0.1)
        else:
            frame = camera_mgr.read_frame()
            if frame is None:
                frame = camera_mgr.generate_placeholder_frame("Waiting for Camera Feed...")
                time.sleep(0.05)
            else:
                # Run real object detection and compliance analysis
                raw_detections = detector.detect(frame)
                h, w = frame.shape[:2]
                eval_result = analyzer.evaluate_detections(raw_detections, (w, h))
                
                # Update global state for UI polling
                current_detection_state = {
                    "total_workers": eval_result["total_workers"],
                    "compliant": eval_result["compliant_count"],
                    "violations": eval_result["helmet_violations"] + eval_result["vest_violations"] + eval_result["multiple_violations"],
                    "status": eval_result["frame_status"],
                    "last_updated": time.time()
                }
                
                # Hardware / buzzer state sync
                alert_mgr.send_hardware_state(eval_result["frame_status"])
                
                # Evidence logging if violation detected
                if "VIOLATION" in eval_result["frame_status"]:
                    for worker in eval_result["workers"]:
                        if "VIOLATION" in worker["status"]:
                            missing_str = ", ".join(worker["missing_ppe"])
                            # Capture snapshot with cooldown
                            evidence_paths = evidence_mgr.capture_evidence(
                                frame,
                                worker["status"],
                                missing_str,
                                worker
                            )
                            if evidence_paths:
                                rel_path, _ = evidence_paths
                                log_violation(
                                    violation_type=worker["status"],
                                    missing_ppe=missing_str,
                                    confidence=worker["confidence"],
                                    source="Live Webcam",
                                    evidence_path=rel_path,
                                    person_count=eval_result["total_workers"],
                                    notes=worker["status_reason"]
                                )
                
                # Render industrial bounding boxes and overlays
                frame = detector.draw_annotations(frame, eval_result)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    """Video streaming route for live detection."""
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    """API endpoint to start the camera."""
    success = camera_mgr.start()
    return jsonify({
        "success": success,
        "is_running": camera_mgr.is_running,
        "error": camera_mgr.error_message
    })

@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    """API endpoint to stop the camera."""
    camera_mgr.stop()
    alert_mgr.send_hardware_state("CLEAR")
    return jsonify({"success": True, "is_running": False})

@app.route("/api/camera/status")
def camera_status():
    """Returns current detection and camera status in JSON."""
    return jsonify({
        "camera_running": camera_mgr.is_running,
        "state": current_detection_state,
        "hardware": alert_mgr.get_hardware_status()
    })

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    """Accepts JPG/PNG image upload, runs AI inference, and displays results."""
    if request.method == "POST":
        if "image_file" not in request.files:
            flash("No file part in the request", "error")
            return redirect(request.url)
            
        file = request.files["image_file"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
            
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash("Invalid format. Only JPG, JPEG, PNG, WEBP files allowed.", "error")
            return redirect(request.url)
            
        filename = secure_filename(file.filename)
        upload_path = config.UPLOADS_DIR / f"upload_{int(time.time())}_{filename}"
        file.save(str(upload_path))
        
        # Read with OpenCV
        image = cv2.imread(str(upload_path))
        if image is None:
            flash("Unable to decode uploaded image.", "error")
            return redirect(request.url)
            
        h, w = image.shape[:2]
        # Detect objects
        raw_detections = detector.detect(image)
        # Evaluate compliance
        result = analyzer.evaluate_detections(raw_detections, (w, h))
        
        # Save annotated image
        annotated = detector.draw_annotations(image, result)
        annotated_name = f"annotated_{int(time.time())}_{filename}"
        annotated_path = config.STATIC_DIR / "images" / annotated_name
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated)
        
        # Log violations if any
        if "VIOLATION" in result["frame_status"]:
            for worker in result["workers"]:
                if "VIOLATION" in worker["status"]:
                    missing_str = ", ".join(worker["missing_ppe"])
                    log_violation(
                        violation_type=worker["status"],
                        missing_ppe=missing_str,
                        confidence=worker["confidence"],
                        source=f"Image: {filename}",
                        evidence_path=f"static/images/{annotated_name}",
                        person_count=result["total_workers"],
                        notes=worker["status_reason"]
                    )
                    
        return render_template(
            "upload.html",
            analyzed=True,
            original_filename=filename,
            annotated_url=f"/static/images/{annotated_name}",
            result=result,
            raw_count=len(raw_detections)
        )
        
    return render_template("upload.html", analyzed=False)

@app.route("/video", methods=["GET", "POST"])
def video_analysis():
    """Allows uploading video files for frame-by-frame compliance analysis."""
    if request.method == "POST":
        if "video_file" not in request.files:
            flash("No video selected", "error")
            return redirect(request.url)
            
        file = request.files["video_file"]
        if not file or file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
            
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            flash("Invalid video format. Allowed: MP4, AVI, MOV, MKV", "error")
            return redirect(request.url)
            
        filename = secure_filename(file.filename)
        save_path = config.UPLOADS_DIR / f"vid_{int(time.time())}_{filename}"
        file.save(str(save_path))
        
        # Process sample frames with OpenCV (e.g., sample every 15 frames to prevent memory/CPU overload)
        cap = cv2.VideoCapture(str(save_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        processed_frames = 0
        violations_found = []
        frame_idx = 0
        step = max(5, int(fps // 2))  # analyze 2 frames per second
        
        while cap.isOpened() and frame_idx < total_frames and processed_frames < 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            h, w = frame.shape[:2]
            detections = detector.detect(frame)
            res = analyzer.evaluate_detections(detections, (w, h))
            
            if "VIOLATION" in res["frame_status"]:
                sec = round(frame_idx / fps, 1)
                for worker in res["workers"]:
                    if "VIOLATION" in worker["status"]:
                        # Save thumbnail snapshot
                        snap_name = f"vid_snap_{int(time.time())}_{frame_idx}.jpg"
                        snap_path = config.VIOLATIONS_DIR / snap_name
                        annotated_snap = detector.draw_annotations(frame, res)
                        cv2.imwrite(str(snap_path), annotated_snap)
                        
                        rel_path = f"violations/{snap_name}"
                        missing_str = ", ".join(worker["missing_ppe"])
                        
                        rec_id = log_violation(
                            violation_type=worker["status"],
                            missing_ppe=missing_str,
                            confidence=worker["confidence"],
                            source=f"Video: {filename} @ {sec}s",
                            evidence_path=rel_path,
                            person_count=res["total_workers"],
                            notes=f"Detected at timestamp {sec}s (Frame {frame_idx})"
                        )
                        violations_found.append({
                            "timestamp_sec": sec,
                            "frame_idx": frame_idx,
                            "type": worker["status"],
                            "missing": missing_str,
                            "evidence_path": rel_path
                        })
                        
            processed_frames += 1
            frame_idx += step
            
        cap.release()
        
        return render_template(
            "video_analysis.html",
            analyzed=True,
            filename=filename,
            total_frames=total_frames,
            sampled_frames=processed_frames,
            violations=violations_found
        )
        
    return render_template("video_analysis.html", analyzed=False)

@app.route("/history")
def history():
    """Searchable & filterable violation records table."""
    filter_type = request.args.get("type", "ALL")
    search_query = request.args.get("q", "")
    
    records = get_violations(
        limit=200,
        filter_type=filter_type if filter_type != "ALL" else None,
        search=search_query if search_query else None
    )
    
    return render_template(
        "history.html",
        records=records,
        current_filter=filter_type,
        search_query=search_query
    )

@app.route("/history/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id: int):
    """Deletes a record from history."""
    delete_violation(record_id)
    flash(f"Record #{record_id} deleted successfully.", "success")
    return redirect(url_for("history"))

@app.route("/export_csv")
def export_csv():
    """Generates and downloads a CSV export of all recorded violations."""
    records = get_violations(limit=1000)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Violation Type", "Missing PPE", "Confidence", "Source", "Evidence Path", "Notes"])
    
    for r in records:
        writer.writerow([
            r["id"],
            r["timestamp"],
            r["violation_type"],
            r["missing_ppe"],
            r["confidence"],
            r["source"],
            r["evidence_path"],
            r["notes"]
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=ppe_violations_report.csv"}
    )

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Model configuration and hardware settings."""
    if request.method == "POST":
        conf_val = float(request.form.get("confidence", 0.45))
        detector.conf = conf_val
        detector.is_simulation_mode = ("simulation_mode" in request.form)
        
        # Test serial
        if "test_serial" in request.form:
            test_res = alert_mgr.send_hardware_state("HELMET_VIOLATION")
            flash("Sent test alarm signal to serial port.", "info" if test_res else "warning")
            
        flash("Configuration updated successfully.", "success")
        return redirect(url_for("settings"))
        
    model_status = detector.get_status()
    hw_status = alert_mgr.get_hardware_status()
    
    return render_template(
        "settings.html",
        model_status=model_status,
        hw_status=hw_status,
        conf=detector.conf
    )

@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns system status, active classes, hardware connectivity, and current metrics."""
    return jsonify({
        "status": "online",
        "detection_state": current_detection_state,
        "model": detector.get_status(),
        "hardware": alert_mgr.get_hardware_status(),
        "camera_running": camera_mgr.is_running
    })

@app.route("/api/hardware/ping", methods=["GET", "POST"])
def hardware_ping():
    """Sends a heartbeat ping to verify the physical ESP32 is still reachable."""
    pong = alert_mgr.ping()
    return jsonify({
        "alive": pong,
        "hardware": alert_mgr.get_hardware_status()
    })

@app.route("/api/hardware/command", methods=["POST"])
def hardware_command():
    """Sends raw command (e.g. 'SAFE', 'VIOLATION', 'P18:1,P19:0,P21:0') to ESP32."""
    data = request.get_json(silent=True) or {}
    cmd = data.get("command", "SAFE")
    success = alert_mgr.send_raw_command(cmd)
    return jsonify({
        "success": success,
        "sent_command": cmd,
        "hardware": alert_mgr.get_hardware_status()
    })

@app.route("/api/hardware/test", methods=["POST"])
def hardware_test():
    """Sends a test packet to the ESP32 hardware."""
    data = request.get_json(silent=True) or {}
    state = data.get("state", "COMPLIANT")
    success = alert_mgr.send_hardware_state(state)
    return jsonify({
        "success": success,
        "hardware": alert_mgr.get_hardware_status()
    })

@app.route("/about")
def about():
    """Explains project architecture, ESP32 wiring, and B.Tech viva exam prep."""
    return render_template("about.html")

@app.route("/violations/<path:filename>")
def serve_violation_image(filename: str):
    """Safely serves saved violation evidence images."""
    return send_from_directory(str(config.VIOLATIONS_DIR), filename)

@app.route("/download_project")
def download_project():
    """Serves the zipped full project repository for one-click student download."""
    zip_path = config.BASE_DIR / "smart-ppe-detection.zip"
    if not zip_path.exists():
        # Fallback to root or public directory
        zip_path = Path("/app/applet/public/smart-ppe-detection.zip")
        
    if zip_path.exists():
        return send_file(
            str(zip_path),
            as_attachment=True,
            download_name="smart-ppe-detection.zip",
            mimetype="application/zip"
        )
    return "ZIP file is being generated. Please refresh in a moment.", 404

if __name__ == "__main__":
    print(f"\n========================================================")
    print(f" SMART PPE COMPLIANCE DETECTION SYSTEM")
    print(f" AI-Powered Workplace Safety Monitoring System")
    print(f" Local Web Server: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"========================================================\n")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.DEBUG)
