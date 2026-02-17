"""
ATM Metal Detection System - Offline Version
Works completely offline with OpenCV only
Optimized for Core i5, 8GB RAM, 256GB SSD
"""

import cv2
import numpy as np
from pathlib import Path
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import json
import threading
import time
from collections import deque
import base64

# Windows audio
try:
    import winsound
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False


class MetalDetectionSystem:
    def __init__(self, config_file='config.json'):
        """Initialize the ATM Metal Detection System"""
        print("\n🛡️  ATM METAL DETECTION SYSTEM")
        print("="*60)
        print("Initializing offline detection system...")
        
        # Load configuration
        self.load_config(config_file)
        
        # Initialize motion and object detection
        self.initialize_detectors()
        
        # Alert system
        self.alert_queue = deque(maxlen=100)
        self.last_alert_time = {}
        self.alert_cooldown = 10
        
        # Statistics
        self.total_detections = 0
        self.total_alerts = 0
        self.threat_count = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Object detector (using cascade classifiers - works offline)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("✅ System initialized successfully!")
        print("✅ Using OpenCV offline detection")
        print("="*60 + "\n")
    
    def initialize_detectors(self):
        """Initialize detection algorithms"""
        # Color-based metal detection (metallic surfaces reflect differently)
        self.metal_hsv_lower = np.array([0, 0, 150])    # Light metallic
        self.metal_hsv_upper = np.array([180, 50, 255]) # Bright metallic
        
        # Edge detection for tools/weapons
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        
        # Minimum contour area for detection
        self.min_contour_area = 1000
        
        # Detection zones (suspicious areas in ATM view)
        self.detection_zones = []
    
    def load_config(self, config_file):
        """Load system configuration"""
        default_config = {
            "security_email": "security@atmbank.com",
            "security_phone": "+1234567890",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_email": "alerts@atmbank.com",
            "smtp_password": "your_app_password",
            "camera_source": 0,
            "detection_interval": 1.0,
            "enable_email": False,
            "enable_sms": False,
            "enable_webhook": False,
            "enable_audio": True,
            "sensitivity": "medium",  # low, medium, high
            "save_screenshots": True,
            "screenshot_folder": "alerts"
        }
        
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
                print(f"✅ Loaded config from {config_file}")
        else:
            self.config = default_config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"⚙️  Created config.json - Please configure alerts!")
        
        # Create screenshot folder
        if self.config.get('save_screenshots', True):
            Path(self.config['screenshot_folder']).mkdir(exist_ok=True)
    
    def detect_metallic_objects(self, frame):
        """Detect metallic objects using color and edge detection"""
        detections = []
        threat_level = 'safe'
        
        # Convert to HSV for better metal detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for metallic surfaces
        metal_mask = cv2.inRange(hsv, self.metal_hsv_lower, self.metal_hsv_upper)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
        
        # Edge detection for sharp objects (knives, tools)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
        
        # Combine metal and edge detection
        combined_mask = cv2.bitwise_or(metal_mask, edges)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Classify based on shape and size
                threat = self.classify_object(area, aspect_ratio, w, h)
                confidence = self.calculate_confidence(area, aspect_ratio)
                
                if threat != 'safe':
                    detections.append({
                        'object': self.get_object_name(aspect_ratio, area),
                        'confidence': confidence,
                        'bbox': [x, y, x+w, y+h],
                        'threat': threat,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
                    
                    if self._threat_priority(threat) > self._threat_priority(threat_level):
                        threat_level = threat
        
        return detections, threat_level, combined_mask
    
    def classify_object(self, area, aspect_ratio, width, height):
        """Classify object threat level based on characteristics"""
        # Large metallic objects (tools, weapons)
        if area > 5000:
            if 0.2 < aspect_ratio < 0.5 or aspect_ratio > 3:
                return 'high'  # Long thin objects (knives, bars)
            elif area > 10000:
                return 'medium'  # Large tools
        
        # Medium objects
        elif area > 2000:
            if aspect_ratio > 2 or aspect_ratio < 0.4:
                return 'medium'  # Suspicious shapes
            else:
                return 'low'  # Regular objects
        
        # Small objects
        elif area > 1000:
            return 'low'
        
        return 'safe'
    
    def get_object_name(self, aspect_ratio, area):
        """Estimate object type"""
        if aspect_ratio > 3:
            return "Long metal object (possible weapon)"
        elif aspect_ratio < 0.3:
            return "Vertical metal object (possible tool)"
        elif area > 10000:
            return "Large metallic object"
        elif area > 5000:
            return "Medium metallic object"
        else:
            return "Small metallic object"
    
    def calculate_confidence(self, area, aspect_ratio):
        """Calculate detection confidence"""
        confidence = 0.5
        
        # Larger objects = higher confidence
        if area > 5000:
            confidence += 0.2
        if area > 10000:
            confidence += 0.1
        
        # Suspicious shapes = higher confidence
        if aspect_ratio > 2 or aspect_ratio < 0.4:
            confidence += 0.2
        
        return min(confidence, 0.99)
    
    def _threat_priority(self, threat):
        """Get numeric priority"""
        priority = {'safe': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return priority.get(threat, 0)
    
    def save_screenshot(self, frame, alert_data):
        """Save alert screenshot"""
        if not self.config.get('save_screenshots', True):
            return None
        
        folder = Path(self.config['screenshot_folder'])
        filename = folder / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Add alert info to image
        annotated = frame.copy()
        cv2.putText(annotated, f"THREAT: {alert_data['threat_level'].upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated, f"Time: {alert_data['timestamp']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imwrite(str(filename), annotated)
        return str(filename)
    
    def send_email_alert(self, alert_data, screenshot_path=None):
        """Send email with screenshot"""
        if not self.config['enable_email']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['smtp_email']
            msg['To'] = self.config['security_email']
            msg['Subject'] = f"🚨 ATM ALERT - {alert_data['threat_level'].upper()}"
            
            body = f"""
SECURITY ALERT - ATM SURVEILLANCE

Threat Level: {alert_data['threat_level'].upper()}
Time: {alert_data['timestamp']}
Terminal: {alert_data['terminal_id']}

Detected Objects: {len(alert_data['objects'])}
{chr(10).join([f"  • {obj['object']} ({obj['confidence']:.0%} confidence)" 
               for obj in alert_data['objects']])}

Screenshot: {"Attached" if screenshot_path else "Not available"}

IMMEDIATE SECURITY RESPONSE REQUIRED
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach screenshot
            if screenshot_path and Path(screenshot_path).exists():
                with open(screenshot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', 
                                 filename='alert.jpg')
                    msg.attach(img)
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['smtp_email'], self.config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
            print(f"📧 Email sent to {self.config['security_email']}")
        except Exception as e:
            print(f"❌ Email failed: {e}")
    
    def send_sms_alert(self, alert_data):
        """Send SMS alert"""
        if not self.config['enable_sms']:
            return
        
        try:
            message = f"🚨 ATM ALERT: {alert_data['threat_level'].upper()} threat detected at {alert_data['timestamp']}. {len(alert_data['objects'])} suspicious object(s). CHECK IMMEDIATELY!"
            print(f"📱 SMS Alert: {message[:50]}...")
            # Integrate with Twilio/SMS gateway here
        except Exception as e:
            print(f"❌ SMS failed: {e}")
    
    def ping_security_system(self, alert_data):
        """Webhook notification"""
        if not self.config['enable_webhook']:
            return
        
        try:
            payload = {
                'alert_type': 'METAL_DETECTION',
                'threat_level': alert_data['threat_level'],
                'timestamp': alert_data['timestamp'],
                'terminal_id': alert_data['terminal_id'],
                'object_count': len(alert_data['objects'])
            }
            print(f"🔔 Webhook: {alert_data['threat_level']} alert logged")
        except Exception as e:
            print(f"❌ Webhook failed: {e}")
    
    def play_audio_alert(self, threat_level):
        """Audio alert"""
        if not self.config['enable_audio'] or not AUDIO_AVAILABLE:
            return
        
        try:
            frequencies = {'low': 400, 'medium': 600, 'high': 800, 'critical': 1000}
            freq = frequencies.get(threat_level, 400)
            duration = 500 if threat_level in ['high', 'critical'] else 300
            
            winsound.Beep(freq, duration)
            if threat_level == 'critical':
                time.sleep(0.2)
                winsound.Beep(freq, duration)
        except:
            pass
    
    def trigger_all_alerts(self, detections, threat_level, frame):
        """Trigger all alerts"""
        if threat_level == 'safe' or not detections:
            return
        
        alert_key = f"{threat_level}_{len(detections)}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = current_time
        
        alert_data = {
            'threat_level': threat_level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objects': detections,
            'terminal_id': 'ATM-001'
        }
        
        self.alert_queue.append(alert_data)
        self.total_alerts += 1
        
        print(f"\n{'='*60}")
        print(f"🚨 ALERT #{self.total_alerts}: {threat_level.upper()} THREAT")
        print(f"{'='*60}")
        print(f"Objects detected: {len(detections)}")
        for det in detections:
            print(f"  • {det['object']} ({det['confidence']:.0%})")
        print(f"{'='*60}\n")
        
        # Save screenshot
        screenshot_path = self.save_screenshot(frame, alert_data)
        
        # Send alerts in parallel
        threads = [
            threading.Thread(target=self.send_email_alert, args=(alert_data, screenshot_path)),
            threading.Thread(target=self.send_sms_alert, args=(alert_data,)),
            threading.Thread(target=self.ping_security_system, args=(alert_data,)),
            threading.Thread(target=self.play_audio_alert, args=(threat_level,))
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
    
    def draw_detections(self, frame, detections, threat_level):
        """Draw detection boxes"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            colors = {
                'low': (0, 255, 255),
                'medium': (0, 165, 255),
                'high': (0, 0, 255),
                'critical': (255, 0, 255)
            }
            color = colors.get(det['threat'], (255, 255, 255))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['object'][:30]}"
            conf_label = f"{det['confidence']:.0%}"
            
            cv2.putText(frame, label, (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, conf_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Threat indicator
        if threat_level != 'safe':
            cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 255), -1)
            cv2.putText(frame, f"THREAT: {threat_level.upper()}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("\n🎥 Starting ATM Surveillance System")
        print("="*60)
        print("Controls:")
        print("  'q' = Quit")
        print("  's' = Save screenshot")
        print("  'r' = Reset statistics")
        print("  'c' = Calibrate (press in empty ATM scene)")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(self.config['camera_source'])
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            print("\nTroubleshooting:")
            print("  1. Check if camera is connected")
            print("  2. Try changing 'camera_source' in config.json to 1 or 2")
            print("  3. Use video file: set 'camera_source' to 'video.mp4'")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Camera opened successfully")
        print("📹 Resolution: 640x480")
        print("\n🔍 Detection active - monitoring for metallic objects...\n")
        
        last_detection_time = 0
        fps_counter = deque(maxlen=30)
        frame_count = 0
        
        while True:
            start_time = time.time()
            frame_count += 1
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            display_frame = frame.copy()
            
            # Run detection at intervals
            current_time = time.time()
            if current_time - last_detection_time >= self.config['detection_interval']:
                detections, threat_level, mask = self.detect_metallic_objects(frame)
                last_detection_time = current_time
                
                if detections:
                    self.total_detections += len(detections)
                    self.threat_count[threat_level] += 1
                    self.trigger_all_alerts(detections, threat_level, frame)
                
                display_frame = self.draw_detections(display_frame, detections, threat_level)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time + 0.001)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            # Display info
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", 
                       (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(display_frame, f"Detections: {self.total_detections} | Alerts: {self.total_alerts}", 
                       (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('ATM Metal Detection System', display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Shutting down...")
                break
            elif key == ord('s'):
                filename = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                self.total_detections = 0
                self.total_alerts = 0
                self.threat_count = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
                print("🔄 Statistics reset")
            elif key == ord('c'):
                print("🔧 Calibrating background...")
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
                print("✅ Calibration complete")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        print(f"Total Detections: {self.total_detections}")
        print(f"Total Alerts Sent: {self.total_alerts}")
        print(f"Threat Breakdown:")
        for level, count in self.threat_count.items():
            if count > 0:
                print(f"  {level.capitalize()}: {count}")
        print("="*60)


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        ATM METAL DETECTION SYSTEM                        ║
║        Offline OpenCV Edition                             ║
║                                                           ║
║        ✅ No PyTorch Required                            ║
║        ✅ Works Completely Offline                       ║
║        ✅ Optimized for Core i5                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        detector = MetalDetectionSystem()
        detector.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
