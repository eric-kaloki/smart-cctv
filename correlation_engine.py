import time
import math
import requests
from models import DetectionEvent, Trail

class CorrelationEngine:
    def __init__(self, camera_map, bot_token="", chat_id=""):
        self.camera_map = camera_map
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.active_trails = []
        
        # Thresholds for heuristic matching
        self.MAX_COLOR_DISTANCE = 60.0  
        self.MAX_ASPECT_RATIO_DIFF = 0.20 

    def process_event(self, new_event: DetectionEvent):
        matched_trail = None
        
        for trail in self.active_trails:
            if trail.status != "active":
                continue
                
            last_event = trail.events[-1]
            
            # RULE 1: Physical Adjacency
            current_cam_info = self.camera_map["cameras"].get(last_event.camera_id, {})
            if new_event.camera_id not in current_cam_info.get("adjacent_to", []) and new_event.camera_id != last_event.camera_id:
                 continue 

            # RULE 2: Transit Time Logic
            if not self._is_valid_transit(last_event, new_event):
                continue # Rejected! Time doesn't make sense physically.

            # RULE 3: Appearance Match
            if self._is_appearance_match(last_event.appearance_vector, new_event.appearance_vector):
                matched_trail = trail
                break

        if matched_trail:
            matched_trail.add_event(new_event)
            if new_event.camera_id != matched_trail.events[-2].camera_id:
                print(f"🔗 [ENGINE] Trail {matched_trail.trail_id} traversed to {new_event.camera_id}!")
                
            self._check_incident_rules(matched_trail)
        else:
            new_trail = Trail()
            new_trail.add_event(new_event)
            self.active_trails.append(new_trail)
            print(f"🆕 [ENGINE] New Trail Started ({new_trail.trail_id}) at {new_event.camera_id}")

        self._cleanup_stale_trails()

    def _is_valid_transit(self, last_event: DetectionEvent, new_event: DetectionEvent):
        """Checks if the travel time between two cameras makes physical sense."""
        # If they are still on the same camera, time is always valid
        if last_event.camera_id == new_event.camera_id:
            return True

        transit_key = f"{last_event.camera_id}_to_{new_event.camera_id}"
        transit_info = self.camera_map.get("transit_model", {}).get(transit_key)
        
        if not transit_info:
            return True # Fallback if no specific config exists

        time_diff = new_event.timestamp - last_event.timestamp
        expected = transit_info["expected_seconds"]
        tolerance = transit_info["tolerance_seconds"]
        
        min_time = expected - tolerance
        max_time = expected + tolerance

        if min_time <= time_diff <= max_time:
            return True
        else:
            print(f"⏱️ [REJECTED] Transit {transit_key} took {time_diff:.1f}s. Expected {expected}s. (Likely a different person!)")
            return False

    def _is_appearance_match(self, vec1, vec2):
        if vec1["type"] != "heuristic" or vec2["type"] != "heuristic":
            return False 

        ar_diff = abs(vec1["aspect_ratio"] - vec2["aspect_ratio"]) / max(vec1["aspect_ratio"], vec2["aspect_ratio"])
        if ar_diff > self.MAX_ASPECT_RATIO_DIFF:
            return False

        torso_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1["torso_rgb"], vec2["torso_rgb"])))
        legs_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1["legs_rgb"], vec2["legs_rgb"])))

        if torso_dist < self.MAX_COLOR_DISTANCE and legs_dist < self.MAX_COLOR_DISTANCE:
            return True
        return False

    def _check_incident_rules(self, trail: Trail):
        unique_cameras = set([e.camera_id for e in trail.events])
        if len(unique_cameras) >= 2 and trail.status != "incident":
            trail.status = "incident"
            
            start_time = trail.events[0].timestamp
            end_time = trail.events[-1].timestamp
            total_time = int(end_time - start_time)
            
            path_str = trail.get_path_summary()
            
            print(f"\n🚨🚨 [CRITICAL ALERT] Suspicious traversal detected! 🚨🚨")
            print(f"Time Taken: {total_time} seconds")
            print(f"Path: {path_str}\n")
            
            self._send_telegram_alert(trail.trail_id, path_str, total_time)

    def _send_telegram_alert(self, trail_id, path_str, total_time):
        """Sends the compiled story directly to your phone."""
        if not self.bot_token or not self.chat_id:
            print(">> No Telegram Credentials found. Skipping phone push.")
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            message = (
                f"🚨 **Smart CCTV: Critical Incident** 🚨\n\n"
                f"**Trail ID:** `{trail_id}`\n"
                f"**Path Taken:** {path_str}\n"
                f"**Duration:** {total_time} seconds\n\n"
                f"⚠️ *Subject traversed 3 secured zones within anomaly threshold.*"
            )
            data = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
            requests.post(url, data=data)
            print("✅ Incident Report sent to Telegram successfully!")
        except Exception as e:
            print(f"❌ Telegram push failed: {e}")

    def _cleanup_stale_trails(self):
        current_time = time.time()
        for trail in self.active_trails:
            if trail.status == "active" and (current_time - trail.last_updated) > 300:
                trail.status = "closed"