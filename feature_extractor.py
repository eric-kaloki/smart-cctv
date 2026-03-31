import cv2
import numpy as np

class AppearanceExtractor:
    def __init__(self, mode="heuristic"):
        """
        mode: "heuristic" (Fast/MVP) or "deep_reid" (Future AI Upgrade)
        """
        self.mode = mode
        
        if self.mode == "deep_reid":
            # FUTURE: Load the AI model here so it stays in memory
            # self.reid_model = load_osnet_model("osnet_x0_25.onnx")
            print("Deep Re-ID mode initialized (Placeholder)")

    def get_vector(self, frame, box):
        """Routes the image to the correct extraction method."""
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        
        # Failsafe for bad boxes at the edge of the screen
        if crop.size == 0:
            return None 

        if self.mode == "heuristic":
            return self._extract_heuristic(crop)
        elif self.mode == "deep_reid":
            return self._extract_deep_reid(crop)

    def _extract_heuristic(self, crop):
        """
        THE MVP STRICT MODE: Zero-AI, mathematically cheap heuristics.
        Returns a dictionary vector of colors and aspect ratio.
        """
        h, w, _ = crop.shape
        aspect_ratio = round(h / (w + 0.0001), 2) # +0.0001 prevents division by zero

        # Split person into Torso (top 50%) and Legs (bottom 50%)
        half_h = h // 2
        torso = crop[0:half_h, :]
        legs = crop[half_h:h, :]

        # Get dominant color by averaging the pixels
        # cv2 uses BGR, we convert to RGB for easier human debugging
        torso_color = cv2.mean(torso)[:3][::-1] 
        legs_color = cv2.mean(legs)[:3][::-1]

        return {
            "type": "heuristic",
            "aspect_ratio": aspect_ratio,
            "torso_rgb": [int(c) for c in torso_color],
            "legs_rgb": [int(c) for c in legs_color]
        }

    def _extract_deep_reid(self, crop):
        """
        THE FUTURE UPGRADE: Deep Learning Person Re-Identification.
        DOCUMENTATION FOR SECONDARY IMPLEMENTATION:
        1. Resize the crop to exactly 256x128 pixels (standard Re-ID size).
        2. Pass the crop through a lightweight model like OSNet or MobileNet.
        3. The model outputs a 512-dimensional array of float numbers (an embedding).
        4. Return that embedding.
        """
        # FUTURE CODE:
        # resized = cv2.resize(crop, (128, 256))
        # embedding = self.reid_model.predict(resized)
        # return {"type": "embedding", "vector": embedding.tolist()}
        pass