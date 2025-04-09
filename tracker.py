import numpy as np
from scipy.optimize import linear_sum_assignment

class IOUTracker:
    def __init__(self, iou_thresh=0.01, max_lost=5):
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.next_id = 1
        self.enter_count = 0
        self.exit_count = 0
        self.max_lost = max_lost  # Max frames before deleting a track

    def iou(self, box1, box2):
        """Compute IoU between two bounding boxes (xyxy format)."""
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - inter
        return inter / union if union > 0 else 0

    def match_tracks(self, detections):
        """Perform optimal assignment using Hungarian matching."""
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self.iou(track['bbox'], det)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU (negate for minimization)

        matches, unmatched_detections = [], list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > self.iou_thresh:
                matches.append((r, c))
                unmatched_detections.remove(c)

        return matches, unmatched_detections

    def update(self, detections):
        """Update tracked objects with Hungarian matching and remove lost tracks."""
        matches, unmatched_detections = self.match_tracks(detections)

        # Update existing tracks
        updated_tracks = []
        for t_idx, track in enumerate(self.tracks):
            matched = False
            for m_track, m_det in matches:
                if m_track == t_idx:
                    track['bbox'] = detections[m_det]
                    track['lost'] = 0
                    updated_tracks.append(track)
                    matched = True
                    break
            if not matched:
                track['lost'] += 1  # Increase lost count

        # Remove lost tracks
        self.tracks = [t for t in updated_tracks if t['lost'] <= self.max_lost]

        # Add new detections as new tracks
        for i in unmatched_detections:
            self.tracks.append({'id': self.next_id, 'bbox': detections[i], 'counted': False, 'lost': 0})
            self.next_id += 1

    def point_position(self, x, y, line_start, line_end):
        """Determine the position of a point relative to a line."""
        x1, y1 = line_start
        x2, y2 = line_end
        position = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        return position

    def count_crossing(self, line_start, line_end):
        """Count objects crossing the virtual line."""
        for track in self.tracks:
            x1, y1, x2, y2 = track['bbox']
            lower_left_x, lower_left_y = x1, y2  # Lower-left corner

            current_position = self.point_position(lower_left_x, lower_left_y, line_start, line_end)

            if "previous_position" not in track:
                track["previous_position"] = current_position

            if not track["counted"]:
                if track["previous_position"] < 0 and current_position > 0:
                    self.enter_count += 1
                    track["counted"] = True
                elif track["previous_position"] > 0 and current_position < 0:
                    self.exit_count += 1
                    track["counted"] = True

            track["previous_position"] = current_position
