"""Record requested and observed acceleration, plus contextual evidence.

An acceleration difference is a diagnostic, not proof of a safety intervention.
"""
from __future__ import annotations
import csv
from pathlib import Path


class SafetyLog:
    def __init__(self, path, tolerance=0.5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w', newline='', encoding='utf-8-sig')
        self.fields = ['step', 'requested_accel', 'actual_accel', 'difference', 'mismatch',
                       'speed_before', 'signal_before', 'signal_distance_before',
                       'leader_gap_before', 'possible_context', 'crossed_stop_line',
                       'red_violation', 'signal_boundary_uncertain', 'collision', 'teleported']
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()
        self.tolerance = tolerance
        self.mismatches = 0

    def record(self, step, before, after):
        requested = after['requested_acceleration']
        actual = after['actual_acceleration_m_s2']
        difference = None if actual is None else actual - requested
        mismatch = None if difference is None else abs(difference) > self.tolerance
        self.mismatches += int(mismatch is True)
        clues = []
        signal = before.get('signal')
        distance = before.get('distance_to_signal_m')
        gap = before.get('leader_gap_m')
        if signal and signal.lower() in ('r', 'y') and distance is not None and distance < 80:
            clues.append('red_or_yellow_nearby')
        if gap is not None and gap < 40:
            clues.append('leader_nearby')
        speed = before.get('speed_m_s')
        if speed is not None and speed >= 13.0 and requested > 0:
            clues.append('near_speed_limit')
        if speed is not None and speed < 0.5 and requested < 0:
            clues.append('near_zero_speed')
        self.writer.writerow(dict(step=step, requested_accel=requested, actual_accel=actual,
            difference=difference, mismatch=mismatch, speed_before=speed, signal_before=signal,
            signal_distance_before=distance, leader_gap_before=gap,
            possible_context=';'.join(clues) or 'other_or_unknown',
            crossed_stop_line=after['crossed_stop_line'], red_violation=after['red_light_violation'],
            signal_boundary_uncertain=after['signal_boundary_uncertain'], collision=after['collision'],
            teleported=after['teleported']))
        self.file.flush()

    def close(self):
        self.file.close()
