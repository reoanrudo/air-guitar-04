"""
Air Guitar Pro - Pygame + WebRTC P2P
Main game with MediaPipe hand detection and real-time communication.
"""

import asyncio
import json
import logging
import random
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pygame
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import object_to_string, string_to_object

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Game constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Rhythm game constants
NOTE_SPEED = 8
HIT_ZONE_X = WINDOW_WIDTH - 200
HIT_WINDOW = 80
STRUM_VELOCITY_THRESHOLD = 15
SPAWN_INTERVAL = 1100  # ms

# Colors
COLOR_BG = (2, 6, 23)
COLOR_TRACK = (15, 23, 42)
COLOR_STRING_INACTIVE = (50, 50, 50)
COLOR_STRING_ACTIVE = (251, 146, 60)
COLOR_NOTE = (249, 115, 22)
COLOR_NOTE_MISS = (69, 10, 10)
COLOR_PERFECT = (250, 204, 21)
COLOR_GREAT = (56, 189, 248)
COLOR_MISS = (239, 68, 68)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_SLATE = (100, 116, 139)

# Guitar strings (E2, A2, D3, G3, B3, E4)
STRING_FREQUENCIES = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'E']


class AudioEngine:
    """Simple audio engine using pygame.mixer for guitar sounds."""

    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sounds = {}
        self.generate_guitar_sounds()

    def generate_guitar_sounds(self):
        """Generate simple guitar-like sounds using numpy."""
        sample_rate = 44100

        for i, freq in enumerate(STRING_FREQUENCIES):
            duration = 1.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)

            # Guitar-like tone: mix of fundamental and harmonics
            wave = (0.6 * np.sin(2 * np.pi * freq * t) +
                   0.3 * np.sin(2 * np.pi * freq * 2 * t) +
                   0.1 * np.sin(2 * np.pi * freq * 3 * t))

            # Envelope
            envelope = np.exp(-3 * t)
            wave = wave * envelope * 0.5

            # Convert to 16-bit PCM
            wave = (wave * 32767).astype(np.int16)

            # Stereo
            stereo = np.column_stack((wave, wave))

            sound = pygame.sndarray.make_sound(stereo)
            sound.set_volume(0.3)
            self.sounds[i] = sound

    def play_strum(self, fret_states, direction):
        """Play strum sound based on fret states."""
        now = pygame.time.get_ticks()
        indices = range(6) if direction == 'down' else range(5, -1, -1)

        for i, string_idx in enumerate(indices):
            if 0 <= string_idx < len(STRING_FREQUENCIES):
                fret = fret_states[string_idx] if string_idx < len(fret_states) else 0
                freq = STRING_FREQUENCIES[string_idx] * (2 ** (fret / 12))

                # Find closest pre-generated sound
                sound_idx = string_idx
                if sound_idx in self.sounds:
                    delay = i * 15  # 15ms between strings
                    pygame.time.set_timer(pygame.event.Event(pygame.USEREVENT, {
                        'sound': sound_idx, 'time': now + delay
                    }), delay)

    def play_note(self, string_idx, fret):
        """Play a single note."""
        if string_idx in self.sounds:
            self.sounds[string_idx].play()

    def play_miss(self):
        """Play miss sound."""
        # Simple click sound
        duration = 0.1
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * 200 * t) * np.exp(-30 * t)
        wave = (wave * 16384).astype(np.int16)
        stereo = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo)
        sound.set_volume(0.2)
        sound.play()


class HandDetector:
    """MediaPipe hand detection for strumming."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        """Detect hands and return landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results.multi_hand_landmarks if results else None

    def get_strum_direction(self, landmarks, frame_height):
        """Detect strum direction from hand movement."""
        if not landmarks:
            return None

        # Get finger tip positions (index, middle, ring)
        finger_tips = [8, 12, 16]
        y_positions = []

        for tip_idx in finger_tips:
            y = landmarks.landmark[tip_idx].y
            y_positions.append(y * frame_height)

        avg_y = sum(y_positions) / len(y_positions)

        # Store previous position for velocity calculation
        if not hasattr(self, 'prev_y'):
            self.prev_y = avg_y
            self.prev_time = datetime.now()
            return None

        # Calculate velocity
        curr_time = datetime.now()
        dt = (curr_time - self.prev_time).total_seconds()

        if dt > 0:
            velocity = (avg_y - self.prev_y) / dt
            self.prev_y = avg_y
            self.prev_time = curr_time

            if abs(velocity) > STRUM_VELOCITY_THRESHOLD * 60:  # Scale to match detection rate
                return 'down' if velocity > 0 else 'up'

        return None


class Note:
    """Rhythm game note."""

    def __init__(self, fret):
        self.x = -100
        self.fret = fret
        self.hit = False
        self.missed = False


class Particle:
    """Visual effect particle."""

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-20, 20)
        self.vy = random.uniform(-20, 20)
        self.life = 1.0
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.03
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(self.life * 255)
            color = (*self.color[:3], alpha)
            # Create a surface with per-pixel alpha
            s = pygame.Surface((int(16 * self.life), int(16 * self.life)), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(8 * self.life), int(8 * self.life)), int(8 * self.life))
            surface.blit(s, (int(self.x), int(self.y)))


class Game:
    """Main game class."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Air Guitar Pro - Python")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self.running = True
        self.game_started = False
        self.camera_ready = False

        # Audio
        self.audio = AudioEngine()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Hand detection
        self.hand_detector = HandDetector()

        # Game state
        self.fret_states = [0, 0, 0, 0, 0, 0]
        self.score = 0
        self.combo = 0
        self.notes = []
        self.particles = []
        self.last_note_time = 0
        self.last_strum_time = 0

        # Hit rating display
        self.hit_rating_text = None
        self.hit_rating_color = COLOR_TEXT_WHITE
        self.hit_rating_time = 0

        # WebRTC connection
        self.data_channel = None

    def set_data_channel(self, channel):
        """Set WebRTC data channel for receiving fret data."""
        self.data_channel = channel

    def handle_fret_update(self, data):
        """Handle fret state update from mobile controller."""
        if isinstance(data, list) and len(data) == 6:
            self.fret_states = data

    def spawn_note(self):
        """Spawn a new note."""
        now = pygame.time.get_ticks()
        if now - self.last_note_time > SPAWN_INTERVAL:
            fret = random.choice([0, 3, 5, 7, 10, 12])
            self.notes.append(Note(fret))
            self.last_note_time = now

    def update_notes(self, strum_direction):
        """Update notes and check for hits."""
        now = pygame.time.get_ticks()
        center_y = 200

        for note in self.notes[:]:
            if note.hit or note.missed:
                continue

            note.x += NOTE_SPEED

            # Check for hit
            if strum_direction and abs(note.x - HIT_ZONE_X) < HIT_WINDOW:
                note.hit = True
                is_perfect = abs(note.x - HIT_ZONE_X) < HIT_WINDOW / 3

                self.score += 1000 if is_perfect else 500
                self.combo += 1

                self.hit_rating_text = "PERFECT" if is_perfect else "GREAT"
                self.hit_rating_color = COLOR_PERFECT if is_perfect else COLOR_GREAT
                self.hit_rating_time = now

                self.audio.play_strum(self.fret_states, strum_direction)

                # Spawn particles
                color = COLOR_PERFECT if is_perfect else COLOR_GREAT
                for _ in range(20):
                    self.particles.append(Particle(HIT_ZONE_X, center_y, color))

                self.last_strum_time = now

            # Check for miss
            elif note.x > HIT_ZONE_X + HIT_WINDOW:
                note.missed = True
                self.combo = 0

                self.hit_rating_text = "MISS"
                self.hit_rating_color = COLOR_MISS
                self.hit_rating_time = now

                self.audio.play_miss()

        # Remove off-screen or hit notes
        self.notes = [n for n in self.notes if n.x < WINDOW_WIDTH + 100 and not n.hit]

    def update_particles(self):
        """Update and remove dead particles."""
        self.particles = [p for p in self.particles if p.update()]

    def draw_video_feed(self, frame):
        """Draw camera feed as background."""
        if frame is not None:
            # Flip horizontally
            frame = cv2.flip(frame, 1)
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Rotate for pygame
            frame = np.rot90(frame)
            # Create pygame surface
            surface = pygame.surfarray.make_surface(frame)
            # Scale to window
            surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
            # Draw with transparency
            surface.set_alpha(60)
            self.screen.blit(surface, (0, 0))

    def draw_rhythm_track(self):
        """Draw the rhythm game track."""
        track_y = 80
        track_height = 240

        # Track background
        pygame.draw.rect(self.screen, COLOR_TRACK, (0, track_y, WINDOW_WIDTH, track_height))

        # Hit zone line
        line_color = COLOR_PERFECT if self.combo > 5 else COLOR_GREAT
        pygame.draw.line(self.screen, line_color, (HIT_ZONE_X, track_y + 20),
                        (HIT_ZONE_X, track_y + track_height - 20), 12)

        # Draw notes
        center_y = track_y + track_height // 2
        for note in self.notes:
            if not note.hit:
                color = COLOR_NOTE if not note.missed else COLOR_NOTE_MISS
                rect = pygame.Rect(note.x - 50, center_y - 40, 100, 80)
                pygame.draw.rect(self.screen, color, rect, border_radius=15)

                # Fret number
                text = self.font_medium.render(f"F{note.fret}", True, COLOR_TEXT_WHITE)
                text_rect = text.get_rect(center=(note.x, center_y))
                self.screen.blit(text, text_rect)

    def draw_strings(self):
        """Draw guitar strings."""
        strum_zone_y = WINDOW_HEIGHT * 0.55
        for i in range(6):
            y_offset = i * 25
            active = self.fret_states[i] > 0

            color = COLOR_STRING_ACTIVE if active else COLOR_STRING_INACTIVE
            width = 6 if active else 2

            start_pos = (0, strum_zone_y + y_offset)
            end_pos = (WINDOW_WIDTH, strum_zone_y + y_offset + 120)

            pygame.draw.line(self.screen, color, start_pos, end_pos, width)

    def draw_ui(self):
        """Draw game UI."""
        # Score
        score_text = self.font_large.render(f"{self.score:,}", True, COLOR_TEXT_WHITE)
        self.screen.blit(score_text, (50, 40))

        # Combo
        if self.combo > 0:
            combo_text = self.font_large.render(str(self.combo), True, COLOR_STRING_ACTIVE)
            combo_rect = combo_text.get_rect(right=WINDOW_WIDTH - 50, bottom=WINDOW_HEIGHT - 100)
            self.screen.blit(combo_text, combo_rect)

            combo_label = self.font_medium.render("COMBO!", True, COLOR_TEXT_WHITE)
            combo_label_rect = combo_label.get_rect(right=WINDOW_WIDTH - 50, top=combo_rect.bottom + 10)
            self.screen.blit(combo_label, combo_label_rect)

        # Hit rating
        now = pygame.time.get_ticks()
        if self.hit_rating_text and now - self.hit_rating_time < 500:
            rating = self.font_large.render(self.hit_rating_text, True, self.hit_rating_color)
            rating_rect = rating.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(rating, rating_rect)

        # Fret states display
        fret_y = WINDOW_HEIGHT - 80
        for i, fret in enumerate(self.fret_states):
            color = COLOR_STRING_ACTIVE if fret > 0 else COLOR_TEXT_SLATE
            fret_text = self.font_small.render(f"{STRING_NAMES[i]}: {fret}", True, color)
            self.screen.blit(fret_text, (50 + i * 100, fret_y))

    def draw_particles(self):
        """Draw all particles."""
        for particle in self.particles:
            particle.draw(self.screen)

    def draw_loading(self):
        """Draw loading screen."""
        self.screen.fill(COLOR_BG)

        text = self.font_medium.render("INITIALIZING CAMERA...", True, COLOR_TEXT_WHITE)
        rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(text, rect)

        pygame.display.flip()

    def draw_start_screen(self):
        """Draw start screen."""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((2, 6, 23, 250))
        self.screen.blit(overlay, (0, 0))

        # Title
        title = self.font_large.render("AIR GUITAR PRO", True, COLOR_STRING_ACTIVE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)

        # Instructions
        instr = self.font_small.render("Move hand in strum zone (right side) to play", True, COLOR_TEXT_SLATE)
        instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(instr, instr_rect)

        # Start prompt
        prompt = self.font_medium.render("Press SPACE to Start", True, COLOR_TEXT_WHITE)
        prompt_rect = prompt.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 80))
        self.screen.blit(prompt, prompt_rect)

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE and not self.game_started and self.camera_ready:
                        self.game_started = True

            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                self.camera_ready = False
                self.draw_loading()
                continue
            else:
                self.camera_ready = True

            # Clear screen
            self.screen.fill(COLOR_BG)

            # Draw video feed
            self.draw_video_feed(frame)

            if not self.game_started:
                self.draw_start_screen()
                self.clock.tick(FPS)
                continue

            # Detect hands
            landmarks = self.hand_detector.detect(frame)
            strum_direction = None

            if landmarks:
                # Get strum direction
                strum_direction = self.hand_detector.get_strum_direction(landmarks[0], WINDOW_HEIGHT)
                now = pygame.time.get_ticks()

                if strum_direction and now - self.last_strum_time > 150:
                    # Spawn particles at hand position
                    wrist = landmarks[0].landmark[0]
                    hand_x = int((1 - wrist.x) * WINDOW_WIDTH)
                    hand_y = int(wrist.y * WINDOW_HEIGHT)

                    for _ in range(10):
                        self.particles.append(Particle(hand_x, hand_y, COLOR_STRING_ACTIVE))

            # Game logic
            self.spawn_note()
            self.update_notes(strum_direction)
            self.update_particles()

            # Draw everything
            self.draw_rhythm_track()
            self.draw_strings()
            self.draw_ui()
            self.draw_particles()

            pygame.display.flip()
            self.clock.tick(FPS)

        # Cleanup
        self.cap.release()
        pygame.quit()


def main():
    """Entry point."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
