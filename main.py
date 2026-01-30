"""
Air Guitar Pro - Main Entry Point
Integrated Pygame game + WebRTC server
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pygame
from aiohttp import web as aio_web
from aiortc import RTCPeerConnection, RTCSessionDescription

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

# Guitar strings
STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'E']


# Song data structure
class Song:
    """Song data with BPM and notes."""

    def __init__(self, name, bpm, notes, duration=None):
        self.name = name
        self.bpm = bpm
        self.notes = notes  # List of (time_ms, fret) tuples
        self.duration = duration or (notes[-1][0] + 3000 if notes else 60000)
        self.note_index = 0
        self.start_time = None

    def get_notes_at_time(self, current_time_ms):
        """Get notes that should spawn at current time."""
        notes_to_spawn = []
        while (self.note_index < len(self.notes) and
               self.notes[self.note_index][0] <= current_time_ms):
            notes_to_spawn.append(self.notes[self.note_index][1])
            self.note_index += 1
        return notes_to_spawn

    def reset(self):
        """Reset song for replay."""
        self.note_index = 0
        self.start_time = None


# Simple beginner songs
BEGINNER_SONGS = {
    'twinkle': Song(
        "Twinkle Twinkle Little Star",
        bpm=100,
        notes=[
            (0, 0), (1000, 0), (2000, 3), (3000, 3), (4000, 5), (5000, 5), (6000, 3),
            (7000, 0), (8000, 0), (9000, 3), (10000, 3), (11000, 5), (12000, 5), (13000, 7),
            (14000, 5), (15000, 5), (16000, 3), (17000, 3), (18000, 0), (19000, 0), (20000, 0),
            # Repeat
            (21000, 0), (22000, 0), (23000, 3), (24000, 3), (25000, 5), (26000, 5), (27000, 3),
        ],
        duration=30000  # 30 seconds
    ),
    'happy_birthday': Song(
        "Happy Birthday",
        bpm=95,
        notes=[
            (0, 0), (500, 0), (1000, 2), (1500, 0), (2000, 5), (2500, 4),
            (3000, 0), (3500, 0), (4000, 2), (4500, 0), (5000, 7), (5500, 5),
            (6000, 0), (6500, 0), (7000, 9), (7500, 5), (8000, 4), (8500, 2), (9000, 3),
            (9500, 1), (10000, 1), (10500, 0), (11000, 0), (11500, 7), (12000, 5),
        ],
        duration=25000
    ),
}


class GameState:
    """Shared game state accessible by WebRTC and game loop."""

    def __init__(self):
        self.fret_states = [0, 0, 0, 0, 0, 0]
        self.score = 0
        self.combo = 0
        self.running = True


game_state = GameState()


class WebRTCServer:
    """WebRTC signaling server."""

    def __init__(self, game_state_ref):
        self.game_state = game_state_ref
        self.pcs = set()
        self.app = None
        self.runner = None

    def create_pc(self):
        """Create a new RTCPeerConnection."""
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel created: {channel.label}")

            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    if data.get("type") == "FRET_UPDATE":
                        self.game_state.fret_states = data.get("payload", [0, 0, 0, 0, 0, 0])
                        logger.info(f"FRET update: {self.game_state.fret_states}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message}")

        return pc

    async def offer_handler(self, request):
        """Handle WebRTC offer from mobile client."""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = self.create_pc()
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return aio_web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })

    async def mobile_handler(self, request):
        """Serve mobile controller page."""
        html_path = Path(__file__).parent / "templates" / "mobile_webrtc.html"
        if html_path.exists():
            return aio_web.Response(text=html_path.read_text(), content_type="text/html")
        return aio_web.Response(text="Not found", status=404)

    async def index_handler(self, request):
        """Index page."""
        return aio_web.Response(text="Air Guitar Pro - Running. Use /mobile for controller.")

    async def start(self):
        """Start the WebRTC server."""
        self.app = aio_web.Application()
        self.app.router.add_get("/", self.index_handler)
        self.app.router.add_get("/mobile", self.mobile_handler)
        self.app.router.add_post("/offer", self.offer_handler)

        self.runner = aio_web.AppRunner(self.app)
        await self.runner.setup()
        site = aio_web.TCPSite(self.runner, "0.0.0.0", 8081)
        await site.start()

        logger.info("WebRTC server running on http://0.0.0.0:8081")

    async def stop(self):
        """Stop the server."""
        if self.runner:
            await self.runner.cleanup()
        for pc in self.pcs:
            await pc.close()
        self.pcs.clear()


class MotionDetector:
    """Motion detection for strumming using frame difference."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.prev_frame = None
        self.last_strum_time = 0
        self.motion_level = 0
        self.motion_upper = 0
        self.motion_lower = 0
        self.prev_center_y = None  # Previous center of motion

        # Strum state tracking to prevent continuous triggering
        self.waiting_for_opposite_half = False
        self.last_strum_half = None  # 'upper' or 'lower'

        # Strum zone (right side of screen)
        self.strum_zone = {
            'x': int(width * 0.6),
            'y': int(height * 0.5),
            'w': int(width * 0.35),
            'h': int(height * 0.4)
        }
        self.strum_mid_y = self.strum_zone['y'] + self.strum_zone['h'] // 2
        self.zone_mid_y = self.strum_zone['h'] // 2  # Relative middle of zone

    def detect_strum(self, frame):
        """Detect strum direction using motion in strum zone."""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.strum_zone['x'], self.strum_zone['y'], self.strum_zone['w'], self.strum_zone['h']

        # Extract strum zone
        current_zone = gray[y:y+h, x:x+w]
        prev_zone = self.prev_frame[y:y+h, x:x+w]

        # Calculate difference
        diff = cv2.absdiff(current_zone, prev_zone)

        # Find center of motion (weighted average Y position)
        diff_sum = np.sum(diff)
        if diff_sum > 0:
            # Create Y coordinates array
            y_coords = np.arange(h).reshape(-1, 1)
            # Calculate weighted Y position
            center_y = int(np.sum(diff * y_coords) / diff_sum)
        else:
            center_y = self.zone_mid_y

        # For visualization
        self.motion_upper = np.mean(diff[:h//2, :])
        self.motion_lower = np.mean(diff[h//2:, :])
        self.motion_level = max(self.motion_upper, self.motion_lower)

        self.prev_frame = gray

        # Detect strum by tracking motion center crossing midline
        now = datetime.now().timestamp() * 1000
        if now - self.last_strum_time > 150:  # Cooldown between strums
            # Check if there's enough motion
            if diff_sum > 3000:
                if self.prev_center_y is not None:
                    velocity = center_y - self.prev_center_y

                    # Check which half we're in
                    current_half = 'upper' if center_y < self.zone_mid_y else 'lower'

                    # Check if crossed midline
                    mid_crossed = (self.prev_center_y < self.zone_mid_y and center_y >= self.zone_mid_y) or \
                                  (self.prev_center_y > self.zone_mid_y and center_y <= self.zone_mid_y)

                    if mid_crossed:
                        # Check if we were waiting for opposite half (prevents double-triggering)
                        if not self.waiting_for_opposite_half:
                            # First crossing - valid strum
                            self.last_strum_time = now
                            self.last_strum_half = current_half
                            self.waiting_for_opposite_half = True
                            return 'down' if velocity > 0 else 'up'
                        else:
                            # We were waiting for opposite half - check if we're now in opposite half
                            expected_half = 'lower' if self.last_strum_half == 'upper' else 'upper'
                            if current_half == expected_half:
                                # Successfully moved to opposite half - ready for next strum
                                self.waiting_for_opposite_half = False

            self.prev_center_y = center_y
        else:
            # Still in cooldown, reset waiting state
            if now - self.last_strum_time > 80:
                self.waiting_for_opposite_half = False
                self.last_strum_half = None

        return None

    def get_strum_zone_rect(self):
        """Get strum zone rectangle for drawing."""
        return (self.strum_zone['x'], self.strum_zone['y'],
                self.strum_zone['w'], self.strum_zone['h'])

    def get_motion_levels(self):
        """Get current motion levels for visualization."""
        return self.motion_upper, self.motion_lower


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
            s = pygame.Surface((int(16 * self.life), int(16 * self.life)), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color[:3], alpha), (int(8 * self.life), int(8 * self.life)), int(8 * self.life))
            surface.blit(s, (int(self.x), int(self.y)))


class Note:
    """Rhythm game note."""

    def __init__(self, fret):
        self.x = -100
        self.fret = fret
        self.hit = False
        self.missed = False


class AudioEngine:
    """Simple audio engine."""

    def __init__(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.enabled = True
        except:
            logger.warning("Audio init failed, continuing without sound")
            self.enabled = False
        # Base frequencies for open strings (E2, A2, D3, G3, B3, E4)
        self.base_freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
        self.sample_rate = 44100
        self.bgm = None

    def load_bgm(self, song_name):
        """Load BGM MP3 file."""
        if not self.enabled:
            return

        bgm_path = Path(__file__).parent / f"{song_name}.mp3"
        if bgm_path.exists():
            try:
                self.bgm = pygame.mixer.Sound(str(bgm_path))
                self.bgm.set_volume(0.5)
                logger.info(f"BGM loaded: {bgm_path}")
            except Exception as e:
                logger.warning(f"Failed to load BGM: {e}")
                self.bgm = None
        else:
            logger.warning(f"BGM file not found: {bgm_path}")
            self.bgm = None

    def play_bgm(self):
        """Play BGM."""
        if self.bgm and self.enabled:
            self.bgm.play(loops=-1)  # Loop indefinitely

    def stop_bgm(self):
        """Stop BGM."""
        if self.bgm and self.enabled:
            self.bgm.stop()

    def generate_note_sound(self, freq):
        """Generate a guitar-like note sound for given frequency."""
        duration = 0.6
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Guitar-like tone with harmonics
        wave = (0.5 * np.sin(2 * np.pi * freq * t) +
               0.25 * np.sin(2 * np.pi * freq * 2 * t) +
               0.125 * np.sin(2 * np.pi * freq * 3 * t) +
               0.06 * np.sin(2 * np.pi * freq * 4 * t))

        # ADSR envelope
        attack = int(0.01 * self.sample_rate)
        decay = int(0.1 * self.sample_rate)
        sustain = int(0.3 * self.sample_rate)
        release = int(0.19 * self.sample_rate)

        envelope = np.ones_like(t)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        envelope[attack+decay:attack+decay+sustain] = 0.7
        envelope[attack+decay+sustain:] = np.linspace(0.7, 0, release)

        wave = wave * envelope * 0.4
        wave = (wave * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))

        return pygame.sndarray.make_sound(stereo)

    def play_strum(self, fret_states, direction='down'):
        """Play strum with fret positions."""
        if not self.enabled:
            return

        # Determine string order based on strum direction
        if direction == 'down':
            string_order = [0, 1, 2, 3, 4, 5]  # Low to high E
        else:
            string_order = [5, 4, 3, 2, 1, 0]  # High to low E

        for i, string_idx in enumerate(string_order):
            if string_idx < len(fret_states):
                fret = fret_states[string_idx]

                # Calculate frequency: base_freq * 2^(fret/12)
                freq = self.base_freqs[string_idx] * (2 ** (fret / 12))

                # Generate and play sound
                sound = self.generate_note_sound(freq)
                sound.set_volume(0.3)

                # Play all at once (simpler, no delay)
                sound.play()

    def play_miss(self):
        """Play miss sound."""
        pass


class Game:
    """Main game class."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Air Guitar Pro - Python")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self.game_started = False
        self.camera_ready = False

        # Song system
        self.current_song = None
        self.song_start_time = None
        self.available_songs = BEGINNER_SONGS
        self.selected_song_name = 'twinkle'  # Default song

        self.audio = AudioEngine()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.motion_detector = MotionDetector(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.notes = []
        self.particles = []
        self.last_note_time = 0
        self.hit_rating_text = None
        self.hit_rating_color = COLOR_TEXT_WHITE
        self.hit_rating_time = 0

    def start_song(self):
        """Start the selected song."""
        self.current_song = self.available_songs[self.selected_song_name]
        self.current_song.reset()
        self.song_start_time = pygame.time.get_ticks()
        # Load and play BGM
        self.audio.load_bgm(self.selected_song_name)
        self.audio.play_bgm()

    def spawn_note(self):
        """Spawn notes from song data."""
        if not self.current_song or not self.song_start_time:
            # Random mode when no song
            now = pygame.time.get_ticks()
            if now - self.last_note_time > SPAWN_INTERVAL:
                self.notes.append(Note(random.choice([0, 3, 5, 7, 10, 12])))
                self.last_note_time = now
            return

        # Song mode
        current_time = pygame.time.get_ticks()
        song_time = current_time - self.song_start_time

        # Get notes that should spawn now
        frets = self.current_song.get_notes_at_time(song_time)
        for fret in frets:
            self.notes.append(Note(fret))

        # Check if song ended
        if song_time > self.current_song.duration:
            # Loop the song
            self.start_song()

    def update_notes(self, strum_direction):
        """Update notes."""
        now = pygame.time.get_ticks()
        center_y = 200

        for note in self.notes[:]:
            if note.hit or note.missed:
                continue

            note.x += NOTE_SPEED

            if strum_direction and abs(note.x - HIT_ZONE_X) < HIT_WINDOW:
                note.hit = True
                is_perfect = abs(note.x - HIT_ZONE_X) < HIT_WINDOW / 3

                game_state.score += 1000 if is_perfect else 500
                game_state.combo += 1

                self.hit_rating_text = "PERFECT" if is_perfect else "GREAT"
                self.hit_rating_color = COLOR_PERFECT if is_perfect else COLOR_GREAT
                self.hit_rating_time = now

                self.audio.play_strum(game_state.fret_states, strum_direction)

                color = COLOR_PERFECT if is_perfect else COLOR_GREAT
                for _ in range(15):
                    self.particles.append(Particle(HIT_ZONE_X, center_y, color))

            elif note.x > HIT_ZONE_X + HIT_WINDOW:
                note.missed = True
                game_state.combo = 0
                self.hit_rating_text = "MISS"
                self.hit_rating_color = COLOR_MISS
                self.hit_rating_time = now
                self.audio.play_miss()

        self.notes = [n for n in self.notes if n.x < WINDOW_WIDTH + 100 and not n.hit]

    def update_particles(self):
        """Update particles."""
        self.particles = [p for p in self.particles if p.update()]

    def draw_video_feed(self, frame):
        """Draw video background."""
        if frame is not None:
            # No flip - show camera as-is (not mirrored)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
            surface.set_alpha(60)
            self.screen.blit(surface, (0, 0))

    def draw_rhythm_track(self):
        """Draw track."""
        track_y, track_height = 80, 240
        pygame.draw.rect(self.screen, COLOR_TRACK, (0, track_y, WINDOW_WIDTH, track_height))

        line_color = COLOR_PERFECT if game_state.combo > 5 else COLOR_GREAT
        pygame.draw.line(self.screen, line_color, (HIT_ZONE_X, track_y + 20),
                        (HIT_ZONE_X, track_y + track_height - 20), 12)

        center_y = track_y + track_height // 2
        for note in self.notes:
            if not note.hit:
                color = COLOR_NOTE if not note.missed else COLOR_NOTE_MISS
                rect = pygame.Rect(note.x - 50, center_y - 40, 100, 80)
                pygame.draw.rect(self.screen, color, rect, border_radius=15)
                text = self.font_medium.render(f"F{note.fret}", True, COLOR_TEXT_WHITE)
                self.screen.blit(text, text.get_rect(center=(note.x, center_y)))

    def draw_strings(self):
        """Draw guitar strings."""
        strum_zone_y = WINDOW_HEIGHT * 0.55

        # Draw strum zone indicator
        zone_x, zone_y, zone_w, zone_h = self.motion_detector.get_strum_zone_rect()
        zone_rect = pygame.Rect(zone_x, zone_y, zone_w, zone_h)
        pygame.draw.rect(self.screen, (30, 41, 59), zone_rect, border_radius=10)
        pygame.draw.rect(self.screen, (71, 85, 105), zone_rect, 2, border_radius=10)

        # Draw motion visualization
        motion_upper, motion_lower = self.motion_detector.get_motion_levels()
        max_motion = 50  # For normalization

        # Upper half motion indicator
        if motion_upper > 0:
            upper_intensity = min(255, int(motion_upper / max_motion * 255))
            upper_rect = pygame.Rect(zone_x + 5, zone_y + 5, zone_w - 10, zone_h // 2 - 10)
            upper_surface = pygame.Surface((upper_rect.width, upper_rect.height), pygame.SRCALPHA)
            upper_surface.fill((251, 146, 60, upper_intensity // 3))  # Orange with transparency
            self.screen.blit(upper_surface, upper_rect)

            # Motion indicator bar
            bar_width = min(zone_w - 20, int(motion_upper / max_motion * (zone_w - 20)))
            if bar_width > 0:
                pygame.draw.rect(self.screen, (251, 146, 60),
                               (zone_x + 10, zone_y + zone_h // 4 - 5, bar_width, 10))

        # Lower half motion indicator
        if motion_lower > 0:
            lower_intensity = min(255, int(motion_lower / max_motion * 255))
            lower_rect = pygame.Rect(zone_x + 5, zone_y + zone_h // 2 + 5, zone_w - 10, zone_h // 2 - 10)
            lower_surface = pygame.Surface((lower_rect.width, lower_rect.height), pygame.SRCALPHA)
            lower_surface.fill((56, 189, 248, lower_intensity // 3))  # Blue with transparency
            self.screen.blit(lower_surface, lower_rect)

            # Motion indicator bar
            bar_width = min(zone_w - 20, int(motion_lower / max_motion * (zone_w - 20)))
            if bar_width > 0:
                pygame.draw.rect(self.screen, (56, 189, 248),
                               (zone_x + 10, zone_y + zone_h * 3 // 4 - 5, bar_width, 10))

        # Strum zone label
        label = self.font_small.render("STRUM ZONE", True, (100, 116, 139))
        self.screen.blit(label, (zone_x + 10, zone_y + 10))

        # Motion level text
        motion_text = self.font_small.render(f"Motion: {int(self.motion_detector.motion_level)}", True, (150, 150, 150))
        self.screen.blit(motion_text, (zone_x + 10, zone_y + zone_h - 25))

        # Draw strings
        for i in range(6):
            y_offset = i * 25
            active = game_state.fret_states[i] > 0
            color = COLOR_STRING_ACTIVE if active else COLOR_STRING_INACTIVE
            width = 6 if active else 2
            start_pos = (0, strum_zone_y + y_offset)
            end_pos = (WINDOW_WIDTH, strum_zone_y + y_offset + 120)
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)

    def draw_ui(self):
        """Draw UI."""
        score_text = self.font_large.render(f"{game_state.score:,}", True, COLOR_TEXT_WHITE)
        self.screen.blit(score_text, (50, 40))

        if game_state.combo > 0:
            combo_text = self.font_large.render(str(game_state.combo), True, COLOR_STRING_ACTIVE)
            self.screen.blit(combo_text, combo_text.get_rect(right=WINDOW_WIDTH - 50, bottom=WINDOW_HEIGHT - 100))

            combo_label = self.font_medium.render("COMBO!", True, COLOR_TEXT_WHITE)
            self.screen.blit(combo_label, combo_label.get_rect(right=WINDOW_WIDTH - 50, top=WINDOW_HEIGHT - 80))

        now = pygame.time.get_ticks()
        if self.hit_rating_text and now - self.hit_rating_time < 500:
            rating = self.font_large.render(self.hit_rating_text, True, self.hit_rating_color)
            self.screen.blit(rating, rating.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)))

        for i, fret in enumerate(game_state.fret_states):
            color = COLOR_STRING_ACTIVE if fret > 0 else COLOR_TEXT_SLATE
            fret_text = self.font_small.render(f"{STRING_NAMES[i]}: {fret}", True, color)
            self.screen.blit(fret_text, (50 + i * 100, WINDOW_HEIGHT - 80))

    def draw_particles(self):
        """Draw particles."""
        for particle in self.particles:
            particle.draw(self.screen)

    def draw_start_screen(self):
        """Draw start screen."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((2, 6, 23, 250))
        self.screen.blit(overlay, (0, 0))

        title = self.font_large.render("AIR GUITAR PRO", True, COLOR_STRING_ACTIVE)
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80)))

        # Show song name
        if self.selected_song_name in self.available_songs:
            song = self.available_songs[self.selected_song_name]
            song_text = self.font_medium.render(f"♪ {song.name} ♪", True, COLOR_PERFECT)
            self.screen.blit(song_text, song_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 10)))

        prompt = self.font_medium.render("Press SPACE to Start | ESC to Quit", True, COLOR_TEXT_WHITE)
        self.screen.blit(prompt, prompt.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60)))

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while game_state.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_state.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game_state.running = False
                    elif event.key == pygame.K_SPACE and not self.game_started and self.camera_ready:
                        self.game_started = True
                        self.start_song()  # Start song with game

            ret, frame = self.cap.read()
            if not ret:
                self.camera_ready = False
                continue
            self.camera_ready = True

            self.screen.fill(COLOR_BG)
            self.draw_video_feed(frame)

            if not self.game_started:
                self.draw_start_screen()
                self.clock.tick(FPS)
                continue

            # Detect motion for strumming
            strum_direction = self.motion_detector.detect_strum(frame)

            if strum_direction:
                # Add particles at strum zone
                zone_x, zone_y, zone_w, zone_h = self.motion_detector.get_strum_zone_rect()
                center_x = zone_x + zone_w // 2
                center_y = zone_y + zone_h // 2
                for _ in range(8):
                    self.particles.append(Particle(center_x, center_y, COLOR_STRING_ACTIVE))

            self.spawn_note()
            self.update_notes(strum_direction)
            self.update_particles()

            self.draw_rhythm_track()
            self.draw_strings()
            self.draw_ui()
            self.draw_particles()

            pygame.display.flip()
            self.clock.tick(FPS)

        self.cap.release()
        self.audio.stop_bgm()
        pygame.quit()


async def run_game():
    """Run game in async context."""
    print("Starting Air Guitar Pro...")
    webrtc_server = WebRTCServer(game_state)
    await webrtc_server.start()
    print("WebRTC server started")

    try:
        game = Game()
        game.run()
    finally:
        await webrtc_server.stop()
        logger.info("Server stopped")


def main():
    """Entry point."""
    try:
        asyncio.run(run_game())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
