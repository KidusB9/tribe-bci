"""
Predictive Speller GUI — PyQt6 binary-tree word selector with P300 flashing.

Replaces the terminal renderer with a proper fullscreen GUI designed for
paralyzed users. The screen flashes groups of words; the user's brain
produces a P300 "yes" response to the attended group. The system narrows
down via a binary tree until a single word is selected. An integrated
language model predicts the next likely word, allowing the user to skip
the tree entirely for common continuations.

Usage:
    python -m reverse_bci.ui.predictive_speller [--headset simulated]

Keyboard shortcuts (for caregivers / testing):
    1-6      Manually select a group (bypasses P300)
    Backspace  Undo last word
    Escape     Clear sentence / stop flashing
    F11        Toggle fullscreen
    Space      Start/pause flashing cycle
"""

import sys
import time
import math
import logging
import threading
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette, QKeyEvent, QShortcut, QKeySequence

from reverse_bci.ui.p300_speller import P300Config, P300Detector, WordTree
from reverse_bci.text_decoder import BCI_VOCABULARY, TextDecoder, TextDecoderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette — high-contrast, optimised for visibility from a hospital bed
# ---------------------------------------------------------------------------

COLORS = {
    "bg":              "#0D1117",
    "panel":           "#161B22",
    "panel_border":    "#30363D",
    "text":            "#E6EDF3",
    "text_dim":        "#8B949E",
    "flash":           "#FFD700",
    "flash_text":      "#000000",
    "selected":        "#238636",
    "selected_text":   "#FFFFFF",
    "prediction":      "#1F6FEB",
    "prediction_text": "#FFFFFF",
    "sentence_bg":     "#0D1117",
    "sentence_text":   "#58A6FF",
    "uncertain":       "#DA3633",
    "group_hover":     "#21262D",
}

FONT_FAMILY = "Segoe UI"
GLOBAL_STYLE = f"""
    QMainWindow, QWidget {{
        background-color: {COLORS['bg']};
        color: {COLORS['text']};
        font-family: '{FONT_FAMILY}';
    }}
"""


def _build_vocabulary() -> list[str]:
    """Flatten BCI_VOCABULARY minus letters/numbers for the word tree."""
    categories = [
        "essential", "needs", "emotions", "communication",
        "people", "body", "time", "questions",
    ]
    words: list[str] = []
    seen: set[str] = set()
    for cat in categories:
        for w in BCI_VOCABULARY.get(cat, []):
            if w not in seen:
                seen.add(w)
                words.append(w)
    return words


# ---------------------------------------------------------------------------
# Word-group card widget
# ---------------------------------------------------------------------------

class GroupCard(QFrame):
    """One of the N on-screen groups. Highlights when flashed."""

    clicked = pyqtSignal(int)

    def __init__(self, group_index: int, parent=None):
        super().__init__(parent)
        self.group_index = group_index
        self._words: list[str] = []
        self._is_flashing = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        self._index_label = QLabel(f"Group {group_index + 1}")
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._index_label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
        layout.addWidget(self._index_label)

        self._words_label = QLabel("")
        self._words_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._words_label.setWordWrap(True)
        self._words_label.setFont(QFont(FONT_FAMILY, 18, QFont.Weight.DemiBold))
        layout.addWidget(self._words_label, stretch=1)

        self._apply_idle_style()

    # -- public API --

    def set_words(self, words: list[str]):
        self._words = words
        display = "  ·  ".join(words) if words else "(empty)"
        self._words_label.setText(display)

    def flash_on(self):
        self._is_flashing = True
        self.setStyleSheet(f"""
            GroupCard {{
                background-color: {COLORS['flash']};
                border: 3px solid {COLORS['flash']};
                border-radius: 12px;
            }}
        """)
        self._index_label.setStyleSheet(f"color: {COLORS['flash_text']};")
        self._words_label.setStyleSheet(f"color: {COLORS['flash_text']};")

    def flash_off(self):
        self._is_flashing = False
        self._apply_idle_style()

    def mark_selected(self):
        self.setStyleSheet(f"""
            GroupCard {{
                background-color: {COLORS['selected']};
                border: 3px solid {COLORS['selected']};
                border-radius: 12px;
            }}
        """)
        self._index_label.setStyleSheet(f"color: {COLORS['selected_text']};")
        self._words_label.setStyleSheet(f"color: {COLORS['selected_text']};")

    # -- internals --

    def _apply_idle_style(self):
        self.setStyleSheet(f"""
            GroupCard {{
                background-color: {COLORS['panel']};
                border: 2px solid {COLORS['panel_border']};
                border-radius: 12px;
            }}
            GroupCard:hover {{
                background-color: {COLORS['group_hover']};
            }}
        """)
        self._index_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        self._words_label.setStyleSheet(f"color: {COLORS['text']};")

    def mousePressEvent(self, event):
        self.clicked.emit(self.group_index)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Prediction chip widget
# ---------------------------------------------------------------------------

class PredictionChip(QPushButton):
    """Clickable next-word prediction."""

    word_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.DemiBold))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(48)
        self.setMinimumWidth(100)
        self._word = ""
        self._apply_style()
        self.clicked.connect(self._on_click)

    def set_prediction(self, word: str, rank: int):
        self._word = word
        self.setText(word)
        self.setVisible(True)
        self._apply_style()

    def clear_prediction(self):
        self._word = ""
        self.setText("")
        self.setVisible(False)

    def _apply_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['prediction']};
                color: {COLORS['prediction_text']};
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #388BFD;
            }}
            QPushButton:pressed {{
                background-color: #1158C7;
            }}
        """)

    def _on_click(self):
        if self._word:
            self.word_selected.emit(self._word)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PredictiveSpellerWindow(QMainWindow):
    """Full-screen P300 predictive speller GUI."""

    _flash_signal = pyqtSignal(int)       # group index to flash (-1 = all off)
    _groups_signal = pyqtSignal(list)      # new group layout
    _selection_signal = pyqtSignal(int, float)  # (group_idx, confidence)
    _uncertain_signal = pyqtSignal(float)  # confidence
    _word_signal = pyqtSignal(str)         # word decoded

    def __init__(
        self,
        vocabulary: Optional[list[str]] = None,
        config: Optional[P300Config] = None,
        sfreq: float = 256.0,
        n_channels: int = 4,
        eeg_getter: Optional[Callable] = None,
    ):
        super().__init__()
        self.config = config or P300Config()
        self.vocabulary = vocabulary or _build_vocabulary()
        self.sfreq = sfreq
        self.n_channels = n_channels
        self._eeg_getter = eeg_getter

        self.tree = WordTree(self.vocabulary, self.config.group_size)
        self.detector = P300Detector(sfreq, n_channels, self.config)

        self._sentence: list[str] = []
        self._is_flashing = False
        self._flash_thread: Optional[threading.Thread] = None

        self._next_word_predictions = _COMMON_NEXT_WORDS

        self._init_ui()
        self._connect_signals()
        self._refresh_groups()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _init_ui(self):
        self.setWindowTitle("Predictive Speller — Reverse BCI")
        self.setStyleSheet(GLOBAL_STYLE)
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(24, 16, 24, 16)
        root_layout.setSpacing(12)

        # -- Top: sentence display --
        self._sentence_frame = QFrame()
        self._sentence_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['sentence_bg']};
                border: 2px solid {COLORS['panel_border']};
                border-radius: 12px;
            }}
        """)
        sent_layout = QVBoxLayout(self._sentence_frame)
        sent_layout.setContentsMargins(24, 16, 24, 16)

        self._sentence_label = QLabel("Begin by focusing on the group that contains your word.")
        self._sentence_label.setFont(QFont(FONT_FAMILY, 26, QFont.Weight.Bold))
        self._sentence_label.setStyleSheet(f"color: {COLORS['sentence_text']};")
        self._sentence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sentence_label.setWordWrap(True)
        sent_layout.addWidget(self._sentence_label)

        root_layout.addWidget(self._sentence_frame)

        # -- Predictions row --
        pred_layout = QHBoxLayout()
        pred_layout.setSpacing(10)

        pred_label = QLabel("Next word:")
        pred_label.setFont(QFont(FONT_FAMILY, 14))
        pred_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        pred_layout.addWidget(pred_label)

        self._prediction_chips: list[PredictionChip] = []
        for i in range(self.config.max_predictions):
            chip = PredictionChip()
            chip.word_selected.connect(self._on_prediction_selected)
            chip.clear_prediction()
            self._prediction_chips.append(chip)
            pred_layout.addWidget(chip)

        pred_layout.addStretch()

        # Undo / Clear buttons
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setFont(QFont(FONT_FAMILY, 13))
        self._undo_btn.setMinimumHeight(40)
        self._undo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['panel']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['panel_border']};
                border-radius: 8px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{ background-color: {COLORS['group_hover']}; }}
        """)
        self._undo_btn.clicked.connect(self._undo_word)
        pred_layout.addWidget(self._undo_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFont(QFont(FONT_FAMILY, 13))
        self._clear_btn.setMinimumHeight(40)
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['uncertain']};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{ background-color: #E5534B; }}
        """)
        self._clear_btn.clicked.connect(self._clear_sentence)
        pred_layout.addWidget(self._clear_btn)

        root_layout.addLayout(pred_layout)

        # -- Group cards grid --
        self._grid_frame = QFrame()
        self._grid_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._grid_layout = QGridLayout(self._grid_frame)
        self._grid_layout.setSpacing(12)

        self._group_cards: list[GroupCard] = []
        n = self.config.group_size
        cols = 3 if n >= 6 else 2
        rows = math.ceil(n / cols)
        for i in range(n):
            card = GroupCard(i)
            card.clicked.connect(self._on_group_clicked)
            self._group_cards.append(card)
            r, c = divmod(i, cols)
            self._grid_layout.addWidget(card, r, c)

        root_layout.addWidget(self._grid_frame, stretch=1)

        # -- Bottom status bar --
        status_layout = QHBoxLayout()
        status_layout.setSpacing(24)

        self._status_label = QLabel("Press Space to start flashing  ·  1-6 to select manually")
        self._status_label.setFont(QFont(FONT_FAMILY, 13))
        self._status_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        status_layout.addWidget(self._status_label, stretch=1)

        self._depth_label = QLabel("")
        self._depth_label.setFont(QFont(FONT_FAMILY, 13))
        self._depth_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        status_layout.addWidget(self._depth_label)

        self._remaining_label = QLabel("")
        self._remaining_label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
        self._remaining_label.setStyleSheet(f"color: {COLORS['sentence_text']};")
        status_layout.addWidget(self._remaining_label)

        root_layout.addLayout(status_layout)

        # -- Flash timer (drives the flashing cycle from the main thread) --
        self._flash_timer = QTimer(self)
        self._flash_timer.setTimerType(Qt.TimerType.PreciseTimer)

    # -----------------------------------------------------------------------
    # Signal wiring
    # -----------------------------------------------------------------------

    def _connect_signals(self):
        self._flash_signal.connect(self._handle_flash)
        self._groups_signal.connect(self._handle_groups)
        self._selection_signal.connect(self._handle_selection)
        self._uncertain_signal.connect(self._handle_uncertain)
        self._word_signal.connect(self._handle_word)

    # -----------------------------------------------------------------------
    # Keyboard
    # -----------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        # 1-6  → manual group selection (for caregiver / testing)
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_6:
            idx = key - Qt.Key.Key_1
            if idx < len(self._group_cards):
                self._on_group_clicked(idx)
            return

        if key == Qt.Key.Key_Space:
            self._toggle_flashing()
            return

        if key == Qt.Key.Key_Backspace:
            self._undo_word()
            return

        if key == Qt.Key.Key_Escape:
            if self._is_flashing:
                self._stop_flashing()
            else:
                self._clear_sentence()
            return

        if key == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            return

        super().keyPressEvent(event)

    # -----------------------------------------------------------------------
    # Group display
    # -----------------------------------------------------------------------

    def _refresh_groups(self):
        groups = self.tree.get_groups()
        for i, card in enumerate(self._group_cards):
            if i < len(groups):
                card.set_words(groups[i])
                card.flash_off()
                card.setVisible(bool(groups[i]))
            else:
                card.set_words([])
                card.setVisible(False)

        self._depth_label.setText(f"Level {self.tree.depth}")
        self._remaining_label.setText(f"{self.tree.n_remaining} words remaining")

    # -----------------------------------------------------------------------
    # Manual group selection (click or keypress)
    # -----------------------------------------------------------------------

    def _on_group_clicked(self, group_idx: int):
        if self._is_flashing:
            return

        groups = self.tree.get_groups()
        if group_idx >= len(groups) or not groups[group_idx]:
            return

        self._group_cards[group_idx].mark_selected()
        QTimer.singleShot(300, lambda: self._select_group(group_idx))

    def _select_group(self, group_idx: int):
        self.tree.select_group(group_idx)

        if self.tree.is_selected():
            word = self.tree.get_selected_word()
            if word:
                self._add_word(word)
            self.tree.reset()

        self._refresh_groups()

    # -----------------------------------------------------------------------
    # P300 flashing cycle
    # -----------------------------------------------------------------------

    def _toggle_flashing(self):
        if self._is_flashing:
            self._stop_flashing()
        else:
            self._start_flashing()

    def _start_flashing(self):
        if self._is_flashing:
            return
        self._is_flashing = True
        self._status_label.setText("Flashing — focus on the group with your word")
        self._status_label.setStyleSheet(f"color: {COLORS['flash']};")

        self._flash_thread = threading.Thread(target=self._flash_loop, daemon=True)
        self._flash_thread.start()

    def _stop_flashing(self):
        self._is_flashing = False
        self._status_label.setText("Press Space to start flashing  ·  1-6 to select manually")
        self._status_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        for card in self._group_cards:
            card.flash_off()

    def _flash_loop(self):
        """Run the P300 stimulus cycle in a background thread.

        Emits Qt signals to update the UI from the main thread.
        """
        while self._is_flashing and not self.tree.is_selected():
            groups = self.tree.get_groups()
            n_groups = len([g for g in groups if g])

            epochs_per_group: dict[int, list[np.ndarray]] = {i: [] for i in range(n_groups)}

            for _round in range(self.config.rounds_per_selection):
                order = list(range(n_groups))
                np.random.shuffle(order)

                for gidx in order:
                    if not self._is_flashing:
                        return

                    # Flash on
                    self._flash_signal.emit(gidx)
                    time.sleep(self.config.flash_duration)

                    # Capture EEG epoch
                    if self._eeg_getter:
                        epoch = self._eeg_getter()
                        if epoch is not None:
                            epochs_per_group[gidx].append(epoch)

                    # Flash off
                    self._flash_signal.emit(-1)
                    time.sleep(self.config.inter_flash_interval)

            if not self._is_flashing:
                return

            # P300 detection
            best_group, confidence = self.detector.detect(epochs_per_group)

            if confidence >= self.config.min_confidence:
                self._selection_signal.emit(best_group, confidence)
            else:
                self._uncertain_signal.emit(confidence)

        self._is_flashing = False

    # -----------------------------------------------------------------------
    # Signal handlers (run on the main / UI thread)
    # -----------------------------------------------------------------------

    def _handle_flash(self, group_idx: int):
        for card in self._group_cards:
            card.flash_off()
        if 0 <= group_idx < len(self._group_cards):
            self._group_cards[group_idx].flash_on()

    def _handle_groups(self, groups: list):
        self._refresh_groups()

    def _handle_selection(self, group_idx: int, confidence: float):
        self._group_cards[group_idx].mark_selected()
        self._status_label.setText(f"Selected group {group_idx + 1}  (confidence {confidence:.0%})")
        self._status_label.setStyleSheet(f"color: {COLORS['selected']};")

        QTimer.singleShot(500, lambda: self._select_group(group_idx))

    def _handle_uncertain(self, confidence: float):
        self._status_label.setText(f"Uncertain ({confidence:.0%}) — repeating…")
        self._status_label.setStyleSheet(f"color: {COLORS['uncertain']};")

    def _handle_word(self, word: str):
        self._add_word(word)

    # -----------------------------------------------------------------------
    # Sentence management
    # -----------------------------------------------------------------------

    def _add_word(self, word: str):
        self._sentence.append(word)
        self._update_sentence_display()
        self._update_predictions()

    def _undo_word(self):
        if self._sentence:
            self._sentence.pop()
            self._update_sentence_display()
            self._update_predictions()
            self.tree.reset()
            self._refresh_groups()

    def _clear_sentence(self):
        self._sentence.clear()
        self._update_sentence_display()
        self._update_predictions()
        self.tree.reset()
        self._refresh_groups()

    def _update_sentence_display(self):
        if self._sentence:
            text = " ".join(self._sentence)
            self._sentence_label.setText(text)
        else:
            self._sentence_label.setText(
                "Begin by focusing on the group that contains your word."
            )

    # -----------------------------------------------------------------------
    # Next-word predictions
    # -----------------------------------------------------------------------

    def _update_predictions(self):
        predictions = self._get_predictions()
        for i, chip in enumerate(self._prediction_chips):
            if i < len(predictions):
                chip.set_prediction(predictions[i], i)
            else:
                chip.clear_prediction()

    def _get_predictions(self) -> list[str]:
        if not self._sentence:
            return ["i want", "i need", "i feel", "help", "yes", "no"]

        last = self._sentence[-1].lower()
        preds = self._next_word_predictions.get(last, [])

        if not preds and len(self._sentence) >= 2:
            bigram = f"{self._sentence[-2].lower()} {last}"
            preds = self._next_word_predictions.get(bigram, [])

        if not preds:
            preds = ["yes", "no", "please", "help"]

        return preds[: self.config.max_predictions]

    def _on_prediction_selected(self, word: str):
        if self._is_flashing:
            self._stop_flashing()
        self.tree.reset()
        self._add_word(word)
        self._refresh_groups()

    # -----------------------------------------------------------------------
    # Public API (for integration with BCIPipeline)
    # -----------------------------------------------------------------------

    def set_eeg_source(self, getter: Callable[[], Optional[np.ndarray]]):
        self._eeg_getter = getter

    @property
    def sentence(self) -> str:
        return " ".join(self._sentence)

    def set_predictions_dict(self, predictions: dict[str, list[str]]):
        self._next_word_predictions = predictions


# ---------------------------------------------------------------------------
# Common next-word predictions (same as p300_speller.py, extended)
# ---------------------------------------------------------------------------

_COMMON_NEXT_WORDS: dict[str, list[str]] = {
    "i": ["want", "need", "feel", "think", "am", "don't know"],
    "i want": ["water", "food", "help", "medicine", "rest", "to"],
    "i need": ["help", "water", "medicine", "bathroom", "rest", "doctor"],
    "i feel": ["pain", "tired", "happy", "sad", "cold", "hot"],
    "i think": ["yes", "no", "maybe", "so", "help"],
    "i know": ["yes", "no", "thank you"],
    "please": ["help", "water", "stop", "wait", "come", "turn"],
    "thank": ["you"],
    "thank you": ["please", "everyone", "doctor", "nurse"],
    "yes": ["please", "now", "more", "thank you"],
    "no": ["stop", "don't understand", "thank you", "wait"],
    "help": ["me", "please", "now", "doctor"],
    "want": ["water", "food", "help", "sleep", "medicine"],
    "need": ["help", "water", "medicine", "bathroom", "doctor"],
    "feel": ["pain", "tired", "happy", "sad", "cold", "hot"],
    "pain": ["help", "medicine", "doctor", "stop", "head", "back"],
    "water": ["please", "more", "cold", "now"],
    "food": ["please", "more", "hungry", "now"],
    "turn": ["left", "right", "light", "tv", "music"],
    "more": ["water", "food", "medicine", "light", "music"],
    "less": ["light", "quiet", "pain", "medicine"],
    "tell": ["me", "doctor", "nurse", "family"],
    "show": ["me"],
    "don't understand": ["repeat", "help", "show me"],
    "happy": ["thank you", "love", "family", "yes"],
    "sad": ["help", "family", "love", "lonely"],
    "tired": ["sleep", "rest", "later"],
    "stop": ["please", "pain", "now"],
    "doctor": ["help", "please", "now", "medicine"],
    "nurse": ["help", "please", "water", "medicine"],
    "mom": ["love", "help", "please", "thank you"],
    "dad": ["love", "help", "please", "thank you"],
    "love": ["you", "family", "mom", "dad"],
    "good": ["morning", "afternoon", "evening", "night"],
    "hello": ["everyone", "doctor", "nurse", "friend"],
    "goodbye": ["love", "thank you", "everyone"],
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Predictive Speller — P300 BCI word selector"
    )
    parser.add_argument(
        "--headset", default="simulated",
        choices=["simulated", "muse", "emotiv_epoc", "emotiv_insight"],
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--group-size", type=int, default=6,
        help="Number of word groups displayed (default 6)",
    )
    parser.add_argument(
        "--flash-duration", type=float, default=0.2,
        help="Flash duration in seconds",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="P300 detection rounds per selection level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = P300Config(
        group_size=args.group_size,
        flash_duration=args.flash_duration,
        rounds_per_selection=args.rounds,
    )

    # Optional: hook up a real headset for EEG
    eeg_getter = None
    if args.headset != "simulated":
        try:
            from reverse_bci.headsets.simulated import SimulatedHeadset
            headset = SimulatedHeadset(headset_type="muse", embed_signal=True)
            headset.connect()
            headset.start_streaming()

            def _get_epoch():
                data = headset.get_latest_window(
                    int(0.6 * headset.sfreq)  # 600 ms epoch
                )
                return data

            eeg_getter = _get_epoch
            logger.info("Connected to simulated headset for demo P300")
        except Exception as e:
            logger.warning("Could not set up headset: %s", e)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Window, QColor(COLORS["bg"]))
    dark.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    dark.setColor(QPalette.ColorRole.Base, QColor(COLORS["panel"]))
    dark.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    dark.setColor(QPalette.ColorRole.Button, QColor(COLORS["panel"]))
    dark.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    app.setPalette(dark)

    window = PredictiveSpellerWindow(
        config=config,
        eeg_getter=eeg_getter,
    )

    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
