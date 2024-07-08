# chappie.py

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar, QListWidget,
                             QTreeWidget, QTreeWidgetItem, QSplitter, QTextEdit, QInputDialog,
                             QLineEdit, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QTimer, QSettings, QThread
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import pyqtgraph as pg
import numpy as np
import librosa
import logging
from chappie_processor import ChappieProcessor
from chappie_utils import seconds_to_time, time_to_seconds

# Set up logging
logging.basicConfig(filename='chappie.log', level=logging.DEBUG)

class WaveformWidget(pg.PlotWidget):
    region_selected = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.setLabel('left', 'Amplitude')
        self.setLabel('bottom', 'Time')
        self.playhead = pg.InfiniteLine(pos=0, angle=90, movable=False, pen='r')
        self.addItem(self.playhead)
        self.waveform_item = None
        self.time_axis = None
        self.audio_duration = 0
        self.chapter_regions = []

        self.scene().sigMouseClicked.connect(self.on_mouse_clicked)

        self.time_axis_item = pg.AxisItem(orientation='bottom')
        self.setAxisItems({'bottom': self.time_axis_item})

    def plot_waveform(self, y, sr):
        try:
            self.clear()
            self.addItem(self.playhead)
            self.time_axis = np.arange(0, len(y)) / sr
            self.audio_duration = len(y) / sr

            downsample_factor = max(1, len(y) // 10000)
            y_downsampled = y[::downsample_factor]
            time_downsampled = self.time_axis[::downsample_factor]

            self.waveform_item = pg.PlotCurveItem(time_downsampled, y_downsampled, pen='b')
            self.addItem(self.waveform_item)
            self.setXRange(0, self.audio_duration)
            self.setYRange(y.min(), y.max())

            self.time_axis_item.setScale(1)
            self.time_axis_item.setTickSpacing(60, 30)
            self.time_axis_item.tickStrings = lambda values, scale, spacing: [seconds_to_time(v) for v in values]
        except Exception as e:
            logging.exception(f"Error plotting waveform: {str(e)}")

    def update_playhead(self, position):
        self.playhead.setPos(position)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom(0.9)
        else:
            self.zoom(1.1)
        event.accept()

    def zoom(self, factor):
        current_range = self.viewRange()
        x_min, x_max = current_range[0]
        center = (x_min + x_max) / 2
        new_width = (x_max - x_min) * factor
        max_width = self.audio_duration * 1.2  # Allow 20% more than full duration
        if new_width > max_width:
            new_width = max_width
        self.setXRange(center - new_width/2, center + new_width/2, padding=0)

    def mouseDragEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                pg.PlotWidget.mouseDragEvent(self, event)
        else:
            event.ignore()

    def on_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.sceneBoundingRect().contains(pos):
                mouse_point = self.getViewBox().mapSceneToView(pos)
                x = mouse_point.x()
                if 0 <= x <= self.audio_duration:
                    self.region_selected.emit(x, x)

    def add_chapter_region(self, start_time, end_time):
        region = pg.LinearRegionItem([start_time, end_time], movable=False, brush=pg.mkBrush(100, 100, 255, 50))
        self.addItem(region)
        self.chapter_regions.append(region)

class FileManager:
    def __init__(self):
        self.mp3_path = None
        self.transcript_path = None
        self.srt_path = None

    def check_files(self, base_path):
        self.mp3_path = base_path if base_path.endswith('.mp3') else None
        base_name = os.path.splitext(base_path)[0]
        self.transcript_path = f"{base_name}_transcript.txt" if os.path.exists(f"{base_name}_transcript.txt") else None
        self.srt_path = f"{base_name}_transcript.srt" if os.path.exists(f"{base_name}_transcript.srt") else None

    def files_status(self):
        return {
            'mp3': bool(self.mp3_path),
            'transcript': bool(self.transcript_path),
            'srt': bool(self.srt_path)
        }

class ChapterManager:
    def __init__(self):
        self.chapters = []

    def add_chapter(self, start, end, title):
        self.chapters.append({"start": start, "end": end, "title": title})

    def get_chapters(self):
        return self.chapters

class ProcessingThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, processor, srt_content):
        super().__init__()
        self.processor = processor
        self.srt_content = srt_content

    def run(self):
        try:
            result = self.processor.process_srt(self.srt_content)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ChappieGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chappie - Audio Chapter Creation and Navigation Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.file_manager = FileManager()
        self.chapter_manager = ChapterManager()
        self.chappie_processor = None
        self.settings = QSettings("YourCompany", "Chappie")

        self.setup_ui()

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

    def setup_ui(self):
        self.load_audio_button = QPushButton("Load Audio")
        self.load_audio_button.clicked.connect(self.load_audio_file)
        self.layout.addWidget(self.load_audio_button)

        self.waveform_widget = WaveformWidget()
        self.waveform_widget.region_selected.connect(self.on_waveform_clicked)
        self.layout.addWidget(self.waveform_widget)

        playback_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        playback_layout.addWidget(self.play_pause_button)
        playback_layout.addWidget(self.stop_button)
        self.layout.addLayout(playback_layout)

        self.chapter_list = QListWidget()
        self.chapter_list.itemClicked.connect(self.on_chapter_clicked)
        self.layout.addWidget(self.chapter_list)

        self.chapter_summary = QTextEdit()
        self.chapter_summary.setReadOnly(True)
        self.layout.addWidget(self.chapter_summary)

        self.set_api_key_button = QPushButton("Set API Key")
        self.set_api_key_button.clicked.connect(self.set_api_key)
        self.layout.addWidget(self.set_api_key_button)

        self.process_chapters_button = QPushButton("Process Chapters")
        self.process_chapters_button.clicked.connect(self.process_chapters)
        self.layout.addWidget(self.process_chapters_button)

        self.process_directory_button = QPushButton("Process Directory")
        self.process_directory_button.clicked.connect(self.process_directory)
        self.layout.addWidget(self.process_directory_button)

        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderLabels(["Table of Contents"])
        self.layout.addWidget(self.toc_tree)

    def load_audio_file(self):
        try:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("Audio Files (*.mp3);;All Files (*)")
            if file_dialog.exec():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    file_path = file_paths[0]
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                    self.file_manager.check_files(file_path)
                    self.update_file_status()
                    y, sr = librosa.load(file_path, sr=None)
                    self.waveform_widget.plot_waveform(y, sr)
                    self.media_player.setSource(QUrl.fromLocalFile(file_path))
                    QApplication.restoreOverrideCursor()
                    
                    # Check if SRT file exists and update UI
                    if not self.file_manager.srt_path:
                        QMessageBox.warning(self, "No SRT File", "No corresponding SRT file found.")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            logging.exception(f"Error loading audio file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load audio file: {str(e)}")

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_pause_button.setText("Play")
        else:
            self.media_player.play()
            self.play_pause_button.setText("Pause")
        QTimer.singleShot(100, self.update_playhead)

    def stop_audio(self):
        self.media_player.stop()
        self.play_pause_button.setText("Play")

    def update_playhead(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            position = self.media_player.position() / 1000.0
            self.waveform_widget.update_playhead(position)
            QTimer.singleShot(100, self.update_playhead)

    def on_waveform_clicked(self, time, _):
        self.media_player.setPosition(int(time * 1000))
        self.waveform_widget.update_playhead(time)

    def on_chapter_clicked(self, item):
        index = self.chapter_list.row(item)
        chapter = self.chapter_manager.get_chapters()[index]
        self.media_player.setPosition(int(chapter['start'] * 1000))
        self.play_audio()
        if 'summary' in chapter:
            self.chapter_summary.setText(f"Chapter Summary: {chapter['summary']}")

    def set_api_key(self):
        api_key, ok = QInputDialog.getText(self, "Set API Key", "Enter your OpenAI API Key:", QLineEdit.EchoMode.Password)
        if ok and api_key:
            self.settings.setValue("api_key", api_key)
            self.chappie_processor = ChappieProcessor(api_key)
            QMessageBox.information(self, "API Key Set", "API Key has been set successfully.")

    def process_chapters(self):
        api_key = self.settings.value("api_key")
        if not api_key:
            QMessageBox.warning(self, "No API Key", "Please set your OpenAI API key in the settings first.")
            return
        if not self.file_manager.srt_path:
            QMessageBox.warning(self, "No SRT File", "No SRT file found for the current audio.")
            return

        try:
            with open(self.file_manager.srt_path, 'r', encoding='utf-8') as srt_file:
                srt_content = srt_file.read()
            
            if not self.chappie_processor:
                self.chappie_processor = ChappieProcessor(api_key)
            
            self.processing_thread = ProcessingThread(self.chappie_processor, srt_content)
            self.processing_thread.finished.connect(self.on_processing_complete)
            self.processing_thread.error.connect(self.on_processing_error)
            self.processing_thread.start()
            
            self.progress_dialog = QProgressDialog("Processing chapters...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.show()
        except Exception as e:
            logging.exception(f"Error processing chapters: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process chapters: {str(e)}")

    def on_processing_complete(self, result):
        self.progress_dialog.close()
        if 'chapters' in result and 'chapter_summaries' in result and 'overall_summary' in result:
            self.chapter_manager.chapters = result['chapters']
            self.update_chapter_list()
            self.update_table_of_contents()
            self.chapter_summary.setText(f"Overall Summary: {result['overall_summary']}")
            for chapter in result['chapters']:
                self.waveform_widget.add_chapter_region(chapter['start'], chapter['end'])
            QMessageBox.information(self, "Processing Complete", "Chapters have been processed successfully.")
        else:
            QMessageBox.warning(self, "Processing Issue", "The chapter processing didn't return the expected results.")

    def on_processing_error(self, error_message):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Error", f"Failed to process chapters: {error_message}")

    def process_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            api_key = self.settings.value("api_key")
            if not api_key:
                QMessageBox.warning(self, "No API Key", "Please set your OpenAI API key in the settings first.")
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                if not self.chappie_processor:
                    self.chappie_processor = ChappieProcessor(api_key)
                
                results = self.chappie_processor.process_directory(directory)
                self.update_table_of_contents(results)
                QMessageBox.information(self, "Processing Complete", f"Processed {len(results)} files successfully.")
            except Exception as e:
                logging.exception(f"Error processing directory: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to process directory: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()

    def update_table_of_contents(self, results=None):
        self.toc_tree.clear()
        if results:
            for filename, result in results.items():
                file_item = QTreeWidgetItem(self.toc_tree)
                file_item.setText(0, filename)
                for i, chapter in enumerate(result['chapters']):
                    chapter_item = QTreeWidgetItem(file_item)
                    chapter_item.setText(0, f"Chapter {i+1}: {result['chapter_summaries'][i]}")
        elif self.chapter_manager.chapters:
            for i, chapter in enumerate(self.chapter_manager.chapters):
                item = QTreeWidgetItem(self.toc_tree)
                item.setText(0, f"Chapter {i+1}: {chapter.get('title', '')}")
        self.toc_tree.expandAll()

    def update_file_status(self):
        status = self.file_manager.files_status()
        status_text = f"MP3: {'✓' if status['mp3'] else '✗'} | "
        status_text += f"Transcript: {'✓' if status['transcript'] else '✗'} | "
        status_text += f"SRT: {'✓' if status['srt'] else '✗'}"
        QMessageBox.information(self, "File Status", status_text)

    def update_chapter_list(self):
        self.chapter_list.clear()
        for chapter in self.chapter_manager.get_chapters():
            title = chapter.get('title', 'Untitled Chapter')
            start = seconds_to_time(chapter.get('start', 0))
            end = seconds_to_time(chapter.get('end', 0))
            self.chapter_list.addItem(f"{title} ({start} - {end})")

    def play_audio(self):
        self.media_player.play()
        self.play_pause_button.setText("Pause")
        QTimer.singleShot(100, self.update_playhead)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = ChappieGUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.exception("An error occurred:")
        print(f"An error occurred: {str(e)}")