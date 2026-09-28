"""Main application window integrating PDF viewer and chat."""

import os
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, QThread, QObject, Signal, QSettings
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QStatusBar, QMenuBar, QMenu, QAction,
    QFileDialog, QMessageBox, QToolBar, QLabel, QComboBox,
    QPushButton, QColorDialog
)

from .pdf_viewer import PdfViewerWidget
from .chat_widget import ChatWidget
from ..document.loader import DocumentLoader
from ..document.parser import DocumentParser
from ..rag.retriever import Retriever


class MainWindow(QMainWindow):
    """Main application window with PDF viewer and chat."""
    
    def __init__(self):
        super().__init__()
        
        self.document_loader: Optional[DocumentLoader] = None
        self.document_parser: Optional[DocumentParser] = None
        self.retriever: Optional[Retriever] = None
        self.current_pdf_path: Optional[str] = None
        self.highlights: List = []  # Store highlight annotations
        
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_drag_drop()
        
        # Load settings
        self._load_settings()
    
    def _setup_ui(self) -> None:
        """Initialize the main UI."""
        self.setWindowTitle("Literature Buddy")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Splitter for PDF viewer and chat
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        
        # PDF viewer (left side, 70%)
        self.pdf_viewer = PdfViewerWidget()
        self.pdf_viewer.pdf_loaded.connect(self._on_pdf_loaded)
        self.pdf_viewer.error_occurred.connect(self._on_error)
        splitter.addWidget(self.pdf_viewer)
        
        # Chat widget (right side, 30%)
        self.chat_widget = ChatWidget()
        splitter.addWidget(self.chat_widget)
        
        # Set initial sizes
        splitter.setSizes([840, 360])
        
        main_layout.addWidget(splitter)
    
    def _setup_menu(self) -> None:
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open PDF", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_pdf)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu - Highlight only (removed Underline)
        edit_menu = menubar.addMenu("&Edit")
        
        highlight_action = QAction("&Highlight Selected Text", self)
        highlight_action.setShortcut("Ctrl+H")
        highlight_action.triggered.connect(self._on_highlight)
        edit_menu.addAction(highlight_action)
        
        clear_action = QAction("&Clear All Highlights", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self._on_clear_highlights)
        edit_menu.addAction(clear_action)
        
        edit_menu.addSeparator()
        
        # Highlight color selector
        color_action = QAction("Highlight &Color...", self)
        color_action.triggered.connect(self._on_select_color)
        edit_menu.addAction(color_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self) -> None:
        """Create toolbar with highlight and clear buttons (removed underline)."""
        toolbar = QToolBar("Annotation Tools")
        toolbar.setMovable(False)
        toolbar.setIconSize(Qt.size(24, 24))
        self.addToolBar(toolbar)
        
        # Highlight button
        self.highlight_action = QAction("🖍️ Highlight", self)
        self.highlight_action.setToolTip("Highlight selected text (Ctrl+H)")
        self.highlight_action.triggered.connect(self._on_highlight)
        toolbar.addAction(self.highlight_action)
        
        toolbar.addSeparator()
        
        # Clear highlights button
        self.clear_action = QAction("🗑️ Clear Highlights", self)
        self.clear_action.setToolTip("Clear all highlights (Ctrl+L)")
        self.clear_action.triggered.connect(self._on_clear_highlights)
        toolbar.addAction(self.clear_action)
        
        toolbar.addSeparator()
        
        # Highlight color button
        self.color_button = QPushButton("🎨 Color")
        self.color_button.setToolTip("Select highlight color")
        self.color_button.setFixedWidth(80)
        self.color_button.clicked.connect(self._on_select_color)
        toolbar.addWidget(self.color_button)
        
        # Current color indicator
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(24, 24)
        self.color_indicator.setStyleSheet("background-color: yellow; border: 1px solid black;")
        self.color_indicator.setToolTip("Current highlight color")
        toolbar.addWidget(self.color_indicator)
        
        self.highlight_color = Qt.yellow  # Default highlight color
    
    def _setup_statusbar(self) -> None:
        """Create status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready - Open a PDF to start")
    
    def _setup_drag_drop(self) -> None:
        """Enable drag and drop on main window."""
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter - accept PDF files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                url = urls[0]
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.pdf'):
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop - load PDF file."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                url = urls[0]
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.pdf'):
                        event.acceptProposedAction()
                        self._load_pdf(file_path)
                        return
        event.ignore()
    
    def _on_open_pdf(self) -> None:
        """Handle menu open PDF action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PDF",
            str(Path.home()),
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self._load_pdf(file_path)
    
    def _load_pdf(self, file_path: str) -> None:
        """Load PDF file and initialize document processing."""
        try:
            self.statusbar.showMessage(f"Loading: {file_path}")
            
            # Validate file
            if not os.path.exists(file_path):
                self._show_error("File not found")
                return
            
            # Load PDF through viewer
            self.pdf_viewer._load_pdf_safe(file_path)
            
        except Exception as e:
            self._show_error(f"Failed to load PDF: {str(e)}")
    
    def _on_pdf_loaded(self, file_path: str) -> None:
        """Handle PDF loaded signal."""
        self.current_pdf_path = file_path
        self.statusbar.showMessage(f"Loaded: {Path(file_path).name}")
        
        # Clear previous highlights
        self.highlights.clear()
        
        # Initialize document processing in background
        self._initialize_document_processing(file_path)
    
    def _initialize_document_processing(self, file_path: str) -> None:
        """Initialize document loader, parser, and retriever."""
        try:
            # Create document loader
            self.document_loader = DocumentLoader(file_path)
            
            # Create parser
            self.document_parser = DocumentParser()
            
            # Parse document
            chunks = self.document_parser.parse(file_path)
            
            # Create retriever with chunks
            self.retriever = Retriever(chunks)
            
            # Set up chat with context
            context = self._build_context_from_chunks(chunks)
            self.chat_widget.set_pdf_context(context)
            
            # Set model backend (adjust based on your actual backend)
            # self.chat_widget.set_model_backend(self.retriever)
            
            self.statusbar.showMessage(f"Ready - {len(chunks)} chunks indexed")
            
        except Exception as e:
            self._show_error(f"Failed to process document: {str(e)}")
    
    def _build_context_from_chunks(self, chunks) -> str:
        """Build context string from document chunks."""
        if not chunks:
            return ""
        
        # Combine first few chunks as context (adjust as needed)
        context_chunks = chunks[:5]
        context = "\n\n".join([str(chunk) for chunk in context_chunks])
        return context
    
    def _on_highlight(self) -> None:
        """Add highlight annotation to selected text."""
        if not self.pdf_viewer.current_pdf:
            self._show_error("No PDF loaded")
            return
        
        try:
            # Get selected text from PDF viewer
            # This assumes pdf_viewer has a method to get selection
            if hasattr(self.pdf_viewer, 'get_selected_text'):
                selected_text = self.pdf_viewer.get_selected_text()
                
                if not selected_text:
                    self._show_error("No text selected")
                    return
                
                # Add highlight annotation
                page_num = self.pdf_viewer.current_page if hasattr(self.pdf_viewer, 'current_page') else 0
                rect = self.pdf_viewer.get_selection_rect() if hasattr(self.pdf_viewer, 'get_selection_rect') else None
                
                if rect:
                    # Create highlight annotation
                    page = self.pdf_viewer.current_pdf[page_num]
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=self.highlight_color)
                    highlight.update()
                    
                    # Store highlight reference
                    self.highlights.append(highlight)
                    
                    self.statusbar.showMessage(f"Highlighted: {selected_text[:50]}...")
                else:
                    self._show_error("Could not get selection rectangle")
            else:
                self._show_error("Selection not supported in current viewer")
                
        except Exception as e:
            self._show_error(f"Highlight failed: {str(e)}")
    
    def _on_clear_highlights(self) -> None:
        """Clear all highlight annotations."""
        if not self.pdf_viewer.current_pdf:
            self._show_error("No PDF loaded")
            return
        
        try:
            # Remove all highlight annotations
            for page_num in range(len(self.pdf_viewer.current_pdf)):
                page = self.pdf_viewer.current_pdf[page_num]
                annots = page.annots()
                
                if annots:
                    for annot in annots:
                        if annot.type[0] == 8:  # Highlight annotation type
                            page.delete_annot(annot)
            
            # Clear highlights list
            self.highlights.clear()
            
            # Refresh display
            if hasattr(self.pdf_viewer, 'refresh'):
                self.pdf_viewer.refresh()
            
            self.statusbar.showMessage("All highlights cleared")
            
        except Exception as e:
            self._show_error(f"Clear highlights failed: {str(e)}")
    
    def _on_select_color(self) -> None:
        """Open color dialog to select highlight color."""
        color = QColorDialog.getColor(self.highlight_color, self, "Select Highlight Color")
        
        if color.isValid():
            self.highlight_color = color
            self.color_indicator.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid black;"
            )
            self.statusbar.showMessage(f"Highlight color set to {color.name()}")
    
    def _on_error(self, error_msg: str) -> None:
        """Handle error signal from widgets."""
        self._show_error(error_msg)
    
    def _show_error(self, message: str) -> None:
        """Show error message to user."""
        QMessageBox.critical(self, "Error", message)
        self.statusbar.showMessage(f"Error: {message}")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Literature Buddy",
            "Literature Buddy\n\n"
            "A PDF reader with AI-powered chat.\n\n"
            "Features:\n"
            "- Drag and drop PDF files\n"
            "- Click to open PDFs\n"
            "- Chat about document content\n"
            "- Highlight text (underline removed)\n"
            "- Clear all highlights\n"
            "- RAG-based retrieval\n\n"
            "© 2026"
        )
    
    def _load_settings(self) -> None:
        """Load application settings."""
        settings = QSettings("LiteratureBuddy", "MainWindow")
        
        # Restore window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore highlight color
        color = settings.value("highlight_color", Qt.yellow)
        if isinstance(color, str):
            # Convert from string if needed
            from PySide6.QtGui import QColor
            self.highlight_color = QColor(color)
        
        self.statusbar.showMessage("Settings loaded")
    
    def _save_settings(self) -> None:
        """Save application settings."""
        settings = QSettings("LiteratureBuddy", "MainWindow")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("highlight_color", self.highlight_color.name())
    
    def closeEvent(self, event) -> None:
        """Clean up on application close."""
        # Save settings
        self._save_settings()
        
        # Clean up chat widget
        if hasattr(self, 'chat_widget'):
            self.chat_widget.close()
        
        # Clean up PDF viewer
        if hasattr(self, 'pdf_viewer'):
            self.pdf_viewer.close()
        
        event.accept()