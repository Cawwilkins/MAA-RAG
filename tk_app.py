import json
import os
import queue
import re
import threading
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk, messagebox

from config import (
    METADATA_FACETS_PATH,
    RETRIEVE_ONLY_DEFAULT_TOP_K,
    RETRIEVE_ONLY_MIN_TOP_K,
    RETRIEVE_ONLY_MAX_TOP_K,
    RETRIEVE_ONLY_DEFAULT_RATIO,
)
from rag_service import RAGService


TEMPLATE_OPTIONS = {
    "Q/A": "q",
    "Summary": "s",
    "Exposure Analysis": "e",
    "Timeline": "t",
    "Reference Evidence": "r",
    "Comparison": "c",
}

TEMPLATE_DESCRIPTIONS = {
    "Q/A": "Best default. Direct factual answers grounded in retrieved context.",
    "Summary": "High-level summary of document purpose and key points.",
    "Exposure Analysis": "Asbestos-focused analysis: confirmed/possible/unsupported exposure.",
    "Timeline": "Chronological event extraction with dates/years and gaps called out.",
    "Reference Evidence": "Finds and formats matching reference entries/citations.",
    "Comparison": "Compares agreements, differences, contradictions, and missing info.",
}

TEMPLATE_PLACEHOLDERS = {
    "Q/A": "Ask a focused question about the documents (e.g., What does report A1234 say about asbestos exposure?)",
    "Summary": "Ask for a concise summary (e.g., Summarize the purpose and key findings of this document.)",
    "Exposure Analysis": "Ask for exposure analysis (e.g., Analyze confirmed vs possible asbestos exposure in this record.)",
    "Timeline": "Ask for chronology (e.g., Build a timeline of ships, rates, and assignments by year.)",
    "Reference Evidence": "Ask for references (e.g., Find reference entries about Portsmouth Naval Shipyard and asbestos.)",
    "Comparison": "Ask to compare records (e.g., Compare these two documents for differences in exposure evidence.)",
}
REFERENCE_EXTRACT_PLACEHOLDER = (
    "Extract grouped references (e.g., Extract all reference entries about asbestos for Portsmouth.)"
)
RETRIEVE_PRESETS = {
    "Strict": {
        "top_k": 40,
        "ratio": 0.95,
        "summary": "Strict keeps only the strongest matches. Best for very specific terms where precision matters most.",
    },
    "Balanced": {
        "top_k": 60,
        "ratio": 0.85,
        "summary": "Balanced keeps strong matches while allowing some related context. Good default for most searches.",
    },
    "Broad": {
        "top_k": RETRIEVE_ONLY_MAX_TOP_K,
        "ratio": 0.0,
        "summary": "Broad is very permissive. It keeps nearly everything retrieved so you can cast a wide net.",
    },
}


class MAAApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MAA Assistant")
        self.geometry("1280x820")
        self.minsize(1050, 680)

        self.colors = {
            "bg": "#0f172a",
            "panel": "#111827",
            "card": "#1e293b",
            "card_light": "#263449",
            "border": "#334155",
            "text": "#e5e7eb",
            "muted": "#94a3b8",
            "accent": "#38bdf8",
            "accent_dark": "#0284c7",
            "user_bubble": "#2563eb",
            "assistant_bubble": "#1f2937",
            "input": "#020617",
            "danger": "#ef4444",
            "success": "#22c55e",
        }

        self.configure(bg=self.colors["bg"])

        self.rag = None
        self.worker_queue = queue.Queue()
        self.facet_vars = {}
        self._detail_counter = 0
        self._request_start_time: float | None = None
        self._timer_after_id: str | None = None
        self._placeholder_active = False

        self._configure_styles()
        self._build_ui()
        self._load_facets()
        self._start_backend_load()
        self._poll_worker_queue()

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Sidebar.TFrame", background=self.colors["panel"], borderwidth=0)
        style.configure("Main.TFrame", background=self.colors["bg"])
        style.configure("Title.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 9))

        style.configure("Modern.TButton", background=self.colors["accent"], foreground="#00111f", borderwidth=0, padding=(14, 10), font=("Segoe UI", 10, "bold"))
        style.map("Modern.TButton", background=[("disabled", self.colors["border"]), ("active", self.colors["accent_dark"])], foreground=[("disabled", self.colors["muted"]), ("active", "#ffffff")])

        style.configure("Secondary.TButton", background=self.colors["card_light"], foreground=self.colors["text"], borderwidth=0, padding=(12, 8))
        style.map("Secondary.TButton", background=[("active", self.colors["border"])])

        style.configure("TCombobox", fieldbackground=self.colors["input"], background=self.colors["card_light"], foreground=self.colors["text"], arrowcolor=self.colors["text"], bordercolor=self.colors["border"], padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", self.colors["input"])], foreground=[("readonly", self.colors["text"])], selectbackground=[("readonly", self.colors["input"])], selectforeground=[("readonly", self.colors["text"])])
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground="#111111",
            bordercolor=self.colors["border"],
            insertcolor="#111111",
            padding=6,
        )

        style.configure("TCheckbutton", background=self.colors["panel"], foreground=self.colors["text"], focuscolor=self.colors["panel"], font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", self.colors["panel"])], foreground=[("active", self.colors["text"])])
        style.configure("TSeparator", background=self.colors["border"])

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(22, 20))
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.configure(width=380)
        self.sidebar.grid_propagate(False)
        self.sidebar.columnconfigure(0, weight=1)

        ttk.Label(self.sidebar, text="MAA Assistant", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.sidebar, text="Local document intelligence", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 22))

        ttk.Label(self.sidebar, text="Search Settings", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 10))
        ttk.Label(self.sidebar, text="Response length", style="Muted.TLabel").grid(row=3, column=0, sticky="w")
        self.response_length_var = tk.StringVar(value="MEDIUM")
        self.response_length_combo = ttk.Combobox(self.sidebar, textvariable=self.response_length_var, values=["SHORT", "MEDIUM", "LONG"], state="readonly")
        self.response_length_combo.grid(row=4, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(self.sidebar, text="Template", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        self.template_var = tk.StringVar(value="Q/A")
        self.template_combo = ttk.Combobox(self.sidebar, textvariable=self.template_var, values=list(TEMPLATE_OPTIONS.keys()), state="readonly")
        self.template_combo.grid(row=6, column=0, sticky="ew", pady=(4, 14))
        self.template_combo.bind("<<ComboboxSelected>>", self._on_template_changed)
        self.template_help_var = tk.StringVar(value=TEMPLATE_DESCRIPTIONS.get("Q/A", ""))
        ttk.Label(
            self.sidebar,
            textvariable=self.template_help_var,
            style="Muted.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=7, column=0, sticky="w", pady=(0, 10))

        self.show_used_nodes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Show used nodes", variable=self.show_used_nodes_var).grid(row=8, column=0, sticky="w", pady=(0, 16))

        ttk.Separator(self.sidebar).grid(row=9, column=0, sticky="ew", pady=(4, 10))
        ttk.Label(self.sidebar, text="Retrieve-Only Mode", style="Section.TLabel").grid(row=10, column=0, sticky="w", pady=(0, 8))
        self.retrieve_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Retrieve only (skip LLM answer)", variable=self.retrieve_only_var).grid(row=11, column=0, sticky="w", pady=(0, 6))
        self.reference_extract_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.sidebar,
            text="Reference extract mode (group refs by doc/page)",
            variable=self.reference_extract_var,
            command=self._on_reference_extract_toggled,
        ).grid(row=12, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(self.sidebar, text="View up to (max results)", style="Muted.TLabel").grid(row=13, column=0, sticky="w")
        self.retrieve_max_results_var = tk.StringVar(value=str(RETRIEVE_ONLY_DEFAULT_TOP_K))
        ttk.Entry(self.sidebar, textvariable=self.retrieve_max_results_var).grid(row=14, column=0, sticky="ew", pady=(4, 8))
        ttk.Label(self.sidebar, text="Relevance ratio (0.00 - 1.00)", style="Muted.TLabel").grid(row=15, column=0, sticky="w")
        self.retrieve_ratio_var = tk.StringVar(value=f"{RETRIEVE_ONLY_DEFAULT_RATIO:.2f}")
        ttk.Entry(self.sidebar, textvariable=self.retrieve_ratio_var).grid(row=16, column=0, sticky="ew", pady=(4, 14))

        ttk.Label(self.sidebar, text="Retrieve preset", style="Muted.TLabel").grid(row=17, column=0, sticky="w")
        preset_row = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        preset_row.grid(row=18, column=0, sticky="ew", pady=(4, 8))
        preset_row.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(
            preset_row,
            text="Strict",
            style="Secondary.TButton",
            command=lambda: self._apply_retrieve_preset("Strict"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            preset_row,
            text="Balanced",
            style="Secondary.TButton",
            command=lambda: self._apply_retrieve_preset("Balanced"),
        ).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(
            preset_row,
            text="Broad",
            style="Secondary.TButton",
            command=lambda: self._apply_retrieve_preset("Broad"),
        ).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        self.retrieve_preset_help_var = tk.StringVar(value=RETRIEVE_PRESETS["Balanced"]["summary"])
        ttk.Label(
            self.sidebar,
            textvariable=self.retrieve_preset_help_var,
            style="Muted.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=19, column=0, sticky="w", pady=(0, 8))

        ttk.Separator(self.sidebar).grid(row=20, column=0, sticky="ew", pady=(4, 18))
        ttk.Label(self.sidebar, text="Metadata Filters", style="Section.TLabel").grid(row=21, column=0, sticky="w", pady=(0, 10))

        self.filter_canvas = tk.Canvas(self.sidebar, bg=self.colors["panel"], highlightthickness=0, height=370)
        self.filter_canvas.grid(row=22, column=0, sticky="nsew")
        self.sidebar.rowconfigure(22, weight=1)

        self.filter_scrollbar = ttk.Scrollbar(self.sidebar, orient="vertical", command=self.filter_canvas.yview)
        self.filter_scrollbar.grid(row=22, column=1, sticky="ns")

        self.filters_frame = ttk.Frame(self.filter_canvas, style="Sidebar.TFrame")
        self.filters_frame.columnconfigure(0, weight=1)
        self.filter_window = self.filter_canvas.create_window((0, 0), window=self.filters_frame, anchor="nw")
        self.filter_canvas.configure(yscrollcommand=self.filter_scrollbar.set)
        self.filters_frame.bind("<Configure>", self._on_filters_configure)
        self.filter_canvas.bind("<Configure>", self._on_canvas_configure)

        self.clear_filters_button = ttk.Button(self.sidebar, text="Clear Filters", style="Secondary.TButton", command=self._clear_filters)
        self.clear_filters_button.grid(row=23, column=0, sticky="ew", pady=(16, 0))

        self.main = ttk.Frame(self, style="Main.TFrame", padding=(14, 16))
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(1, weight=1)

        self.header = ttk.Frame(self.main, style="Main.TFrame")
        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.header.columnconfigure(0, weight=1)

        tk.Label(self.header, text="Document Chat", bg=self.colors["bg"], fg=self.colors["text"], font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w")
        self.status_pill = tk.Label(self.header, text="Loading...", bg=self.colors["card"], fg=self.colors["muted"], font=("Segoe UI", 9, "bold"), padx=12, pady=6)
        self.status_pill.grid(row=0, column=1, sticky="e")

        self.chat_card = tk.Frame(self.main, bg=self.colors["card"], highlightbackground=self.colors["border"], highlightthickness=1)
        self.chat_card.grid(row=1, column=0, sticky="nsew")
        self.chat_card.columnconfigure(0, weight=1)
        self.chat_card.rowconfigure(0, weight=1)

        self.chat_text = tk.Text(self.chat_card, wrap="word", state="disabled", font=("Segoe UI", 10), padx=18, pady=18, bg=self.colors["card"], fg=self.colors["text"], insertbackground=self.colors["text"], borderwidth=0, highlightthickness=0, selectbackground=self.colors["accent_dark"])
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self.chat_card, command=self.chat_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_text.configure(yscrollcommand=self.scrollbar.set)

        self.chat_text.tag_configure("speaker_user", foreground="#bfdbfe", font=("Segoe UI", 10, "bold"), spacing1=12)
        self.chat_text.tag_configure("speaker_assistant", foreground=self.colors["accent"], font=("Segoe UI", 10, "bold"), spacing1=12)
        self.chat_text.tag_configure("user_msg", foreground="#ffffff", background=self.colors["user_bubble"], lmargin1=18, lmargin2=18, rmargin=90, spacing1=4, spacing3=12)
        self.chat_text.tag_configure("assistant_msg", foreground=self.colors["text"], background=self.colors["assistant_bubble"], lmargin1=18, lmargin2=18, rmargin=70, spacing1=4, spacing3=12)
        self.chat_text.tag_configure("result_header", foreground=self.colors["accent"], font=("Segoe UI", 10, "bold"))
        self.chat_text.tag_configure("result_row", foreground=self.colors["text"])
        self.chat_text.tag_configure("result_toggle", foreground="#93c5fd", underline=True)
        self.chat_text.tag_configure("result_detail", foreground="#cbd5e1")
        self.chat_text.tag_configure("result_link", foreground="#93c5fd", underline=True)

        self.input_card = tk.Frame(self.main, bg=self.colors["input"], highlightbackground=self.colors["border"], highlightthickness=1)
        self.input_card.grid(row=2, column=0, sticky="ew", pady=(14, 8))
        self.input_card.columnconfigure(0, weight=1)

        self.question_entry = tk.Text(self.input_card, height=4, wrap="word", font=("Segoe UI", 11), bg=self.colors["input"], fg=self.colors["text"], insertbackground=self.colors["text"], borderwidth=0, highlightthickness=0, padx=14, pady=12)
        self.question_entry.grid(row=0, column=0, sticky="ew")
        self.question_entry.bind("<Return>", self._on_enter_submit)
        self.question_entry.bind("<KP_Enter>", self._on_enter_submit)
        self.question_entry.bind("<Shift-Return>", self._on_shift_enter_newline)
        self.question_entry.bind("<Shift-KP_Enter>", self._on_shift_enter_newline)
        self.question_entry.bind("<Control-Return>", self._on_shift_enter_newline)
        self.question_entry.bind("<Control-KP_Enter>", self._on_shift_enter_newline)
        self.question_entry.bind("<FocusIn>", self._on_question_focus_in)
        self.question_entry.bind("<FocusOut>", self._on_question_focus_out)

        self.ask_button = ttk.Button(self.input_card, text="Ask", style="Modern.TButton", command=self._ask_question, state="disabled")
        self.ask_button.grid(row=0, column=1, sticky="ns", padx=(8, 10), pady=10)

        self.status_var = tk.StringVar(value="Loading local RAG system...")
        self.status_label = ttk.Label(self.main, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=3, column=0, sticky="w")

        self._append_assistant_message("MAA Assistant is starting locally. Once loading finishes, ask a question below.")
        self._set_question_placeholder(force=True)

    def _on_filters_configure(self, event):
        self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.filter_canvas.itemconfigure(self.filter_window, width=event.width)

    def _load_facets(self):
        for child in self.filters_frame.winfo_children():
            child.destroy()
        facets_path = Path(METADATA_FACETS_PATH)
        if not facets_path.exists():
            ttk.Label(self.filters_frame, text="No metadata facets found.", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            return
        try:
            with facets_path.open("r", encoding="utf-8") as f:
                facets = json.load(f)
        except Exception as e:
            ttk.Label(self.filters_frame, text=f"Could not load facets: {e}", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            return
        row = 0
        self.facet_vars.clear()
        for field, mapping in facets.items():
            if not isinstance(mapping, dict):
                continue
            values = sorted(mapping.keys())
            if not values:
                continue
            label = field.replace("_", " ").title()
            ttk.Label(self.filters_frame, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=(0, 3))
            row += 1
            var = tk.StringVar(value="Any")
            combo = ttk.Combobox(self.filters_frame, textvariable=var, values=["Any"] + values, state="readonly")
            combo.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            row += 1
            self.facet_vars[field] = var

    def _on_template_changed(self, _event=None):
        selected = self.template_var.get()
        self.template_help_var.set(TEMPLATE_DESCRIPTIONS.get(selected, ""))
        current_text = self.question_entry.get("1.0", "end").strip()
        if self._placeholder_active or not current_text:
            self._set_question_placeholder(force=True)

    def _on_reference_extract_toggled(self):
        current_text = self.question_entry.get("1.0", "end").strip()
        if self._placeholder_active or not current_text:
            self._set_question_placeholder(force=True)

    def _apply_retrieve_preset(self, preset_name: str):
        preset = RETRIEVE_PRESETS.get(preset_name)
        if not preset:
            return
        self.retrieve_max_results_var.set(str(preset["top_k"]))
        self.retrieve_ratio_var.set(f"{preset['ratio']:.2f}")
        self.retrieve_preset_help_var.set(str(preset["summary"]))

    def _start_backend_load(self):
        def load_backend():
            try:
                rag = RAGService()
                self.worker_queue.put(("backend_loaded", rag))
            except Exception as e:
                self.worker_queue.put(("error", f"Failed to load backend: {e}"))
        threading.Thread(target=load_backend, daemon=True).start()

    def _poll_worker_queue(self):
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "backend_loaded":
                    self.rag = payload
                    self.ask_button.configure(state="normal")
                    self.status_var.set("Ready.")
                    self.status_pill.configure(text="Ready", fg=self.colors["success"], bg=self.colors["card"])
                    self._append_assistant_message("Ready. The local document assistant is loaded.")
                elif event == "answer":
                    if payload.get("reference_extract"):
                        self._append_reference_extract_results(payload)
                    elif payload.get("retrieve_only"):
                        self._append_retrieve_only_results(payload)
                    else:
                        if payload.get("used_nodes_text") and payload.get("sources"):
                            self._append_used_nodes_results(payload.get("sources", []))
                        self._append_assistant_message(payload["answer"])
                    self.ask_button.configure(state="normal")
                    elapsed = self._stop_request_timer()
                    if elapsed is not None:
                        self.status_var.set(f"Answer returned in {elapsed:.2f} seconds.")
                    else:
                        self.status_var.set("Answer returned.")
                    self.status_pill.configure(text="Ready", fg=self.colors["success"])
                elif event == "error":
                    self.ask_button.configure(state="normal" if self.rag else "disabled")
                    self._stop_request_timer()
                    self.status_var.set("Error.")
                    self.status_pill.configure(text="Error", fg=self.colors["danger"])
                    messagebox.showerror("MAA Assistant Error", payload)
                    self._append_assistant_message(f"Error: {payload}")
                elif event == "progress":
                    self._append_assistant_message(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_queue)

    def _append_user_message(self, text):
        self._append_message("You", text, speaker_tag="speaker_user", message_tag="user_msg")

    def _append_assistant_message(self, text):
        self._append_message("MAA Assistant", text, speaker_tag="speaker_assistant", message_tag="assistant_msg")

    def _append_message(self, speaker, text, speaker_tag, message_tag):
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", f"{speaker}\n", (speaker_tag,))
        self.chat_text.insert("end", f"{text}\n\n", (message_tag,))
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _compact_text(self, value: object) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _compact_value(self, value: object) -> str:
        if isinstance(value, list):
            items = [self._compact_text(item) for item in value if self._compact_text(item)]
            return ", ".join(items)
        return self._compact_text(value)

    def _toggle_details(self, detail_tag: str):
        current = self.chat_text.tag_cget(detail_tag, "elide")
        is_hidden = str(current) == "1"
        self.chat_text.tag_configure(detail_tag, elide=(not is_hidden))

    def _open_local_path(self, raw_path: str):
        try:
            path = Path(raw_path).expanduser()
            if path.exists():
                os.startfile(str(path))  # type: ignore[attr-defined]
                return
            if path.parent.exists():
                os.startfile(str(path.parent))  # type: ignore[attr-defined]
                return
            messagebox.showwarning("Path not found", f"Could not open: {raw_path}")
        except Exception as e:
            messagebox.showerror("Open path failed", str(e))

    def _append_retrieve_only_results(self, payload: dict):
        nodes = payload.get("sources", []) or []
        ratio = payload.get("retrieve_ratio")
        candidates = payload.get("retrieve_candidates_count", 0)
        kept = payload.get("retrieve_kept_count", 0)
        cut = payload.get("retrieve_cut_count", 0)
        requested = payload.get("retrieve_requested_count", candidates)
        scored_count = payload.get("retrieve_scored_count", 0)
        missing_score_count = payload.get("retrieve_missing_score_count", 0)
        top_score = payload.get("retrieve_top_score")
        bottom_score = payload.get("retrieve_bottom_score")
        threshold = payload.get("retrieve_threshold")

        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", "MAA Assistant\n", ("speaker_assistant",))
        self.chat_text.insert("end", "Retrieve-only results\n", ("result_header",))

        if not nodes:
            self.chat_text.insert(
                "end",
                "No results met the current threshold. Try lowering the ratio or increasing max results.\n\n",
                ("assistant_msg",),
            )
            self.chat_text.configure(state="disabled")
            self.chat_text.see("end")
            return

        for i, node in enumerate(nodes, start=1):
            meta = getattr(node, "metadata", {}) or {}
            score = getattr(node, "score", None)
            title = self._compact_text(meta.get("title")) or "Untitled source"
            page = self._compact_text(meta.get("page"))
            row = f"{i}. {title}"
            if page:
                row += f" (p. {page})"
            if isinstance(score, (int, float)):
                row += f" | score: {score:.4f}"

            self.chat_text.insert("end", row + "\n", ("result_row",))

            toggle_tag = f"toggle_details_{self._detail_counter}"
            detail_tag = f"details_block_{self._detail_counter}"
            self._detail_counter += 1

            self.chat_text.insert("end", "   Click to expand details\n", ("result_toggle", toggle_tag))

            fields = [
                ("Document Type", meta.get("doc_type")),
                ("Job Number", meta.get("job_number")),
                ("Section", meta.get("section")),
                ("Source", meta.get("source")),
                ("Source Quality", meta.get("source_quality")),
                ("File Path", meta.get("file_path")),
                ("Source ID", meta.get("source_id")),
                ("Ships", meta.get("ships")),
                ("Ship Classes", meta.get("ship_classes")),
                ("Years Mentioned", meta.get("years_mentioned")),
                ("Rates Mentioned", meta.get("rates_mentioned")),
                ("Shipyards Mentioned", meta.get("shipyards_mentioned")),
            ]
            details_lines = []
            file_path_value = None
            for label, value in fields:
                clean_value = self._compact_value(value)
                if clean_value:
                    if label == "File Path":
                        file_path_value = clean_value
                    else:
                        details_lines.append(f"      {label}: {clean_value}")

            excerpt = self._compact_text(getattr(node, "text", "") or "")[:1200]
            if excerpt:
                details_lines.append(f"      Excerpt: {excerpt}")

            details_text = "\n".join(details_lines)
            if details_text:
                self.chat_text.insert("end", details_text + "\n", ("result_detail", detail_tag))
            if file_path_value:
                link_tag = f"path_link_{self._detail_counter}"
                self.chat_text.insert("end", "      File Path: ", ("result_detail", detail_tag))
                self.chat_text.insert("end", f"{file_path_value}\n", ("result_link", detail_tag, link_tag))
                self.chat_text.tag_bind(link_tag, "<Button-1>", lambda _e, p=file_path_value: self._open_local_path(p))
                self.chat_text.tag_bind(link_tag, "<Enter>", lambda _e: self.chat_text.config(cursor="hand2"))
                self.chat_text.tag_bind(link_tag, "<Leave>", lambda _e: self.chat_text.config(cursor="xterm"))
            self.chat_text.insert("end", "\n", ("result_detail", detail_tag))
            self.chat_text.tag_configure(detail_tag, elide=True)

            self.chat_text.tag_bind(toggle_tag, "<Button-1>", lambda _e, tag=detail_tag: self._toggle_details(tag))
            self.chat_text.tag_bind(toggle_tag, "<Enter>", lambda _e: self.chat_text.config(cursor="hand2"))
            self.chat_text.tag_bind(toggle_tag, "<Leave>", lambda _e: self.chat_text.config(cursor="xterm"))

        summary = (
            f"Requested up to {requested}; retriever returned {candidates}; showing {kept}. "
            f"Cut by threshold: {cut} (relative ratio {ratio:.2f}). "
            f"Scored: {scored_count}, missing scores: {missing_score_count}. "
            f"Top: {top_score}, Bottom: {bottom_score}, Threshold: {threshold}."
        )
        self.chat_text.insert("end", summary + "\n\n", ("assistant_msg",))
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _append_used_nodes_results(self, nodes):
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", "MAA Assistant\n", ("speaker_assistant",))
        self.chat_text.insert("end", "Used nodes\n", ("result_header",))

        if not nodes:
            self.chat_text.insert("end", "No used nodes were returned.\n\n", ("assistant_msg",))
            self.chat_text.configure(state="disabled")
            self.chat_text.see("end")
            return

        for i, node in enumerate(nodes, start=1):
            meta = getattr(node, "metadata", {}) or {}
            score = getattr(node, "score", None)
            title = self._compact_text(meta.get("title")) or "Untitled source"
            page = self._compact_text(meta.get("page"))
            row = f"{i}. {title}"
            if page:
                row += f" (p. {page})"
            if isinstance(score, (int, float)):
                row += f" | score: {score:.4f}"

            self.chat_text.insert("end", row + "\n", ("result_row",))

            toggle_tag = f"toggle_details_{self._detail_counter}"
            detail_tag = f"details_block_{self._detail_counter}"
            self._detail_counter += 1

            self.chat_text.insert("end", "   Click to expand details\n", ("result_toggle", toggle_tag))

            fields = [
                ("Document Type", meta.get("doc_type")),
                ("Job Number", meta.get("job_number")),
                ("Section", meta.get("section")),
                ("Source", meta.get("source")),
                ("Source Quality", meta.get("source_quality")),
                ("File Path", meta.get("file_path")),
                ("Source ID", meta.get("source_id")),
                ("Ships", meta.get("ships")),
                ("Ship Classes", meta.get("ship_classes")),
                ("Years Mentioned", meta.get("years_mentioned")),
                ("Rates Mentioned", meta.get("rates_mentioned")),
                ("Shipyards Mentioned", meta.get("shipyards_mentioned")),
            ]
            details_lines = []
            file_path_value = None
            for label, value in fields:
                clean_value = self._compact_value(value)
                if clean_value:
                    if label == "File Path":
                        file_path_value = clean_value
                    else:
                        details_lines.append(f"      {label}: {clean_value}")

            excerpt = self._compact_text(getattr(node, "text", "") or "")[:1200]
            if excerpt:
                details_lines.append(f"      Excerpt: {excerpt}")

            details_text = "\n".join(details_lines)
            if details_text:
                self.chat_text.insert("end", details_text + "\n", ("result_detail", detail_tag))
            if file_path_value:
                link_tag = f"path_link_{self._detail_counter}"
                self.chat_text.insert("end", "      File Path: ", ("result_detail", detail_tag))
                self.chat_text.insert("end", f"{file_path_value}\n", ("result_link", detail_tag, link_tag))
                self.chat_text.tag_bind(link_tag, "<Button-1>", lambda _e, p=file_path_value: self._open_local_path(p))
                self.chat_text.tag_bind(link_tag, "<Enter>", lambda _e: self.chat_text.config(cursor="hand2"))
                self.chat_text.tag_bind(link_tag, "<Leave>", lambda _e: self.chat_text.config(cursor="xterm"))
            self.chat_text.insert("end", "\n", ("result_detail", detail_tag))
            self.chat_text.tag_configure(detail_tag, elide=True)

            self.chat_text.tag_bind(toggle_tag, "<Button-1>", lambda _e, tag=detail_tag: self._toggle_details(tag))
            self.chat_text.tag_bind(toggle_tag, "<Enter>", lambda _e: self.chat_text.config(cursor="hand2"))
            self.chat_text.tag_bind(toggle_tag, "<Leave>", lambda _e: self.chat_text.config(cursor="xterm"))

        self.chat_text.insert("end", "\n", ("assistant_msg",))
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _query_tokens(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower())
        stop = {"the", "and", "for", "with", "from", "that", "this", "have", "what", "about", "docs", "doc"}
        return [t for t in tokens if t not in stop]

    def _metadata_text_blob(self, meta: dict) -> str:
        fields = [
            meta.get("title"),
            meta.get("doc_type"),
            meta.get("job_number"),
            meta.get("source_id"),
            meta.get("ships"),
            meta.get("ship_classes"),
            meta.get("years_mentioned"),
            meta.get("rates_mentioned"),
            meta.get("shipyards_mentioned"),
        ]
        parts = []
        for field in fields:
            if isinstance(field, list):
                parts.extend(str(v) for v in field if str(v).strip())
            elif field is not None and str(field).strip():
                parts.append(str(field))
        return " ".join(parts).lower()

    def _split_reference_entries(self, text: str) -> list[tuple[str, str]]:
        """
        Split text into labeled reference entries like A), B), AL), etc.
        Returns list of (label, body). Falls back to one unlabeled block.
        """
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        # Capture label positions at line starts: A), B), AL), AT), etc.
        label_matches = list(re.finditer(r"(?m)^\s*([A-Z]{1,3}\))\s*", normalized))
        if not label_matches:
            return [("", normalized)]

        entries: list[tuple[str, str]] = []
        for idx, match in enumerate(label_matches):
            label = match.group(1)
            start = match.end()
            end = label_matches[idx + 1].start() if idx + 1 < len(label_matches) else len(normalized)
            body = normalized[start:end].strip()
            if body:
                entries.append((label, body))

        return entries if entries else [("", normalized)]

    def _append_reference_extract_results(self, payload: dict):
        nodes = payload.get("sources", []) or []
        query = payload.get("query", "")
        q_tokens = self._query_tokens(query)
        ratio = payload.get("retrieve_ratio")
        candidates = payload.get("retrieve_candidates_count", 0)
        kept = payload.get("retrieve_kept_count", 0)
        cut = payload.get("retrieve_cut_count", 0)
        requested = payload.get("retrieve_requested_count", candidates)
        scored_count = payload.get("retrieve_scored_count", 0)
        missing_score_count = payload.get("retrieve_missing_score_count", 0)
        top_score = payload.get("retrieve_top_score")
        bottom_score = payload.get("retrieve_bottom_score")
        threshold = payload.get("retrieve_threshold")

        grouped = defaultdict(lambda: defaultdict(list))
        total_entries = 0
        exact_count = 0
        metadata_supported_count = 0
        context_supported_count = 0

        for node in nodes:
            meta = getattr(node, "metadata", {}) or {}
            source_id = self._compact_text(meta.get("source_id")) or "unknown_source"
            title = self._compact_text(meta.get("title")) or source_id
            page = self._compact_text(meta.get("page")) or "unknown_page"
            file_path = self._compact_text(meta.get("file_path"))
            raw_text = (getattr(node, "text", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            entries = self._split_reference_entries(raw_text)
            metadata_blob = self._metadata_text_blob(meta)

            for label, entry_text in entries:
                lower_entry = entry_text.lower()
                token_hits_entry = [tok for tok in q_tokens if tok in lower_entry]
                remaining = [tok for tok in q_tokens if tok not in token_hits_entry]
                token_hits_meta = [tok for tok in remaining if tok in metadata_blob]

                if q_tokens and len(token_hits_entry) == len(q_tokens):
                    match_type = "Exact"
                    exact_count += 1
                elif q_tokens and token_hits_entry and len(token_hits_entry) + len(token_hits_meta) == len(q_tokens):
                    match_type = "Metadata-supported"
                    metadata_supported_count += 1
                else:
                    match_type = "Context-supported"
                    context_supported_count += 1

                total_entries += 1
                grouped[(source_id, title, file_path)][page].append((match_type, label, entry_text))

        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", "MAA Assistant\n", ("speaker_assistant",))
        self.chat_text.insert("end", "Reference extract results\n", ("result_header",))

        if not nodes:
            self.chat_text.insert("end", "No reference entries found.\n\n", ("assistant_msg",))
            self.chat_text.configure(state="disabled")
            self.chat_text.see("end")
            return

        docs_count = len(grouped)
        pages_count = sum(len(page_map) for page_map in grouped.values())
        summary = (
            f"Found {total_entries} entries from {docs_count} documents across {pages_count} pages. "
            f"Exact: {exact_count} | Metadata-supported: {metadata_supported_count} | "
            f"Context-supported: {context_supported_count}."
        )
        self.chat_text.insert("end", summary + "\n\n", ("assistant_msg",))
        retrieval_summary = (
            f"Requested up to {requested}; retriever returned {candidates}; showing {kept}. "
            f"Cut by threshold: {cut} (relative ratio {ratio:.2f}). "
            f"Scored: {scored_count}, missing scores: {missing_score_count}. "
            f"Top: {top_score}, Bottom: {bottom_score}, Threshold: {threshold}."
        )
        self.chat_text.insert("end", retrieval_summary + "\n\n", ("assistant_msg",))

        for (source_id, title, file_path), page_map in grouped.items():
            doc_header = f"Document: {title} ({sum(len(v) for v in page_map.values())} entries)"
            self.chat_text.insert("end", doc_header + "\n", ("result_row",))

            if file_path:
                link_tag = f"path_link_doc_{self._detail_counter}"
                self._detail_counter += 1
                self.chat_text.insert("end", "   File Path: ", ("result_detail",))
                self.chat_text.insert("end", f"{file_path}\n", ("result_link", link_tag))
                self.chat_text.tag_bind(link_tag, "<Button-1>", lambda _e, p=file_path: self._open_local_path(p))
                self.chat_text.tag_bind(link_tag, "<Enter>", lambda _e: self.chat_text.config(cursor="hand2"))
                self.chat_text.tag_bind(link_tag, "<Leave>", lambda _e: self.chat_text.config(cursor="xterm"))

            for page, entries in sorted(page_map.items(), key=lambda item: str(item[0])):
                page_toggle_tag = f"toggle_details_{self._detail_counter}"
                page_detail_tag = f"details_block_{self._detail_counter}"
                self._detail_counter += 1

                page_header = f"   Page {page} ({len(entries)} entries) - Click to expand"
                self.chat_text.insert("end", page_header + "\n", ("result_toggle", page_toggle_tag))

                lines = []
                for idx, (match_type, label, entry_text) in enumerate(entries, start=1):
                    lines.append(f"      {idx}. Match Type: {match_type}")
                    if label:
                        lines.append(f"         Label: {label}")
                    # Keep multiline structure for legal reference readability.
                    formatted_entry = entry_text[:2400]
                    lines.append("         Entry:")
                    for line in formatted_entry.split("\n"):
                        lines.append(f"           {line}")
                page_text = "\n".join(lines) + "\n\n"
                self.chat_text.insert("end", page_text, ("result_detail", page_detail_tag))
                self.chat_text.tag_configure(page_detail_tag, elide=True)
                self.chat_text.tag_bind(page_toggle_tag, "<Button-1>", lambda _e, tag=page_detail_tag: self._toggle_details(tag))
                self.chat_text.tag_bind(page_toggle_tag, "<Enter>", lambda _e: self.chat_text.config(cursor="hand2"))
                self.chat_text.tag_bind(page_toggle_tag, "<Leave>", lambda _e: self.chat_text.config(cursor="xterm"))

            self.chat_text.insert("end", "\n", ("assistant_msg",))

        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _get_metadata_filters(self):
        filters = {}
        for field, var in self.facet_vars.items():
            value = var.get()
            if value and value != "Any":
                filters[field] = value
        return filters

    def _clear_filters(self):
        for var in self.facet_vars.values():
            var.set("Any")

    def _parse_retrieve_only_settings(self) -> tuple[int, float]:
        try:
            max_results = int(self.retrieve_max_results_var.get().strip())
        except ValueError:
            max_results = RETRIEVE_ONLY_DEFAULT_TOP_K
        max_results = max(RETRIEVE_ONLY_MIN_TOP_K, min(RETRIEVE_ONLY_MAX_TOP_K, max_results))

        try:
            ratio = float(self.retrieve_ratio_var.get().strip())
        except ValueError:
            ratio = RETRIEVE_ONLY_DEFAULT_RATIO
        ratio = max(0.0, min(1.0, ratio))
        return max_results, ratio

    def _start_request_timer(self):
        self._request_start_time = time.perf_counter()
        self._update_timer_status()

    def _update_timer_status(self):
        if self._request_start_time is None:
            return
        elapsed = time.perf_counter() - self._request_start_time
        self.status_var.set(f"Working... {elapsed:.2f}s")
        self._timer_after_id = self.after(100, self._update_timer_status)

    def _stop_request_timer(self) -> float | None:
        if self._timer_after_id is not None:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
            self._timer_after_id = None

        if self._request_start_time is None:
            return None

        elapsed = time.perf_counter() - self._request_start_time
        self._request_start_time = None
        return elapsed

    def _on_enter_submit(self, event):
        if self._placeholder_active:
            return "break"
        self._ask_question()
        return "break"

    def _on_shift_enter_newline(self, event):
        if self._placeholder_active:
            self._clear_question_placeholder()
        self.question_entry.insert("insert", "\n")
        return "break"

    def _set_question_placeholder(self, force: bool = False):
        if not force:
            current = self.question_entry.get("1.0", "end").strip()
            if current:
                return
        if self.reference_extract_var.get():
            placeholder = REFERENCE_EXTRACT_PLACEHOLDER
        else:
            placeholder = TEMPLATE_PLACEHOLDERS.get(self.template_var.get(), "Ask a question about the documents...")
        self.question_entry.delete("1.0", "end")
        self.question_entry.configure(fg=self.colors["muted"])
        self.question_entry.insert("1.0", placeholder)
        self._placeholder_active = True

    def _clear_question_placeholder(self):
        if not self._placeholder_active:
            return
        self.question_entry.delete("1.0", "end")
        self.question_entry.configure(fg=self.colors["text"])
        self._placeholder_active = False

    def _on_question_focus_in(self, _event):
        self._clear_question_placeholder()

    def _on_question_focus_out(self, _event):
        current = self.question_entry.get("1.0", "end").strip()
        if not current:
            self._set_question_placeholder(force=True)

    def _ask_question(self):
        if not self.rag:
            messagebox.showinfo("Still loading", "The local RAG backend is still loading.")
            return
        if self._placeholder_active:
            return
        question = self.question_entry.get("1.0", "end").strip()
        if not question:
            self._set_question_placeholder(force=True)
            return
        self.question_entry.delete("1.0", "end")
        self.question_entry.configure(fg=self.colors["text"])
        self._append_user_message(question)
        show_used_nodes = self.show_used_nodes_var.get()
        response_length = self.response_length_var.get().lower()
        template_choice = TEMPLATE_OPTIONS[self.template_var.get()]
        metadata_filters = self._get_metadata_filters()
        references_only = template_choice == "r"
        retrieve_only = self.retrieve_only_var.get()
        reference_extract = self.reference_extract_var.get()
        if reference_extract:
            retrieve_only = True
            references_only = True
            template_choice = "r"
        retrieve_max_results, retrieve_relevance_ratio = self._parse_retrieve_only_settings()
        self.ask_button.configure(state="disabled")
        self._start_request_timer()
        self.status_pill.configure(text="Searching...", fg=self.colors["accent"])
        def run_question():
            try:
                def emit_progress(message: str):
                    self.worker_queue.put(("progress", message))

                result = self.rag.ask(
                    question=question,
                    response_length=response_length,
                    template_choice=template_choice,
                    metadata_filters=metadata_filters,
                    references_only=references_only,
                    show_used_nodes=show_used_nodes,
                    retrieve_only=retrieve_only,
                    retrieve_max_results=retrieve_max_results,
                    retrieve_relevance_ratio=retrieve_relevance_ratio,
                    progress_callback=emit_progress,
                )
                result["reference_extract"] = reference_extract
                result["query"] = question
                self.worker_queue.put(("answer", result))
            except Exception as e:
                self.worker_queue.put(("error", str(e)))
        threading.Thread(target=run_question, daemon=True).start()


if __name__ == "__main__":
    app = MAAApp()
    app.mainloop()
