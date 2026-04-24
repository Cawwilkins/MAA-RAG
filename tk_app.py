import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from config import METADATA_FACETS_PATH
from rag_service import RAGService


TEMPLATE_OPTIONS = {
    "Q/A": "q",
    "Summary": "s",
    "Exposure Analysis": "e",
    "Timeline": "t",
    "Reference Evidence": "r",
    "Comparison": "c",
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

        style.configure("TCheckbutton", background=self.colors["panel"], foreground=self.colors["text"], focuscolor=self.colors["panel"], font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", self.colors["panel"])], foreground=[("active", self.colors["text"])])
        style.configure("TSeparator", background=self.colors["border"])

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(22, 20))
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.configure(width=310)
        self.sidebar.grid_propagate(False)
        self.sidebar.columnconfigure(0, weight=1)

        ttk.Label(self.sidebar, text="MAA Assistant", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.sidebar, text="Local document intelligence", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 22))

        ttk.Label(self.sidebar, text="Search Settings", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 10))
        ttk.Label(self.sidebar, text="Response length", style="Muted.TLabel").grid(row=3, column=0, sticky="w")
        self.response_length_var = tk.StringVar(value="medium")
        self.response_length_combo = ttk.Combobox(self.sidebar, textvariable=self.response_length_var, values=["short", "medium", "long"], state="readonly")
        self.response_length_combo.grid(row=4, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(self.sidebar, text="Template", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        self.template_var = tk.StringVar(value="Q/A")
        self.template_combo = ttk.Combobox(self.sidebar, textvariable=self.template_var, values=list(TEMPLATE_OPTIONS.keys()), state="readonly")
        self.template_combo.grid(row=6, column=0, sticky="ew", pady=(4, 14))

        self.references_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Search references section only", variable=self.references_only_var).grid(row=7, column=0, sticky="w", pady=(0, 8))

        self.show_used_nodes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.sidebar, text="Show used nodes", variable=self.show_used_nodes_var).grid(row=8, column=0, sticky="w", pady=(0, 16))

        ttk.Separator(self.sidebar).grid(row=9, column=0, sticky="ew", pady=(4, 18))
        ttk.Label(self.sidebar, text="Metadata Filters", style="Section.TLabel").grid(row=10, column=0, sticky="w", pady=(0, 10))

        self.filter_canvas = tk.Canvas(self.sidebar, bg=self.colors["panel"], highlightthickness=0, height=370)
        self.filter_canvas.grid(row=11, column=0, sticky="nsew")
        self.sidebar.rowconfigure(11, weight=1)

        self.filter_scrollbar = ttk.Scrollbar(self.sidebar, orient="vertical", command=self.filter_canvas.yview)
        self.filter_scrollbar.grid(row=11, column=1, sticky="ns")

        self.filters_frame = ttk.Frame(self.filter_canvas, style="Sidebar.TFrame")
        self.filters_frame.columnconfigure(0, weight=1)
        self.filter_window = self.filter_canvas.create_window((0, 0), window=self.filters_frame, anchor="nw")
        self.filter_canvas.configure(yscrollcommand=self.filter_scrollbar.set)
        self.filters_frame.bind("<Configure>", self._on_filters_configure)
        self.filter_canvas.bind("<Configure>", self._on_canvas_configure)

        self.clear_filters_button = ttk.Button(self.sidebar, text="Clear Filters", style="Secondary.TButton", command=self._clear_filters)
        self.clear_filters_button.grid(row=12, column=0, sticky="ew", pady=(16, 0))

        self.main = ttk.Frame(self, style="Main.TFrame", padding=(22, 20))
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

        self.input_card = tk.Frame(self.main, bg=self.colors["input"], highlightbackground=self.colors["border"], highlightthickness=1)
        self.input_card.grid(row=2, column=0, sticky="ew", pady=(14, 8))
        self.input_card.columnconfigure(0, weight=1)

        self.question_entry = tk.Text(self.input_card, height=4, wrap="word", font=("Segoe UI", 11), bg=self.colors["input"], fg=self.colors["text"], insertbackground=self.colors["text"], borderwidth=0, highlightthickness=0, padx=14, pady=12)
        self.question_entry.grid(row=0, column=0, sticky="ew")
        self.question_entry.bind("<Control-Return>", lambda event: self._ask_question())

        self.ask_button = ttk.Button(self.input_card, text="Ask", style="Modern.TButton", command=self._ask_question, state="disabled")
        self.ask_button.grid(row=0, column=1, sticky="ns", padx=(8, 10), pady=10)

        self.status_var = tk.StringVar(value="Loading local RAG system...")
        self.status_label = ttk.Label(self.main, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=3, column=0, sticky="w")

        self._append_assistant_message("MAA Assistant is starting locally. Once loading finishes, ask a question below.")

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
                    self.status_var.set("Ready. Running locally.")
                    self.status_pill.configure(text="Ready", fg=self.colors["success"], bg=self.colors["card"])
                    self._append_assistant_message("Ready. The local document assistant is loaded.")
                elif event == "answer":
                    if payload.get("used_nodes_text"):
                        self._append_assistant_message(payload["used_nodes_text"])
                    self._append_assistant_message(payload["answer"])
                    self.ask_button.configure(state="normal")
                    self.status_var.set("Ready.")
                    self.status_pill.configure(text="Ready", fg=self.colors["success"])
                elif event == "error":
                    self.ask_button.configure(state="normal" if self.rag else "disabled")
                    self.status_var.set("Error.")
                    self.status_pill.configure(text="Error", fg=self.colors["danger"])
                    messagebox.showerror("MAA Assistant Error", payload)
                    self._append_assistant_message(f"Error: {payload}")
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

    def _get_metadata_filters(self):
        filters = {}
        for field, var in self.facet_vars.items():
            value = var.get()
            if value and value != "Any":
                filters[field] = value
        if self.references_only_var.get():
            filters["section"] = "references"
        return filters

    def _clear_filters(self):
        for var in self.facet_vars.values():
            var.set("Any")
        self.references_only_var.set(False)

    def _ask_question(self):
        if not self.rag:
            messagebox.showinfo("Still loading", "The local RAG backend is still loading.")
            return
        question = self.question_entry.get("1.0", "end").strip()
        if not question:
            return
        self.question_entry.delete("1.0", "end")
        self._append_user_message(question)
        show_used_nodes = self.show_used_nodes_var.get()
        response_length = self.response_length_var.get()
        template_choice = TEMPLATE_OPTIONS[self.template_var.get()]
        metadata_filters = self._get_metadata_filters()
        references_only = self.references_only_var.get()
        self.ask_button.configure(state="disabled")
        self.status_var.set("Searching documents...")
        self.status_pill.configure(text="Searching...", fg=self.colors["accent"])
        def run_question():
            try:
                result = self.rag.ask(
                    question=question,
                    response_length=response_length,
                    template_choice=template_choice,
                    metadata_filters=metadata_filters,
                    references_only=references_only,
                    show_used_nodes=show_used_nodes,
                )
                self.worker_queue.put(("answer", result))
            except Exception as e:
                self.worker_queue.put(("error", str(e)))
        threading.Thread(target=run_question, daemon=True).start()


if __name__ == "__main__":
    app = MAAApp()
    app.mainloop()
