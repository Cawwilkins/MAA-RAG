"""
Tkinter launcher for MAA Assistant.

Run:
    python app.py
"""

from tk_app import MAAApp


def main() -> None:
    app = MAAApp()
    app.mainloop()


if __name__ == "__main__":
    main()
