"""Volleyball-only MVP entry point; other sport modules remain isolated."""
from ui.theme import configure_page
from ui.sports.volleyball.page import render


def main() -> None:
    configure_page()
    render()
