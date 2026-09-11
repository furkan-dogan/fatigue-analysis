"""Application shell; sport navigation is the next planned stage."""

from ui.theme import configure_page


def main() -> None:
    configure_page()
    from ui.sports.taekwondo.page import render

    render()
