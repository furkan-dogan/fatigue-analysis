"""Volleyball entry point; measurement workflows are implemented in later stages."""
from ui.components.planned_analysis import planned_analysis


def render() -> None:
    planned_analysis('Voleybol — Video Analizi',
                     'Sıçrama, yana sapma / iniş asimetrisi ve sprint incelemeleri burada yer alacak.')
