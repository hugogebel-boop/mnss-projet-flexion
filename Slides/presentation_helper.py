"""Stub de Slides.presentation_helper — anim Jupyter, sans dépendance wand/IM."""
from IPython.display import HTML


def display_animation(anim):
    """Renvoie l'animation matplotlib en HTML (JS) ; pas de conversion gif via wand."""
    try:
        return HTML(anim.to_jshtml())
    except Exception:
        return anim


def md(text):
    from IPython.display import Markdown
    return Markdown(text)
