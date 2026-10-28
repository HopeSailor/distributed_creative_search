"""Final manuscript figure typography and file export."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.text import Text


def style_context(*args, **kwargs):
    return plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": 8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
            "axes.unicode_minus": True,
            "figure.dpi": 180,
        }
    )


def save(fig, output, number):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for text in fig.findobj(match=Text):
        value = text.get_text()
        if value in list("abcdABCD"):
            text.set_text("(" + value.lower() + ")")
        elif value.startswith(("c: ", "d: ")):
            text.set_text("(" + value[0] + ") " + value[3:])
        text.set_color("#808080" if number == 2 and value == "No" else "#000000")
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(fig.canvas.get_renderer())
        if (
            box.width
            and box.height
            and (
                box.x0 < -1
                or box.y0 < -1
                or box.x1 > fig.bbox.width + 1
                or box.y1 > fig.bbox.height + 1
            )
        ):
            raise ValueError("Text outside figure: " + text.get_text())
    for ext in ["pdf", "svg"]:
        fig.savefig(output / f"figure_{number}.{ext}", facecolor="white", bbox_inches=None)
    fig.savefig(output / f"figure_{number}.png", dpi=300, facecolor="white", bbox_inches=None)
    plt.close(fig)
