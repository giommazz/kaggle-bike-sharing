# utils.py
import matplotlib.pyplot as plt

def savefig_pdf(name, FIGDIR, fig=None):
    """Save current Matplotlib figure or provided `fig` to PDF and close it."""
    if fig is None:
        plt.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")  # vector PDF
        plt.close()
    else:
        fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)