import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main() -> None:
    """Execute the EDA notebook with this Python environment."""
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "eda.ipynb"
    nb = nbformat.read(path, as_version=4)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=km, timeout=600, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print("Executed notebooks/eda.ipynb and saved reports/eda outputs")


if __name__ == "__main__":
    main()
