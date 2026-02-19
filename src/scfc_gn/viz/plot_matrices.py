import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

def show_as_image(img, title, save=True, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(output_dir / f"{timestamp}_{title}.png")
    plt.show()