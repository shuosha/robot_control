from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def identify_ROIs(
    path: str,
    obj_list: List[str]
) -> Dict[str, List[Tuple[int,int]]]:
    """
    Let the user pick 4 corners for each object in obj_list via matplotlib clicks.
    Returns a dict mapping each object name to its 4 corner coords.
    """
    # Load once
    img = mpimg.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    rois: Dict[str, List[Tuple[int,int]]] = {}

    for obj in obj_list:
        corners: List[Tuple[int,int]] = []
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.set_title(f"Click 4 corners for '{obj}'\n(Press right‑click or any key to finish when done)")
        plt.axis('off')

        def onclick(event):
            # Only respond to left clicks inside the axes
            if event.inaxes is not ax or event.button != 1:
                return
            x, y = int(event.xdata), int(event.ydata)
            corners.append((x, y))
            ax.plot(x, y, 'ro')
            fig.canvas.draw()
            print(f"[{obj}] Picked corner #{len(corners)}: ({x}, {y})")
            # Auto‑disconnect if we've got 4
            if len(corners) == 4:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)

        # Connect and show
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        if len(corners) != 4:
            raise RuntimeError(f"Object '{obj}': expected 4 corners, got {len(corners)}")

        rois[obj] = corners
        print(f"[{obj}] Final corners: {corners}")

    return rois

def main():
    identify_ROIs(
        path="logs/teleop/1209_nut_thread_processed/episode_0000/camera_0/rgb/000000.jpg",
        obj_list=["fixed_asset", "held_asset"]
    )

if __name__ == "__main__":
    main()