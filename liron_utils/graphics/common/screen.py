import typing


def get_pixel_color(modal: bool = True) -> None:
    """Print the RGB and hex color of a pixel selected via mouse click.

    Starts mouse and keyboard listeners; the next click reports the pixel under
    the cursor, while pressing Esc cancels the operation.

    Args:
        modal: If True, block the calling thread until a click or Esc is received.
    """
    from threading import Event  # pylint: disable=import-outside-toplevel

    import pyautogui  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel,import-error
    from pynput import (  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel,import-error
        keyboard,
        mouse,
    )

    done = Event()  # Event to block until user clicks or presses Esc

    def on_click(x: int, y: int, button: typing.Any, pressed: bool) -> None:  # pylint: disable=unused-argument
        if pressed:
            rgb = pyautogui.screenshot().getpixel((x, y))
            rgb_norm = tuple(v / 255 for v in rgb)
            hex_str = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            print(f"(x, y)=({x}, {y}) | RGB=({rgb_norm[0]:.3f}, {rgb_norm[1]:.3f}, {rgb_norm[2]:.3f}) | HEX={hex_str}")
            done.set()
            listener_mouse.stop()
            listener_keyboard.stop()

    def on_press(key: typing.Any) -> None:
        if key == keyboard.Key.esc:
            print("Operation cancelled by user.")
            done.set()
            listener_mouse.stop()
            listener_keyboard.stop()

    print("Click to pick a pixel color or press Esc to cancel...")

    listener_mouse = mouse.Listener(on_click=on_click)
    listener_keyboard = keyboard.Listener(on_press=on_press)
    listener_mouse.start()
    listener_keyboard.start()

    if modal:
        done.wait()  # Block until the event is set
