import sys
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Literal

import PySide6.QtCore as Qc
import PySide6.QtGui as Qg
import PySide6.QtWidgets as Qw
from loguru import logger

import handtex.worker_thread as wt
from handtex.data import color_themes, custom_icons
from handtex.error_dialog_driver import ErrorDialog
from handtex.utils import resource_path


# For all show functions, pad the dialog message, so that the dialog is not too narrow for the window title.
MIN_MSG_LENGTH = 50


class SelectableMessageBox(Qw.QMessageBox):
    """
    Subclass the QMessageBox to make the text selectable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(SelectableMessageBox, self).__init__(*args, **kwargs)
        self.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)

        labels = self.findChildren(Qw.QLabel)
        for label in labels:
            label.setTextInteractionFlags(Qc.Qt.TextSelectableByMouse)


def show_exception(
    parent,
    title: str,
    msg: str,
    error_bundle: None | wt.WorkerError | tuple[type, BaseException, TracebackType] = None,
    collect_exception: bool = True,
) -> None:
    """
    Show an exception in a dialog along with logs.
    This automatically gathers the exception information from the current context
    or a given worker error object.

    You can also skip collecting the exception if you have already logged this separately and
    merely wish to open the Issue Reporter dialog for the user.

    :param parent: The parent widget.
    :param title: The title of the dialog.
    :param msg: The message to show.
    :param error_bundle: [Optional] A worker error or a tuple of exception information.
    :param collect_exception: [Optional] Whether to add exception information to the log.
    """

    if collect_exception:
        exception_type: type[BaseException]
        exception_value: BaseException
        exception_traceback: TracebackType

        if error_bundle is not None:
            if isinstance(error_bundle, tuple):
                exception_type, exception_value, exception_traceback = error_bundle
            elif isinstance(error_bundle, wt.WorkerError):
                exception_type = error_bundle.exception_type
                exception_value = error_bundle.value
                exception_traceback = error_bundle.traceback
            else:
                logger.error(f"Invalid error bundle: {error_bundle}")
                exception_type, exception_value, exception_traceback = sys.exc_info()
        else:
            exception_type, exception_value, exception_traceback = sys.exc_info()

        # Ignore the exception if it's a KeyboardInterrupt.
        if exception_type is KeyboardInterrupt:
            logger.warning("User interrupted the process.")
            return

        logger.opt(
            depth=1, exception=(exception_type, exception_value, exception_traceback)
        ).critical(msg)

    box = ErrorDialog(parent, title, msg)
    box.exec()


def show_critical(parent, title: str, msg: str, **kwargs) -> int:
    msg = msg.ljust(MIN_MSG_LENGTH)
    box = SelectableMessageBox(
        Qw.QMessageBox.Critical,
        title,
        msg,
        Qw.QMessageBox.Yes | Qw.QMessageBox.Abort,
        parent,
        **kwargs,
    )
    return box.exec()


def show_warning(parent, title: str, msg: str, **kwargs) -> None:
    msg = msg.ljust(MIN_MSG_LENGTH)
    box = SelectableMessageBox(
        Qw.QMessageBox.Warning, title, msg, Qw.QMessageBox.Ok, parent, **kwargs
    )
    box.exec()


def show_info(parent, title: str, msg: str, **kwargs) -> None:
    msg = msg.ljust(MIN_MSG_LENGTH)
    box = SelectableMessageBox(
        Qw.QMessageBox.Information, title, msg, Qw.QMessageBox.Ok, parent, **kwargs
    )
    box.exec()


def show_question(
    parent, title: str, msg: str, buttons=Qw.QMessageBox.Yes | Qw.QMessageBox.Cancel
) -> int:
    # Note: Yes uses dialog-ok-apply, Cancel uses dialog-cancel.
    msg = msg.ljust(MIN_MSG_LENGTH)
    dlg = Qw.QMessageBox(parent)
    dlg.setWindowTitle(title)
    dlg.setText(msg)
    dlg.setStandardButtons(buttons)
    dlg.setIcon(Qw.QMessageBox.Question)
    return dlg.exec()


def open_file(path: Path) -> None:
    """
    Open any given file with the default application.
    """
    logger.debug(f"Opening file {path}")
    try:
        # Use Qt to open the file, so that it works on all platforms.
        Qg.QDesktopServices.openUrl(Qc.QUrl.fromLocalFile(str(path)))
    except Exception:
        show_exception(None, "File Error", "Failed to open file.")


# Mapping between KDE color keys and QPalette roles
section_role_mapping = {
    "Colors:Button": {
        "BackgroundNormal": Qg.QPalette.Button,
        "ForegroundNormal": Qg.QPalette.ButtonText,
        "ForegroundActive": Qg.QPalette.BrightText,
    },
    "Colors:View": {
        "BackgroundNormal": Qg.QPalette.Base,
        "BackgroundAlternate": Qg.QPalette.AlternateBase,
        "ForegroundNormal": Qg.QPalette.Text,
        "ForegroundLink": Qg.QPalette.Link,
        "ForegroundVisited": Qg.QPalette.LinkVisited,
        "ForegroundInactive": Qg.QPalette.PlaceholderText,
    },
    "Colors:Selection": {
        "BackgroundNormal": Qg.QPalette.Highlight,
        "ForegroundNormal": Qg.QPalette.HighlightedText,
    },
    "Colors:Tooltip": {
        "BackgroundNormal": Qg.QPalette.ToolTipBase,
        "ForegroundNormal": Qg.QPalette.ToolTipText,
    },
    "Colors:Window": {
        "BackgroundNormal": Qg.QPalette.Window,
        "ForegroundNormal": Qg.QPalette.WindowText,
    },
    # Add more mappings as needed
}


def clamp_8bit(value: int) -> int:
    """
    Clamp a value to the 0-255 range.
    """
    return max(0, min(value, 255))


def apply_color_effect(
    source: Qg.QColor, effect_base: Qg.QColor, contrast_amount: float
) -> Qg.QColor:
    """
    Apply a color effect to a source color.

    :param source: The source color.
    :param effect_base: The base color for the effect.
    :param contrast_amount: The amount of contrast to apply.
    :return: The modified color.
    """
    # Essentially alpha blend, with the effect having the contrast amount as the alpha pasted on top.
    r = clamp_8bit(int(source.red() * (1 - contrast_amount) + effect_base.red() * contrast_amount))
    g = clamp_8bit(
        int(source.green() * (1 - contrast_amount) + effect_base.green() * contrast_amount)
    )
    b = clamp_8bit(
        int(source.blue() * (1 - contrast_amount) + effect_base.blue() * contrast_amount)
    )
    return Qg.QColor(r, g, b)


def load_color_palette(theme: str) -> Qg.QPalette:
    """
    Provide a theme name and get a QPalette object.
    The name should match one of the files in the themes folder.

    :param theme: The name of the theme.
    :return: A QPalette object.
    """
    palette = Qg.QPalette()

    file_path = resource_path(color_themes, theme)

    file = Qc.QFile(file_path)
    if file.open(Qc.QFile.ReadOnly | Qc.QFile.Text):
        stream = Qc.QTextStream(file)
        content = stream.readAll()

        # Find the disabled color parameters.
        disabled_color = Qg.QColor.fromRgb(128, 128, 128)  # Default to gray.
        disabled_contrast_amount = 0.0
        in_disabled_section = False
        for line in content.split("\n"):
            line = line.strip()
            if line == "[ColorEffects:Disabled]":
                in_disabled_section = True
                continue
            if not in_disabled_section:
                continue
            if line.startswith("["):
                in_disabled_section = False
                break
            if "=" not in line:
                continue

            # Ok, now we're in the disabled section.
            key, value = map(str.strip, line.split("=", 1))
            if key == "Color":
                r, g, b = map(int, value.split(","))
                disabled_color.setRgb(r, g, b)
            elif key == "ContrastAmount":
                disabled_contrast_amount = float(value)

        section = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
            elif "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                role_mapping = section_role_mapping.get(section, {})
                qt_color_role = role_mapping.get(key, None)
                if qt_color_role is not None:
                    r, g, b = map(int, value.split(","))
                    palette.setColor(Qg.QPalette.Normal, qt_color_role, Qg.QColor(r, g, b))
                    palette.setColor(Qg.QPalette.Inactive, qt_color_role, Qg.QColor(r, g, b))
                    # Calculate the disabled color.
                    disabled_color = apply_color_effect(
                        Qg.QColor(r, g, b), disabled_color, disabled_contrast_amount
                    )
                    palette.setColor(Qg.QPalette.Disabled, qt_color_role, disabled_color)
    else:
        raise ValueError(f"Could not open theme file: {theme}")

    # Fallback calculations
    if not palette.color(Qg.QPalette.Light).isValid():
        base = palette.color(Qg.QPalette.Button)
        palette.setColor(Qg.QPalette.Light, base.lighter(150))

    if not palette.color(Qg.QPalette.Dark).isValid():
        base = palette.color(Qg.QPalette.Window)
        palette.setColor(Qg.QPalette.Dark, base.darker(150))

    if not palette.color(Qg.QPalette.Midlight).isValid():
        light = palette.color(Qg.QPalette.Light)
        button = palette.color(Qg.QPalette.Button)
        midlight = Qg.QColor(
            (light.red() + button.red()) // 2,
            (light.green() + button.green()) // 2,
            (light.blue() + button.blue()) // 2,
        )
        palette.setColor(Qg.QPalette.Midlight, midlight)

    if not palette.color(Qg.QPalette.Mid).isValid():
        dark = palette.color(Qg.QPalette.Dark)
        button = palette.color(Qg.QPalette.Button)
        mid = Qg.QColor(
            (dark.red() + button.red()) // 2,
            (dark.green() + button.green()) // 2,
            (dark.blue() + button.blue()) // 2,
        )
        palette.setColor(Qg.QPalette.Mid, mid)

    if not palette.color(Qg.QPalette.Shadow).isValid():
        palette.setColor(Qg.QPalette.Shadow, Qg.QColor(0, 0, 0))

    return palette


def custom_icon_path(icon_name: str, theme: Literal["dark", "light"] | str = "") -> Path:
    """
    Loads the given icon from the dark, light, or color-agnostic set of custom icons.
    File names may omit the extension, in which case .svg and .png are checked.

    :param icon_name: The icon's filename, with or without extension.
    :param theme: Indicate if the icon should be pulled from the light or dark theme,
        if applicable, otherwise leave blank.
    :return: A full path to the file.
    """
    custom_icon_dir: Path = resource_path(custom_icons)

    if theme:
        custom_icon_dir = custom_icon_dir / theme

    for extension in ("", ".svg", ".png"):
        icon_path = custom_icon_dir / (icon_name + extension)
        if icon_path.is_file():
            return icon_path

    raise FileNotFoundError(f"Failed to load '{custom_icon_dir / icon_name}'")


def load_custom_icon(icon_name: str, theme: Literal["dark", "light"] | str = "") -> Qg.QIcon:
    """
    Loads the given icon from the dark, light, or color-agnostic set of custom icons.
    File names may omit the extension, in which case .svg and .png are checked.
    If the file could not be found, a QIcon with a null pixmap is returned.

    :param icon_name: The icon's filename, with or without extension.
    :param theme: Indicate if the icon should be pulled from the light or dark theme,
        if applicable, otherwise leave blank.
    :return: A QIcon that may have a null pixmap.
    """
    try:
        icon_path = custom_icon_path(icon_name, theme)
        return Qg.QIcon(str(icon_path))
    except FileNotFoundError as e:
        logger.error(e)
        return Qg.QIcon()
