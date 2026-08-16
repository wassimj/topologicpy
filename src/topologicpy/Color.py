# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import math
import numbers
import os
import re
import warnings

import plotly.colors


class Color:
    @staticmethod
    def _flatten(items):
        """Internal: flattens nested lists/tuples."""
        result = []
        for item in items:
            if isinstance(item, (list, tuple)):
                result.extend(Color._flatten(item))
            else:
                result.append(item)
        return result

    @staticmethod
    def _is_real(value) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))

    @staticmethod
    def _valid_alpha(alpha) -> bool:
        return alpha is None or (Color._is_real(alpha) and 0.0 <= float(alpha) <= 1.0)

    @staticmethod
    def _valid_hex(value: str) -> bool:
        if not isinstance(value, str):
            return False
        value = value.strip().lstrip("#")
        return len(value) == 6 and re.fullmatch(r"[0-9a-fA-F]{6}", value) is not None

    @staticmethod
    def _normalize_hex(value: str):
        if not Color._valid_hex(value):
            return None
        return "#" + value.strip().lstrip("#").upper()

    @staticmethod
    def _rgb_channels(rgb):
        if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
            return None
        channels = []
        for value in rgb[:3]:
            if not Color._is_real(value):
                return None
            value = int(round(float(value)))
            if not 0 <= value <= 255:
                return None
            channels.append(value)
        return channels

    @staticmethod
    def _cmyk_channels(cmyk):
        if not isinstance(cmyk, (list, tuple)) or len(cmyk) != 4:
            return None
        channels = []
        for value in cmyk:
            if not Color._is_real(value):
                return None
            value = float(value)
            if not 0.0 <= value <= 1.0:
                return None
            channels.append(value)
        return channels

    @staticmethod
    def _import_webcolors(caller_name: str):
        try:
            import webcolors
            return webcolors
        except Exception:
            print(f"Color.{caller_name} - Information: Installing required webcolors library.")
            try:
                os.system("pip install webcolors")
            except Exception:
                os.system("pip install webcolors --user")
            try:
                import webcolors
                print(f"Color.{caller_name} - Information: webcolors library installed correctly.")
                return webcolors
            except Exception:
                warnings.warn(
                    f"Color.{caller_name} - Error: Could not import webcolors library. Please manually install webcolors. Returning None."
                )
                return None

    @staticmethod
    def _css_color_names(webcolors):
        try:
            return list(webcolors.names("css3"))
        except Exception:
            pass
        try:
            return list(webcolors.CSS3_NAMES_TO_HEX.keys())
        except Exception:
            pass
        try:
            return list(webcolors.CSS3_HEX_TO_NAMES.values())
        except Exception:
            pass
        return Color.CSSNamedColors()

    @staticmethod
    def AddHex(*colors, silent: bool = False):
        """
        Adds the input hexadecimal color codes channel-wise, clipping each channel to a max of 255.

        Parameters
        ----------
        colors : *list or str
            The input list of hexadecimal colors.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The resulting hex color after addition (e.g. '#FF88FF').
        """
        color_list = Color._flatten(list(colors))
        color_list = [Color.AnyToHex(c) for c in color_list]
        color_list = [c for c in color_list if Color._valid_hex(c)]

        if len(color_list) == 0:
            if not silent:
                print("Color.AddHex - Error: The input colors parameter does not contain any valid colors. Returning None.")
            return None

        r = g = b = 0
        for hex_color in color_list:
            hex_color = hex_color.lstrip("#")
            r = min(r + int(hex_color[0:2], 16), 255)
            g = min(g + int(hex_color[2:4], 16), 255)
            b = min(b + int(hex_color[4:6], 16), 255)

        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def AnyToHex(
        color,
        silent: bool = False,
        colorModel: str = "auto",
    ):
        """
        Converts a color to a hexadecimal color string.

        Parameters
        ----------
        color : list, tuple, or str
            The input color. Supported formats include:

            - RGB lists/tuples: [255, 0, 0], [1.0, 0.0, 0.0]
            - RGBA lists/tuples: [255, 0, 0, 0.5], [1.0, 0.0, 0.0, 0.5]
            - CMYK lists/tuples: [0.0, 1.0, 1.0, 0.0]
            - HEX strings: "#RGB", "#RGBA", "#RRGGBB", "#RRGGBBAA"
            - CSS rgb()/rgba(): "rgb(255,0,0)", "rgba(0,0,0,0.5)"
            - CSS hsl()/hsla(): "hsl(0,100%,50%)", "hsla(0,100%,50%,0.5)"
            - CSS named colors: "red", "transparent"
            - A list of color strings, in which case their average color is returned.

            Alpha values are validated but ignored because the return format is
            '#RRGGBB'.

        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is
            False.
        colorModel : str , optional
            The color model used for numeric list or tuple inputs. Valid values are
            "auto", "rgb", "rgba", and "cmyk". Default is "auto".

            In "auto" mode, three-channel numeric inputs are interpreted as RGB.
            Four-channel numeric inputs are interpreted as RGBA only when at least
            one RGB channel is greater than 1 and the fourth channel is between 0
            and 1. A four-channel input whose values are all between 0 and 1 is
            ambiguous and requires an explicit "rgba" or "cmyk" colorModel.

        Returns
        -------
        str
            A hexadecimal color string in the format '#RRGGBB', or None if the
            input cannot be interpreted unambiguously.

        """

        import colorsys
        import inspect
        import re

        valid_models = {"auto", "rgb", "rgba", "cmyk"}

        if not isinstance(colorModel, str):
            if not silent:
                print(
                    "Color.AnyToHex - Error: The input colorModel parameter is "
                    "not a valid string. Returning None."
                )
            return None

        color_model = colorModel.strip().lower()

        if color_model not in valid_models:
            if not silent:
                print(
                    "Color.AnyToHex - Error: The input colorModel parameter must "
                    'be one of "auto", "rgb", "rgba", or "cmyk". Returning None.'
                )
            return None

        def _fail(message=None):
            if not silent:
                if message:
                    print(f"Color.AnyToHex - Error: {message} Returning None.")
                else:
                    print(
                        "Color.AnyToHex - Error: Could not recognize the input "
                        "parameter. Returning None."
                    )

                print("Input Color:", color)

                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])

            return None

        def _clamp_int(value, minimum=0, maximum=255):
            try:
                value = int(round(float(value)))
                return max(minimum, min(maximum, value))
            except Exception:
                return None

        def _hex_from_rgb_channels(channels):
            if channels is None or len(channels) < 3:
                return None

            r = _clamp_int(channels[0])
            g = _clamp_int(channels[1])
            b = _clamp_int(channels[2])

            if r is None or g is None or b is None:
                return None

            return f"#{r:02X}{g:02X}{b:02X}"

        def _numeric_sequence(seq):
            if not isinstance(seq, (list, tuple)):
                return None

            if not all(Color._is_real(value) for value in seq):
                return None

            return [float(value) for value in seq]

        def _rgb_to_hex(values):
            if len(values) != 3:
                return None

            if not all(0.0 <= value <= 255.0 for value in values):
                return None

            if all(0.0 <= value <= 1.0 for value in values):
                values = [value * 255.0 for value in values]

            return _hex_from_rgb_channels(values)

        def _rgba_to_hex(values):
            if len(values) != 4:
                return None

            rgb = values[:3]
            alpha = values[3]

            if not 0.0 <= alpha <= 1.0:
                return None

            if not all(0.0 <= value <= 255.0 for value in rgb):
                return None

            if all(0.0 <= value <= 1.0 for value in rgb):
                rgb = [value * 255.0 for value in rgb]

            return _hex_from_rgb_channels(rgb)

        def _cmyk_to_hex(values):
            if len(values) != 4:
                return None

            if not all(0.0 <= value <= 1.0 for value in values):
                return None

            return Color.CMYKToHex(values, silent=True)

        def _sequence_to_hex(seq):
            numeric = _numeric_sequence(seq)

            if numeric is None:
                return None, None

            if color_model == "rgb":
                return _rgb_to_hex(numeric), None

            if color_model == "rgba":
                return _rgba_to_hex(numeric), None

            if color_model == "cmyk":
                return _cmyk_to_hex(numeric), None

            # Automatic interpretation.
            if len(numeric) == 3:
                return _rgb_to_hex(numeric), None

            if len(numeric) != 4:
                return None, None

            # All four values in 0..1 are valid as both normalized RGBA and CMYK.
            if all(0.0 <= value <= 1.0 for value in numeric):
                return (
                    None,
                    "The four-channel input is ambiguous. Specify "
                    'colorModel="rgba" or colorModel="cmyk".',
                )

            # An unambiguous RGBA input has an alpha value in 0..1 and at least one
            # RGB channel greater than 1.
            if (
                0.0 <= numeric[3] <= 1.0
                and all(0.0 <= value <= 255.0 for value in numeric[:3])
            ):
                return _rgba_to_hex(numeric), None

            return None, None

        def _parse_rgb_component(token):
            token = str(token).strip()

            if token.endswith("%"):
                try:
                    return _clamp_int(float(token[:-1]) * 2.55)
                except Exception:
                    return None

            try:
                value = float(token)
            except Exception:
                return None

            return _clamp_int(value)

        def _parse_alpha_component(token):
            token = str(token).strip()

            if token.endswith("%"):
                try:
                    value = float(token[:-1]) / 100.0
                except Exception:
                    return None
            else:
                try:
                    value = float(token)
                except Exception:
                    return None

            if not 0.0 <= value <= 1.0:
                return None

            return value

        def _split_css_function_args(arg_string):
            arg_string = arg_string.strip().replace("/", " / ")

            if "," in arg_string:
                return [
                    part.strip()
                    for part in arg_string.split(",")
                    if part.strip()
                ]

            return [
                part.strip()
                for part in arg_string.split()
                if part.strip() and part.strip() != "/"
            ]

        def _parse_css_rgb(value):
            match = re.fullmatch(
                r"rgba?\((.*)\)",
                value,
                flags=re.IGNORECASE,
            )

            if not match:
                return None

            parts = _split_css_function_args(match.group(1))

            if len(parts) not in (3, 4):
                return None

            r = _parse_rgb_component(parts[0])
            g = _parse_rgb_component(parts[1])
            b = _parse_rgb_component(parts[2])

            if r is None or g is None or b is None:
                return None

            if len(parts) == 4:
                alpha = _parse_alpha_component(parts[3])
                if alpha is None:
                    return None

            return _hex_from_rgb_channels([r, g, b])

        def _parse_hsl_component(token, is_hue=False):
            token = str(token).strip()

            if is_hue:
                lower = token.lower()

                try:
                    if lower.endswith("deg"):
                        return float(lower[:-3]) % 360.0

                    if lower.endswith("rad"):
                        return (
                            float(lower[:-3])
                            * 180.0
                            / 3.141592653589793
                        ) % 360.0

                    if lower.endswith("turn"):
                        return (float(lower[:-4]) * 360.0) % 360.0

                    return float(lower) % 360.0

                except Exception:
                    return None

            if not token.endswith("%"):
                return None

            try:
                value = float(token[:-1]) / 100.0
            except Exception:
                return None

            if not 0.0 <= value <= 1.0:
                return None

            return value

        def _parse_css_hsl(value):
            match = re.fullmatch(
                r"hsla?\((.*)\)",
                value,
                flags=re.IGNORECASE,
            )

            if not match:
                return None

            parts = _split_css_function_args(match.group(1))

            if len(parts) not in (3, 4):
                return None

            h = _parse_hsl_component(parts[0], is_hue=True)
            s = _parse_hsl_component(parts[1])
            lightness = _parse_hsl_component(parts[2])

            if h is None or s is None or lightness is None:
                return None

            if len(parts) == 4:
                alpha = _parse_alpha_component(parts[3])
                if alpha is None:
                    return None

            r, g, b = colorsys.hls_to_rgb(
                h / 360.0,
                lightness,
                s,
            )

            return _hex_from_rgb_channels([
                r * 255.0,
                g * 255.0,
                b * 255.0,
            ])

        def _normalize_hex_local(value):
            value = value.strip()

            if not value.startswith("#"):
                return None

            hex_value = value[1:].strip()

            if not re.fullmatch(r"[0-9a-fA-F]+", hex_value):
                return None

            if len(hex_value) == 3:
                return "#" + "".join(
                    channel * 2 for channel in hex_value
                ).upper()

            if len(hex_value) == 4:
                return "#" + "".join(
                    channel * 2 for channel in hex_value[:3]
                ).upper()

            if len(hex_value) == 6:
                return "#" + hex_value.upper()

            if len(hex_value) == 8:
                return "#" + hex_value[:6].upper()

            return None

        if isinstance(color, list) and all(
            isinstance(item, str) for item in color
        ):
            if color_model != "auto":
                return _fail(
                    "colorModel applies only to numeric list or tuple inputs."
                )

            return Color.Average(color, silent=silent)

        if isinstance(color, (list, tuple)):
            result, error = _sequence_to_hex(color)

            if error:
                return _fail(error)

            if result is None:
                return _fail()

            return result.upper()

        if isinstance(color, str):
            if color_model != "auto":
                return _fail(
                    "colorModel applies only to numeric list or tuple inputs."
                )

            value = color.strip()
            lower = value.lower()

            if lower == "transparent":
                return "#000000"

            normalized = None

            try:
                normalized = Color._normalize_hex(value)
            except Exception:
                normalized = None

            if normalized is None:
                normalized = _normalize_hex_local(value)

            if normalized is not None:
                return normalized.upper()

            result = _parse_css_rgb(value)

            if result is not None:
                return result.upper()

            result = _parse_css_hsl(value)

            if result is not None:
                return result.upper()

            try:
                named_colors = Color.CSSNamedColors()
                named_lookup = {
                    str(name).lower(): name
                    for name in named_colors
                }

                if lower in named_lookup:
                    rgb = Color.ByCSSNamedColor(
                        lower,
                        silent=silent,
                    )

                    if rgb:
                        return Color.RGBToHex(
                            rgb,
                            silent=silent,
                        ).upper()

            except Exception:
                pass

        return _fail()

    @staticmethod
    def Average(*colors, silent: bool = False):
        """
        Averages the input list of colors after conversion to HEX.

        Parameters
        ----------
        colors : *list or str
            The input color parameter which can be any of RGB, CMYK, CSS named color, or HEX.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            A hexadecimal color string in the format '#RRGGBB'.
        """
        color_list = Color._flatten(list(colors))
        color_list = [Color.AnyToHex(c, silent=True) for c in color_list]
        color_list = [c for c in color_list if Color._valid_hex(c)]

        if len(color_list) == 0:
            if not silent:
                print("Color.Average - Error: The input colors parameter does not contain any valid colors. Returning None.")
            return None

        r_total = g_total = b_total = 0
        for hex_color in color_list:
            hex_color = hex_color.lstrip("#")
            r_total += int(hex_color[0:2], 16)
            g_total += int(hex_color[2:4], 16)
            b_total += int(hex_color[4:6], 16)

        n = len(color_list)
        r = int(round(r_total / n))
        g = int(round(g_total / n))
        b = int(round(b_total / n))
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def ByCSSNamedColor(color, alpha: float = None, silent: bool = False):
        """
        Creates a Color from a CSS named color string. See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Parameters
        ----------
        color : str
            A CSS named color.
        alpha : float , optional
            The desired alpha (transparency value). Default is None which means no alpha value will be included in the returned list.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The color expressed as an [r, g, b] or an [r, g, b, a] list.
        """
        webcolors = Color._import_webcolors("ByCSSNamedColor")
        if webcolors is None:
            return None

        if not isinstance(color, str):
            if not silent:
                print("Color.ByCSSNamedColor - Error: The input color parameter is not a valid string. Returning None.")
            return None

        if not Color._valid_alpha(alpha):
            if not silent:
                print("Color.ByCSSNamedColor - Error: alpha is not within the valid range of 0 to 1. Returning None.")
            return None

        try:
            rgb_list = list(webcolors.name_to_rgb(color.strip().lower()))
            if alpha is not None:
                rgb_list.append(float(alpha))
            return rgb_list
        except Exception:
            if not silent:
                print(f"Color.ByCSSNamedColor - Error: '{color}' is not a valid named CSS color. Returning None.")
            return None

    @staticmethod
    def ByHEX(hex: str, alpha: float = None, silent: bool = False):
        """
        Converts a hexadecimal color string to RGB color values.

        Parameters
        ----------
        hex : str
            A hexadecimal color string in the format '#RRGGBB'.
        alpha : float , optional
            The transparency value. 0.0 means the color is fully transparent, 1.0 means the color is fully opaque. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The color expressed as an [r, g, b] or an [r, g, b, a] list.
        """
        if not isinstance(hex, str):
            if not silent:
                print("Color.ByHEX - Error: The input hex parameter is not a valid string. Returning None.")
            return None

        if not Color._valid_alpha(alpha):
            if not silent:
                print("Color.ByHEX - Error: alpha is not within the valid range of 0 to 1. Returning None.")
            return None

        normalized = Color._normalize_hex(hex)
        if normalized is None:
            if not silent:
                print("Color.ByHEX - Error: Invalid hexadecimal color format. It should be a 6-digit hex value. Returning None.")
            return None

        value = normalized.lstrip("#")
        rgb_list = [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)]
        if alpha is not None:
            rgb_list.append(float(alpha))
        return rgb_list

    @staticmethod
    def ByValueInRange(value: float = 0.5,
                       minValue: float = 0.0,
                       maxValue: float = 1.0,
                       alpha: float = None,
                       colorScale="viridis",
                       silent: bool = False):
        """
        Returns the r, g, b, and optionally a, values corresponding to a value in a range.

        Parameters
        ----------
        value : float , optional
            The input value. Default is 0.5.
        minValue : float , optional
            The input minimum value. Default is 0.0.
        maxValue : float , optional
            The input maximum value. Default is 1.0.
        alpha : float , optional
            The alpha value. Default is None.
        colorScale : str , optional
            The Plotly color scale name, or 'default'. Default is 'viridis'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The color expressed as an [r, g, b] or an [r, g, b, a] list.
        """
        if not Color._is_real(value):
            if not silent:
                print("Color.ByValueInRange - Error: value is not a valid number. Returning None.")
            return None
        if not Color._is_real(minValue):
            if not silent:
                print("Color.ByValueInRange - Error: minValue is not a valid number. Returning None.")
            return None
        if not Color._is_real(maxValue):
            if not silent:
                print("Color.ByValueInRange - Error: maxValue is not a valid number. Returning None.")
            return None
        if not Color._valid_alpha(alpha):
            if not silent:
                print("Color.ByValueInRange - Error: alpha is not within the valid range of 0 to 1. Returning None.")
            return None

        value = float(value)
        minValue = float(minValue)
        maxValue = float(maxValue)
        if minValue > maxValue:
            minValue, maxValue = maxValue, minValue

        value = max(min(value, maxValue), minValue)
        ratio = (value - minValue) / (maxValue - minValue) if maxValue != minValue else 0.0

        def _default_color(ratio_value):
            ratio_value = max(0.0, min(float(ratio_value), 1.0))
            if 0.0 <= ratio_value <= 0.25:
                r, g, b = 0.0, 4.0 * ratio_value, 1.0
            elif ratio_value <= 0.5:
                r, g, b = 0.0, 1.0, 1.0 - 4.0 * (ratio_value - 0.25)
            elif ratio_value <= 0.75:
                r, g, b = 4.0 * (ratio_value - 0.5), 1.0, 0.0
            else:
                r, g, b = 1.0, 1.0 - 4.0 * (ratio_value - 0.75), 0.0
            return [int(round(255 * max(min(r, 1.0), 0.0))),
                    int(round(255 * max(min(g, 1.0), 0.0))),
                    int(round(255 * max(min(b, 1.0), 0.0)))]

        def _hex_to_rgb_string(value):
            rgb = Color.ByHEX(value, silent=True)
            return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        def _continuous_color(colorscale, intermed):
            if len(colorscale) < 1:
                raise ValueError("colorscale must have at least one color")
            if intermed <= 0 or len(colorscale) == 1:
                c = colorscale[0][1]
                return c if not str(c).startswith("#") else _hex_to_rgb_string(c)
            if intermed >= 1:
                c = colorscale[-1][1]
                return c if not str(c).startswith("#") else _hex_to_rgb_string(c)

            low_cutoff, low_color = colorscale[0]
            high_cutoff, high_color = colorscale[-1]
            for cutoff, color in colorscale:
                if intermed > cutoff:
                    low_cutoff, low_color = cutoff, color
                else:
                    high_cutoff, high_color = cutoff, color
                    break

            if str(low_color).startswith("#"):
                low_color = _hex_to_rgb_string(low_color)
            if str(high_color).startswith("#"):
                high_color = _hex_to_rgb_string(high_color)

            local_t = (intermed - low_cutoff) / (high_cutoff - low_cutoff) if high_cutoff != low_cutoff else 0.0
            return plotly.colors.find_intermediate_color(
                lowcolor=low_color,
                highcolor=high_color,
                intermed=local_t,
                colortype="rgb",
            )

        if not colorScale or str(colorScale).lower() == "default":
            rgb_list = _default_color(ratio)
        else:
            try:
                from _plotly_utils.basevalidators import ColorscaleValidator
                cv = ColorscaleValidator("colorscale", "")
                colorscale = cv.validate_coerce(colorScale)
                color_string = _continuous_color(colorscale, ratio)
                color_string = color_string.replace("rgba", "").replace("rgb", "")
                color_string = color_string.replace("(", "").replace(")", "")
                parts = [p.strip() for p in color_string.split(",")]
                rgb_list = [int(math.floor(float(parts[i]))) for i in range(3)]
                rgb_list = [max(0, min(255, c)) for c in rgb_list]
            except Exception:
                if not silent:
                    print("Color.ByValueInRange - Error: Could not process the input colorScale. Returning None.")
                return None

        if alpha is not None:
            rgb_list.append(float(alpha))
        return rgb_list

    @staticmethod
    def CMYKToHex(cmyk, silent: bool = False):
        """
        Converts a CMYK color to its hexadecimal representation.

        Parameters
        ----------
        cmyk : list or tuple
            CMYK values as [C, M, Y, K], each in the range 0 to 1.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The hexadecimal color string for Plotly (e.g., '#FFFFFF').
        """
        channels = Color._cmyk_channels(cmyk)
        if channels is None:
            if not silent:
                print("Color.CMYKToHex - Error: The input cmyk parameter is not a valid list of four values between 0 and 1. Returning None.")
            return None

        c, m, y, k = channels
        r = int(round(255 * (1 - c) * (1 - k)))
        g = int(round(255 * (1 - m) * (1 - k)))
        b = int(round(255 * (1 - y) * (1 - k)))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def CSSNamedColor(color, silent: bool = False):
        """
        Returns the CSS named color that most closely matches the input RGB color.

        Parameters
        ----------
        color : list
            The input color in the format [r, g, b].
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The CSS named color that most closely matches the input color.
        """
        webcolors = Color._import_webcolors("CSSNamedColor")
        if webcolors is None:
            return None

        rgb = Color._rgb_channels(color)
        if rgb is None:
            if not silent:
                print("Color.CSSNamedColor - Error: The input color parameter does not contain valid r, g, b values. Returning None.")
            return None

        rgb_tuple = tuple(rgb)
        try:
            return webcolors.rgb_to_name(rgb_tuple)
        except Exception:
            pass

        min_distance = float("inf")
        closest_name = None
        for name in Color._css_color_names(webcolors):
            try:
                candidate = webcolors.name_to_rgb(name)
                candidate_tuple = (candidate.red, candidate.green, candidate.blue)
            except Exception:
                continue
            distance = sum((candidate_tuple[i] - rgb_tuple[i]) ** 2 for i in range(3))
            if distance < min_distance:
                min_distance = distance
                closest_name = name

        return closest_name

    @staticmethod
    def CSSNamedColors():
        """
        Returns a list of all CSS named colors. See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Returns
        -------
        list
            The list of all CSS named colors.
        """
        return [
            "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond",
            "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue",
            "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkgrey",
            "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon",
            "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink",
            "deepskyblue", "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia",
            "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green", "greenyellow", "grey", "honeydew", "hotpink",
            "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
            "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey", "lightpink",
            "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue",
            "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue",
            "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise",
            "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace",
            "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
            "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple",
            "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna",
            "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan",
            "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"
        ]

    @staticmethod
    def PlotlyColor(color, alpha=1.0, useAlpha=False, silent: bool = False):
        """
        Returns a Plotly color string based on an RGB or RGBA list.

        Parameters
        ----------
        color : list
            The input color list in the format [r, g, b] or [r, g, b, a].
        alpha : float , optional
            The transparency value. Default is 1.0.
        useAlpha : bool , optional
            If set to True, the returned string uses rgba(...). Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The Plotly color string.
        """
        if not isinstance(color, list):
            if not silent:
                print("Color.PlotlyColor - Error: The input color parameter is not a valid list. Returning None.")
            return None

        rgb = Color._rgb_channels(color)
        if rgb is None:
            if not silent:
                print("Color.PlotlyColor - Error: The input color parameter does not contain valid r, g, b values. Returning None.")
            return None

        if len(color) >= 4:
            alpha = color[3]

        if not Color._valid_alpha(alpha):
            if not silent:
                print("Color.PlotlyColor - Error: alpha is not within the valid range of 0 to 1. Returning None.")
            return None

        alpha = float(alpha)
        if alpha < 1:
            useAlpha = True

        if useAlpha:
            return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    @staticmethod
    def RGBToHex(rgb, silent: bool = False):
        """
        Converts RGB color values to a hexadecimal color string.

        Parameters
        ----------
        rgb : list or tuple
            Three integer-like RGB values in the range 0 to 255.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            A hexadecimal color string in the format '#RRGGBB'.
        """
        channels = Color._rgb_channels(rgb)
        if channels is None:
            if not silent:
                print("Color.RGBToHex - Error: The input rgb parameter is not a valid list of three values between 0 and 255. Returning None.")
            return None
        r, g, b = channels
        return f"#{r:02X}{g:02X}{b:02X}"
