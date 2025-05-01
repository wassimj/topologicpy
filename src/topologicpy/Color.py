# Copyright (C) 2025
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import plotly.colors
import math

class Color:

    @staticmethod
    def AnyToHex(color):
        """
        Converts a color to a hexadecimal color string.

        Parameters
        ----------
        color : list or str
            The input color parameter which can be any of RGB, CMYK, CSS Named Color, or Hex

        Returns
        -------
        str
            A hexadecimal color string in the format '#RRGGBB'.
        """
        return_hex = None
        if isinstance(color, list):
            if len(color) == 4: # Probably CMYK
                if all(0 <= x <= 1 for x in color[:4]):
                    return_hex = Color.CMYKToHex(color[:4])
            elif len(color) == 3:
                if all(0 <= x <= 255 for x in color[:3]):
                    return_hex = Color.RGBToHex(color[:3])
        elif isinstance(color, str): # Probably a CSSColor
            if color.lower() in [x.lower() for x in Color.CSSNamedColors()]:
                rgb = Color.ByCSSNamedColor(color.lower())
                return_hex = Color.RGBToHex(rgb)
            else: # Probably alread a HEX or other Plotly-compatible string
                return_hex = color

        if not isinstance(return_hex, str):
            print("Color.AnyToHex - Error: Could not recognize the input parameter. Returning None.")
            return None

        return return_hex.upper()
    
    
    @staticmethod
    def ByCSSNamedColor(color, alpha: float = None):
        """
        Creates a Color from a CSS named color string. See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Parameters
        ----------
        color : str
            A CSS named color.
        alpha : float , optional
            THe desired alpha (transparency value). The default is None which means no alpha value will be included in the returned list.

        Returns
        -------
        list
            The color expressed as an [r, g, b] or an [r, g, b, a] list.
        """
        import warnings
        import os
        try:
            import webcolors
        except:
            print("Color.ByCSSNamedColor - Information: Installing required webcolors library.")
            try:
                os.system("pip install webcolors")
            except:
                os.system("pip install webcolors --user")
            try:
                import webcolors
                print("Color.ByCSSNamedColor - Information: webcolors library installed correctly.")
            except:
                warnings.warn("Color.ByCSSNamedColor - Error: Could not import webcolors library. Please manually install webcolors. Returning None.")
                return None

        if not alpha == None:
            if not 0.0 <= alpha <= 1.0:
                print("Color.ByCSSNamedColor - Error: alpha is not within the valid range of 0 to 1. Returning None.")
                return None
        try:
            # Get RGB values from the named CSS color
            rgbList = list(webcolors.name_to_rgb(color))
            if not alpha == None:
                rgbList.append(alpha)
            return rgbList

        except ValueError:
            print(f"Color.ByCSSNamedColor - Error: '{color}' is not a valid named CSS color. Returning None.")
            return None
    
    @staticmethod
    def ByHEX(hex: str, alpha: float = None):
        """
        Converts a hexadecimal color string to RGB color values.

        Parameters
        ----------
        hex : str
            A hexadecimal color string in the format '#RRGGBB'.
        alpha : float , optional
            The transparency value. 0.0 means the color is fully transparent, 1.0 means the color is fully opaque. The default is None
            which means no transparency value will be included in the returned color.
        Returns
        -------
        list
            The color expressed as an [r, g, b] or an [r, g, b, a] list.

        """
        if not isinstance(hex, str):
            print("Color.HEXtoRGB - Error: The input hex parameter is not a valid string. Returning None.")
            return None
        if not alpha == None:
            if not 0.0 <= alpha <= 1.0:
                print("Color.ByHEX - Error: alpha is not within the valid range of 0 to 1. Returning None.")
                return None
        hex = hex.lstrip('#')
        if len(hex) != 6:
            print("Color.HEXtoRGB - Error: Invalid hexadecimal color format. It should be a 6-digit hex value. Returning None.")
            return None
        r = int(hex[0:2], 16)
        g = int(hex[2:4], 16)
        b = int(hex[4:6], 16)
        rgbList = [r, g, b]
        if not alpha == None:
            rgbList.append(alpha)
        return rgbList

    @staticmethod
    def ByValueInRange(value: float = 0.5, minValue: float = 0.0, maxValue: float = 1.0, alpha: float = None, colorScale="viridis"):
        """
        Returns the r, g, b, (and optionally) a list of numbers representing the red, green, blue and alpha color elements.
        
        Parameters
        ----------
        value : float , optional
            The input value. The default is 0.5.
        minValue : float , optional
            the input minimum value. The default is 0.0.
        maxValue : float , optional
            The input maximum value. The default is 1.0.
        alpha : float , optional
            The alpha (transparency) value. 0.0 means the color is fully transparent, 1.0 means the color is fully opaque. The default is 1.0.
        useAlpha : bool , optional
            If set to True, the returns list includes the alpha value as a fourth element in the list.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.

        Returns
        -------
        list
            The color expressed as an [r, g, b] or an [r, g, b, a] list.

        """
        if not alpha == None:
            if not 0.0 <= alpha <= 1.0:
                print("Color.ByValueInRange - Error: alpha is not within the valid range of 0 to 1. Returning None.")
                return None
        # Code based on: https://stackoverflow.com/questions/62710057/access-color-from-plotly-color-scale

        def hex_to_rgb(value):
            value = str(value)
            value = value.lstrip('#')
            lv = len(value)
            returnValue = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            return str(returnValue)

        def get_color(colorscale_name, loc):
            from _plotly_utils.basevalidators import ColorscaleValidator
            # first parameter: Name of the property being validated
            # second parameter: a string, doesn't really matter in our use case
            cv = ColorscaleValidator("colorscale", "")
            # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
            colorscale = cv.validate_coerce(colorscale_name)
            if hasattr(loc, "__iter__"):
                return [get_continuous_color(colorscale, x) for x in loc]
            color = get_continuous_color(colorscale, loc)
            color = color.replace("rgb", "")
            color = color.replace("(", "")
            color = color.replace(")", "")
            color = color.split(",")
            final_colors = []
            for c in color:
                final_colors.append(math.floor(float(c)))
            return final_colors

        def get_continuous_color(colorscale, intermed):
            """
            Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
            color for any value in that range.

            Plotly doesn't make the colorscales directly accessible in a common format.
            Some are ready to use:
            
                colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

            Others are just swatches that need to be constructed into a colorscale:

                viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
                colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

            :param colorscale: A plotly continuous colorscale defined with RGB string colors.
            :param intermed: value in the range [0, 1]
            :return: color in rgb string format
            :rtype: str
            """

            if len(colorscale) < 1:
                raise ValueError("colorscale must have at least one color")
            if intermed <= 0 or len(colorscale) == 1:
                c = colorscale[0][1]
                return c if c[0] != "#" else hex_to_rgb(c)
            if intermed >= 1:
                c = colorscale[-1][1]
                return c if c[0] != "#" else hex_to_rgb(c)
            for cutoff, color in colorscale:
                if intermed > cutoff:
                    low_cutoff, low_color = cutoff, color
                else:
                    high_cutoff, high_color = cutoff, color
                    break
            if (low_color[0] == "#") or (high_color[0] == "#"):
                # some color scale names (such as cividis) returns:
                # [[loc1, "hex1"], [loc2, "hex2"], ...]
                low_color = hex_to_rgb(low_color)
                high_color = hex_to_rgb(high_color)
            return plotly.colors.find_intermediate_color(
                lowcolor=low_color,
                highcolor=high_color,
                intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
                colortype="rgb",
            )
        
        def get_color_default(ratio):
            r = 0.0
            g = 0.0
            b = 0.0

            finalRatio = ratio;
            if (finalRatio < 0.0):
                finalRatio = 0.0
            elif(finalRatio > 1.0):
                finalRatio = 1.0

            if (finalRatio >= 0.0 and finalRatio <= 0.25):
                r = 0.0
                g = 4.0 * finalRatio
                b = 1.0
            elif (finalRatio > 0.25 and finalRatio <= 0.5):
                r = 0.0
                g = 1.0
                b = 1.0 - 4.0 * (finalRatio - 0.25)
            elif (finalRatio > 0.5 and finalRatio <= 0.75):
                r = 4.0*(finalRatio - 0.5);
                g = 1.0
                b = 0.0
            else:
                r = 1.0
                g = 1.0 - 4.0 * (finalRatio - 0.75)
                b = 0.0

            rcom =  (max(min(r, 1.0), 0.0))
            gcom =  (max(min(g, 1.0), 0.0))
            bcom =  (max(min(b, 1.0), 0.0))

            return [rcom,gcom,bcom]
        
        if minValue > maxValue:
            temp = minValue;
            maxValue = minValue
            maxValue = temp

        val = value
        val = max(min(val,maxValue), minValue) # bracket value to the min and max values
        if (maxValue - minValue) != 0:
            val = (val - minValue)/(maxValue - minValue)
        else:
            val = 0
        if not colorScale or colorScale.lower() == "default":
            rgbList = get_color_default(val)
        else:
            rgbList = get_color(colorScale, val)
        if not alpha == None:
            rgbList.append(alpha)
        return rgbList
    
    @staticmethod
    def CMYKToHex(cmyk):
        """
        Convert a CMYK color (list of 4 values) to its hexadecimal representation.
        
        Parameters
        ----------
        color : list
            cmyk (list or tuple): CMYK color values as [C, M, Y, K], each in the range 0 to 1.

        Returns
        -------
        str: The hexadecimal color string for Plotly (e.g., '#FFFFFF').
        """
        c, m, y, k = cmyk
        
        # Convert CMYK to RGB
        r = 255 * (1 - c) * (1 - k)
        g = 255 * (1 - m) * (1 - k)
        b = 255 * (1 - y) * (1 - k)
        
        # Clamp RGB values to 0-255 range and convert to integers
        r, g, b = int(round(r)), int(round(g)), int(round(b))
        
        # Convert RGB to hex format
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return hex_color
    
    @staticmethod
    def CSSNamedColor(color):
        """
        Returns the CSS Named color that most closely matches the input color. The input color is assumed to be
        in the format [r, g, b]. See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Parameters
        ----------
        color : list
            The input color. This is assumed to be in the format [r, g, b]
        
        Returns
        -------
        str
            The CSS named color that most closely matches the input color.
        """
        import numbers
        import warnings
        import os
        try:
            import webcolors
        except:
            print("Color.CSSNamedColor - Information: Installing required webcolors library.")
            try:
                os.system("pip install webcolors")
            except:
                os.system("pip install webcolors --user")
            try:
                import webcolors
                print("Color.CSSNamedColor - Information: webcolors library installed correctly.")
            except:
                warnings.warn("Color.CSSNamedColor - Error: Could not import webcolors library. Please manually install webcolors. Returning None.")
                return None

        if not isinstance(color, list):
            print("Color.CSSNamedColor - Error: The input color parameter is not a valid list. Returning None.")
            return None
        color = [int(x) for x in color if isinstance(x, numbers.Real)]
        if len(color) < 3:
            print("Color.CSSNamedColor - Error: The input color parameter does not contain valid r, g, b values. Returning None.")
            return None
        color = color[0:3]
        for x in color:
            if not (0 <= x <= 255):
                print("Color.CSSNamedColor - Error: The input color parameter does not contain valid r, g, b values. Returning None.")
                return None

        def est_color(requested_color):
            min_colors = {}
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - requested_color[0]) ** 2
                gd = (g_c - requested_color[1]) ** 2
                bd = (b_c - requested_color[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]

        try:
            closest_color_name = webcolors.rgb_to_name(color)
        except ValueError:
            closest_color_name = est_color(color)
        return closest_color_name

    @staticmethod
    def CSSNamedColors():
        """
        Returns a list of all CSS named colors. See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Parameters
        ----------

        Returns
        -------
        list
            The list of all CSS named colors.

        """
        # List of CSS named colors
        css_named_colors = [
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

        return css_named_colors

    @staticmethod
    def PlotlyColor(color, alpha=1.0, useAlpha=False):
        """
        Returns a plotly color string based on the input list of [r, g, b] or [r, g, b, a]. If your list is [r, g, b], you can optionally specify an alpha value

        Parameters
        ----------
        color : list
            The input color list. This is assumed to be in the format [r, g, b] or [r, g, b, a] where the range is from 0 to 255.
        alpha : float , optional
            The transparency value. 0.0 means the color is fully transparent, 1.0 means the color is fully opaque. The default is 1.0.
        useAlpha : bool , optional
            If set to True, the returns list includes the alpha value as a fourth element in the list.

        Returns
        -------
        str
            The plotly color string.

        """
        if not isinstance(color, list):
            print("Color.PlotlyColor - Error: The input color parameter is not a valid list. Returning None.")
            return None
        if len(color) < 3:
            print("Color.PlotlyColor - Error: The input color parameter contains less than the minimum three elements. Returning None.")
            return None
        if len(color) == 4:
            alpha = color[3]
        alpha = min(max(alpha, 0), 1)
        if alpha < 1:
            useAlpha = True
        if useAlpha:
            return "rgba("+str(color[0])+","+str(color[1])+","+str(color[2])+","+str(alpha)+")"
        return "rgb("+str(color[0])+","+str(color[1])+","+str(color[2])+")"
    
    @staticmethod
    def RGBToHex(rgb):
        """
        Converts RGB color values to a hexadecimal color string.

        Parameters
        ----------
        rgb : tuple
            A tuple containing three integers representing the RGB values.

        Returns
        -------
        str
            A hexadecimal color string in the format '#RRGGBB'.
        """
        if not isinstance(rgb, list):
            print("Color.RGBToHex - Error: The input rgb parameter is not a valid list. Returning None.")
            return None
        r, g, b = rgb
        r = int(r)
        g = int(g)
        b = int(b)
        hex_value = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return hex_value.upper()
    
    
