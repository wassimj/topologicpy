# Copyright (C) 2024
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
    def ByValueInRange(value: float = 0.5, minValue: float = 0.0, maxValue: float = 1.0, alpha: float = 1.0, useAlpha: bool = False, colorScale="viridis"):
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
            The color expressed as an [r,g,b] or an [r,g,b,a] list.

        """
        
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
        if useAlpha:
            rgbList.append(alpha)
        return rgbList
    
    @staticmethod
    def HEXToRGB(hex):
        """
        Converts a hexadecimal color string to RGB color values.

        Parameters
        ----------
        hex : str
            A hexadecimal color string in the format '#RRGGBB'.

        Returns
        -------
        tuple
            A tuple containing three integers representing the RGB values.

        """
        if not isinstance(hex, str):
            print("Color.HEXtoRGB - Error: The input hex parameter is not a valid string. Returning None.")
            return None
        hex = hex.lstrip('#')
        if len(hex) != 6:
            print("Color.HEXtoRGB - Error: Invalid hexadecimal color format. It should be a 6-digit hex value. Returning None.")
            return None
        r = int(hex[0:2], 16)
        g = int(hex[2:4], 16)
        b = int(hex[4:6], 16)
        return [r, g, b]

    @staticmethod
    def PlotlyColor(color, alpha=1.0, useAlpha=False):
        """
        Returns a plotly color string based on the input list of [r,g,b] or [r,g,b,a]. If your list is [r,g,b], you can optionally specify an alpha value

        Parameters
        ----------
        color : list
            The input color list. This is assumed to be in the format [r,g,b] or [r,g,b,a]
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
        hex_value = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return hex_value.upper()
