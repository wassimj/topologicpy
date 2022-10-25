
class Color:
    @staticmethod
    def ByObjectColor(bObject):
        """
        Parameters
        ----------
        bObject : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        color = bObject.color
        return (color[0],color[1], color[2], color[3])
    
    @staticmethod
    def ByValueInRange(value, minValue, maxValue, alpha, useAlpha):
        """
        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        minValue : TYPE
            DESCRIPTION.
        maxValue : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        useAlpha : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # value = item[0]
        # minValue = item[1]
        # maxValue = item[2]
        # alpha = item[3]
        # useAlpha = item[4]
        
        def getColor(ratio):
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
        
        color = None
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
        rgbList = getColor(val)
        if useAlpha:
            rgbList.append(alpha)
        return tuple(rgbList)
