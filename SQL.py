class SQL:
    @staticmethod
    def SQLQuery(model, EPReportName, EPReportForString, EPTableName, EPColumnName, EPRowName, EPUnits):
        """
        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        EPReportName : TYPE
            DESCRIPTION.
        EPReportForString : TYPE
            DESCRIPTION.
        EPTableName : TYPE
            DESCRIPTION.
        EPColumnName : TYPE
            DESCRIPTION.
        EPRowName : TYPE
            DESCRIPTION.
        EPUnits : TYPE
            DESCRIPTION.

        Returns
        -------
        doubleValue : TYPE
            DESCRIPTION.

        """
        # model = item[0]
        # EPReportName = item[1]
        # EPReportForString = item[2]
        # EPTableName = item[3]
        # EPColumnName = item[4]
        # EPRowName = item[5]
        # EPUnits = item[6]
        
        def doubleValueFromQuery(sqlFile, EPReportName, EPReportForString, EPTableName, EPColumnName, EPRowName, EPUnits):
            doubleValue = 0.0
            query = "SELECT Value FROM tabulardatawithstrings WHERE ReportName='" + EPReportName + "' AND ReportForString='" + EPReportForString + "' AND TableName = '" + EPTableName + "' AND RowName = '" + EPRowName + "' AND ColumnName= '" + EPColumnName + "' AND Units='" + EPUnits + "'";
            osOptionalDoubleValue = sqlFile.execAndReturnFirstDouble(query)
            if (osOptionalDoubleValue.is_initialized()):
                doubleValue = osOptionalDoubleValue.get()
            else:
                raise Exception("Failed to get a double value from the SQL file.")
            return doubleValue
        
        sqlFile = model.sqlFile().get()
        doubleValue = doubleValueFromQuery(sqlFile, EPReportName, EPReportForString, EPTableName, EPColumnName, EPRowName, EPUnits)
        return doubleValue