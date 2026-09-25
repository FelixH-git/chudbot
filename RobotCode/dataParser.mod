MODULE dataParser

    ! 1. The RECORD is defined here. Because it is in this module,
    ! every other module in the robot automatically knows what a ShapeJob is.
    RECORD ShapeJob
        string job_Shape;
        num x_val;
        num y_val;
        num rz_val;
    ENDRECORD

    ! The parsing function. It takes the raw string, fills out the record,
    ! and returns TRUE if full, FALSE if string invalid.
    FUNC bool ParseJobString(string raw_message, INOUT ShapeJob parsed_job)
        VAR string x_str;
        VAR string y_str;
        VAR string angle_str;
        VAR num idx1;
        VAR num idx2;
        VAR num idx3;
        VAR bool ok := FALSE;

        ! Find 1st comma
        idx1 := StrFind(raw_message, 1, ",");
        IF idx1 > 0 THEN
            parsed_job.job_shape := StrPart(raw_message, 1, idx1 - 1);

            ! Find 2nd comma
            idx2 := StrFind(raw_message, idx1 + 1, ",");
            IF idx2 > 0 THEN
                x_str := StrPart(raw_message, idx1 + 1, idx2 - (idx1 + 1));

                ! Find 3rd comma
                idx3 := StrFind(raw_message, idx2 + 1, ",");
                IF idx3 > 0 THEN
                    y_str := StrPart(raw_message, idx2 + 1, idx3 - (idx2 + 1));
                    angle_str := StrPart(raw_message, idx3 + 1, StrLen(raw_message) - idx3);

                    ! Convert strings to numbers
                    ok := StrToVal(x_str, parsed_job.x_val);
                    ok := ok AND StrToVal(y_str, parsed_job.y_val);
                    ok := ok AND StrToVal(angle_str, parsed_job.rz_val);

                    IF ok THEN
                        RETURN TRUE; ! Parsing was 100% successful
                    ENDIF
                ENDIF
            ENDIF
        ENDIF

        RETURN FALSE; ! Something failed
    ENDFUNC

ENDMODULE
