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
    ! and returns TRUE if valid, FALSE if string invalid.
    FUNC bool ParseJobString(string raw_message, INOUT ShapeJob parsed_job)
        VAR string x_str;
        VAR string y_str;
        VAR string angle_str;
        VAR num idx1;
        VAR num idx2;
        VAR num idx3;
        VAR num s1;
        VAR num s2;
        VAR num s3;
        VAR num s4;
        VAR num cr_idx;
        VAR bool ok := FALSE;

        ! Exit clause check
        IF raw_message = "exit" OR raw_message = "EXIT" THEN
            parsed_job.job_Shape := "exit";
            parsed_job.x_val := 0;
            parsed_job.y_val := 0;
            parsed_job.rz_val := 0;
            RETURN TRUE;
        ENDIF

        ! 1. Comma-separated format: shape,x,y,angle (or shape,x,y)
        idx1 := StrFind(raw_message, 1, ",");
        IF idx1 > 0 THEN
            parsed_job.job_Shape := StrPart(raw_message, 1, idx1 - 1);
            
            idx2 := StrFind(raw_message, idx1 + 1, ",");
            IF idx2 > 0 THEN
                x_str := StrPart(raw_message, idx1 + 1, idx2 - (idx1 + 1));
                
                idx3 := StrFind(raw_message, idx2 + 1, ",");
                IF idx3 > 0 THEN
                    y_str := StrPart(raw_message, idx2 + 1, idx3 - (idx2 + 1));
                    angle_str := StrPart(raw_message, idx3 + 1, StrLen(raw_message) - idx3);
                ELSE
                    ! Optional 3rd field as Y, angle defaults to 0
                    y_str := StrPart(raw_message, idx2 + 1, StrLen(raw_message) - idx2);
                    angle_str := "0";
                ENDIF
                
                ! Strip any trailing CR / LF
                cr_idx := StrFind(angle_str, 1, "\0D");
                IF cr_idx > 0 THEN
                    angle_str := StrPart(angle_str, 1, cr_idx - 1);
                ENDIF
                cr_idx := StrFind(angle_str, 1, "\0A");
                IF cr_idx > 0 THEN
                    angle_str := StrPart(angle_str, 1, cr_idx - 1);
                ENDIF

                ok := StrToVal(x_str, parsed_job.x_val);
                ok := ok AND StrToVal(y_str, parsed_job.y_val);
                ok := ok AND StrToVal(angle_str, parsed_job.rz_val);
                
                IF ok THEN
                    RETURN TRUE;
                ENDIF
            ENDIF
        ENDIF

        ! 2. Fallback: Semicolon format: SHAPE=...;X=...;Y=...;RZ=...
        s1 := StrFind(raw_message, 1, ";");
        IF s1 > 0 THEN
            IF StrFind(raw_message, 1, "SHAPE=") = 1 THEN
                parsed_job.job_Shape := StrPart(raw_message, 7, s1 - 7);
            ELSE
                parsed_job.job_Shape := StrPart(raw_message, 1, s1 - 1);
            ENDIF

            s2 := StrFind(raw_message, s1 + 1, ";");
            IF s2 > 0 THEN
                x_str := StrPart(raw_message, s1 + 1, s2 - (s1 + 1));
                IF StrFind(x_str, 1, "X=") = 1 THEN
                    x_str := StrPart(x_str, 3, StrLen(x_str) - 2);
                ENDIF

                s3 := StrFind(raw_message, s2 + 1, ";");
                IF s3 > 0 THEN
                    y_str := StrPart(raw_message, s2 + 1, s3 - (s2 + 1));
                    IF StrFind(y_str, 1, "Y=") = 1 THEN
                        y_str := StrPart(y_str, 3, StrLen(y_str) - 2);
                    ENDIF

                    s4 := StrFind(raw_message, s3 + 1, ";");
                    IF s4 > 0 THEN
                        angle_str := StrPart(raw_message, s3 + 1, s4 - (s3 + 1));
                    ELSE
                        angle_str := StrPart(raw_message, s3 + 1, StrLen(raw_message) - s3);
                    ENDIF
                    IF StrFind(angle_str, 1, "RZ=") = 1 THEN
                        angle_str := StrPart(angle_str, 4, StrLen(angle_str) - 3);
                    ENDIF

                    cr_idx := StrFind(angle_str, 1, "\0D");
                    IF cr_idx > 0 THEN
                        angle_str := StrPart(angle_str, 1, cr_idx - 1);
                    ENDIF
                    cr_idx := StrFind(angle_str, 1, "\0A");
                    IF cr_idx > 0 THEN
                        angle_str := StrPart(angle_str, 1, cr_idx - 1);
                    ENDIF

                    ok := StrToVal(x_str, parsed_job.x_val);
                    ok := ok AND StrToVal(y_str, parsed_job.y_val);
                    ok := ok AND StrToVal(angle_str, parsed_job.rz_val);

                    IF ok THEN
                        RETURN TRUE;
                    ENDIF
                ENDIF
            ENDIF
        ENDIF
        
        RETURN FALSE;
    ENDFUNC

ENDMODULE