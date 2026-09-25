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
        VAR num msg_len;
        VAR num idx1;
        VAR num idx2;
        VAR num idx3;
        VAR bool ok := FALSE;

        msg_len := StrLen(raw_message);

        ! Empty message check
        IF msg_len = 0 THEN
            RETURN FALSE;
        ENDIF

        ! Direct exit command check (e.g. from Python server when terminating)
        IF raw_message = "exit" THEN
            parsed_job.job_Shape := "exit";
            parsed_job.x_val := 0;
            parsed_job.y_val := 0;
            parsed_job.rz_val := 0;
            RETURN TRUE;
        ENDIF

        ! Find 1st comma (after shape name)
        ! In RAPID, if not found, StrFind returns msg_len + 1
        idx1 := StrFind(raw_message, 1, ",");
        IF idx1 > msg_len THEN
            RETURN FALSE;
        ENDIF

        parsed_job.job_Shape := StrPart(raw_message, 1, idx1 - 1);

        ! Ensure there are characters after 1st comma
        IF idx1 >= msg_len THEN
            RETURN FALSE;
        ENDIF

        ! Find 2nd comma (after X)
        idx2 := StrFind(raw_message, idx1 + 1, ",");
        IF idx2 > msg_len THEN
            RETURN FALSE;
        ENDIF

        ! Extract X coordinate string
        x_str := StrPart(raw_message, idx1 + 1, idx2 - idx1 - 1);

        ! Ensure there are characters after 2nd comma
        IF idx2 >= msg_len THEN
            RETURN FALSE;
        ENDIF

        ! Find 3rd comma (after Y, before angle if provided)
        idx3 := StrFind(raw_message, idx2 + 1, ",");
        IF idx3 <= msg_len THEN
            ! 3 commas present: "shape,X,Y,angle"
            y_str := StrPart(raw_message, idx2 + 1, idx3 - idx2 - 1);
            IF idx3 < msg_len THEN
                angle_str := StrPart(raw_message, idx3 + 1, msg_len - idx3);
            ELSE
                angle_str := "0";
            ENDIF
        ELSE
            ! Only 2 commas present: "shape,X,Y" (default angle to 0)
            y_str := StrPart(raw_message, idx2 + 1, msg_len - idx2);
            angle_str := "0";
        ENDIF

        ! Convert strings to numbers
        ok := StrToVal(x_str, parsed_job.x_val);
        ok := ok AND StrToVal(y_str, parsed_job.y_val);
        ok := ok AND StrToVal(angle_str, parsed_job.rz_val);

        IF ok THEN
            RETURN TRUE; ! Parsing was 100% successful
        ENDIF

        RETURN FALSE; ! Something failed
    ENDFUNC

ENDMODULE