MODULE MainModule
    ! ============================================================
    !  Entry point + message parsing + pick/sort routine.
    !  The TCP socket itself is the teacher's ServerCommuncatio
    !  module (server.txt) and is not modified here.
    !
    !  Loop: receive one pick job -> move the suction cup to the
    !  AI coordinates -> sort the piece into its tray slot.
    !
    !  Wire format from Python (one message per pick):
    !      SHAPE=pstar;X=86.0;Y=-53.2;RZ=32.2;END\r\n
    ! ============================================================

    ! One parsed pick job. Defined here, visible to every module.
    RECORD ShapeJob
        string job_Shape;
        num x_val;
        num y_val;
        num rz_val;
    ENDRECORD

    VAR ShapeJob currentJob;
    VAR string serverMessage;
    VAR string clientMessage;

    PROC main()
        ! 1. Connect to Python first (the robot is the TCP client).
        RobotAsClientConnect;

        WHILE TRUE DO
            serverMessage := RobotClientReciveMessage();

            IF ParseJobString(serverMessage, currentJob) THEN

                ! "exit" lets the PC shut the robot down cleanly.
                IF currentJob.job_Shape = "exit" THEN
                    clientMessage := "time TO DIE";
                    RobotClienSendMessage(clientMessage);
                    RobotClientCloseAndDisconnect;
                    EXIT;
                ENDIF

                ! Inherit the safe downward wrist angle from the ArUco area,
                ! then apply the X/Y the AI computed, at a safe pick height.
                shape := pArucoHover;
                shape.trans.x := currentJob.x_val;
                shape.trans.y := currentJob.y_val;
                shape.trans.z := 15.0;

                move_to_shape currentJob.job_Shape;

            ELSE
                IF serverMessage <> "" THEN
                    TPWrite "parse failed " + serverMessage;
                ENDIF
            ENDIF
        ENDWHILE
    ENDPROC

    ! --- Parse one line "KEY=value" (separated by ";") into a ShapeJob ---
    ! Returns TRUE only when SHAPE, X, Y and RZ were all found, so a
    ! half-received line can never drive the robot.
    FUNC bool ParseJobString(string raw_message, INOUT ShapeJob parsed_job)
        VAR num search_start := 1;
        VAR num semi_idx;
        VAR num eq_idx;
        VAR string TOKEN;
        VAR string key;
        VAR string val;
        VAR bool got_shape := FALSE;
        VAR bool got_x := FALSE;
        VAR bool got_y := FALSE;
        VAR bool got_rz := FALSE;
        VAR num tmp_num;

        TPWrite "RECEIVED: " + raw_message;

        WHILE search_start <= StrLen(raw_message) DO
            semi_idx := StrFind(raw_message, search_start, ";");
            IF semi_idx = 0 THEN
                semi_idx := StrLen(raw_message) + 1;
            ENDIF

            ! Shield 1: there must be data between the separators.
            IF (semi_idx - search_start) > 0 THEN
                TOKEN := StrPart(raw_message, search_start, semi_idx - search_start);
                eq_idx := StrFind(TOKEN, 1, "=");

                ! Shield 2: '=' present, not first and not last character.
                IF (eq_idx > 1) AND (eq_idx < StrLen(TOKEN)) THEN
                    key := StrPart(TOKEN, 1, eq_idx - 1);
                    val := StrPart(TOKEN, eq_idx + 1, StrLen(TOKEN) - eq_idx);

                    IF key = "SHAPE" THEN
                        parsed_job.job_Shape := val;
                        got_shape := TRUE;
                    ELSEIF key = "X" THEN
                        got_x := StrToVal(val, tmp_num);
                        parsed_job.x_val := tmp_num;
                    ELSEIF key = "Y" THEN
                        got_y := StrToVal(val, tmp_num);
                        parsed_job.y_val := tmp_num;
                    ELSEIF key = "RZ" THEN
                        got_rz := StrToVal(val, tmp_num);
                        parsed_job.rz_val := tmp_num;
                    ENDIF
                ENDIF
            ENDIF

            search_start := semi_idx + 1;
        ENDWHILE

        RETURN got_shape AND got_x AND got_y AND got_rz;
    ENDFUNC

    ! --- Pick one piece and drop it in its tray slot ---
    ! shape_input is the robot key sent by Python: pstar / psquare /
    ! phexa / pcircle / ptriangle (see app/vision.py).
    PROC move_to_shape(string shape_input)
        waitforOK("");
        MoveL pHome, v100, fine, tsuck\WObj:=wobj1;
        MoveL shape, v100, fine, tsuck\WObj:=wobjAruco;
        suckOn;
        WaitTime 1;
        MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;

        ! Sort into the matching tray slot.
        IF shape_input = "pstar" THEN
            MoveL pStar, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ENDIF
        IF shape_input = "psquare" THEN
            MoveL pSquare, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ENDIF
        IF shape_input = "phexa" THEN
            MoveL pHexagon, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ENDIF
        IF shape_input = "pcircle" THEN
            MoveL pCircle, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ENDIF
        IF shape_input = "ptriangle" THEN
            MoveL pTriangle, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ENDIF

        MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
    ENDPROC

    ! --- Suction + operator IO. Only these PROCs touch physical I/O. ---
    PROC suckOn()
        SetDO doValve2, 1;
    ENDPROC

    PROC suckOff()
        SetDO doValve2, 0;
    ENDPROC

    PROC waitforOK(string action)
        TPWrite action;
        WaitDI diOkButton, 1;
        WaitDI diOkButton, 0;
        TPWrite "Performing " + action;
    ENDPROC

ENDMODULE
