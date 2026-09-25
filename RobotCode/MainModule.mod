MODULE MainModule

    VAR ShapeJob currentJob;
    VAR string servermMessage;
    VAR string clinetMessage;
    VAR bool keepRunning := TRUE;

    PROC main()
        TPWrite "ABB Robot Vision Client Started.";
        WHILE keepRunning DO
            servermMessage := RobotClientReciveMessage();

            IF StrLen(servermMessage) > 0 THEN
                TPWrite "SERVER SAYS: " + servermMessage;

                ! Parse the server message
                IF ParseJobString(servermMessage, currentJob) THEN
                    ! Exit clause
                    IF currentJob.job_Shape = "exit" THEN
                        keepRunning := FALSE;
                        RobotClientCloseAndDisconnect;
                        EXIT;
                    ENDIF

                    ! Update the dynamic pick target coordinates and orientation
                    shape.trans.x := currentJob.x_val;
                    shape.trans.y := currentJob.y_val;
                    shape.trans.z := 2.0;
                    shape.rot := OrientZYX(currentJob.rz_val, 0.0, 0.0);

                    ! Move robot to pick and place shape
                    move_to_shape(currentJob.job_Shape);

                    ! Send completion message to server
                    clinetMessage := "DONE";
                    RobotClienSendMessage(clinetMessage);
                ELSE
                    TPWrite "Invalid message received, skipping move.";
                ENDIF
            ELSE
                ! No message waiting, pause briefly before checking again
                WaitTime 0.1;
            ENDIF
        ENDWHILE
    ENDPROC

ENDMODULE