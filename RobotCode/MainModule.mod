MODULE MainModule

    VAR ShapeJob currentJob := ["pcircle", 148.24, 170.09, 0.0];
    VAR string servermMessage := "";
    VAR string clinetMessage := "";
    VAR bool connected := FALSE;
    
    VAR bool keepRunning := TRUE;
    VAR num idx1;
    VAR num idx2;
    VAR num idx3;
    VAR bool parse_ok;

    PROC main()
        ! Ensure currentJob has a default valid shape on startup so it can run immediately
        currentJob.job_Shape := "pcircle";
        currentJob.x_val := 148.24;
        currentJob.y_val := 170.09;
        currentJob.rz_val := 0.0;

        servermMessage := RobotClientReciveMessage();
        TPWrite "SERVER SAYS: " + servermMessage;

        ! Parse the server message if one was received
        IF StrLen(servermMessage) > 0 AND ParseJobString(servermMessage, currentJob) THEN
            ! Exit clause
            IF currentJob.job_Shape = "exit" THEN
                RobotClientCloseAndDisconnect;
                EXIT;
            ENDIF
        ELSE
            TPWrite "Using default shape: " + currentJob.job_Shape;
        ENDIF

        ! Update dynamic pick target coordinates
        shape.trans.x := currentJob.x_val;
        shape.trans.y := currentJob.y_val;
        shape.trans.z := 2.0;
        shape.rot := OrientZYX(currentJob.rz_val, 0.0, 0.0);

        ! Send message to server
        clinetMessage := "Time to die";
        RobotClienSendMessage(clinetMessage);

        ! Disconnect and close if exit
        IF currentJob.job_Shape = "exit" THEN
            RobotClientCloseAndDisconnect;
            EXIT;
        ENDIF

        ! Move to shape using valid shape value
        move_to_shape(currentJob.job_Shape);
    ENDPROC

ENDMODULE
