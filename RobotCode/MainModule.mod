MODULE MainModule

    VAR ShapeJob currentJob;
    VAR string servermMessage;
    VAR string clinetMessage;
    VAR bool connected := FALSE;

    VAR bool keepRunning := TRUE;
    VAR num idx1;
    VAR num idx2;
    VAR num idx3;
    VAR bool parse_ok;
    PROC main()
        servermMessage := RobotClientReciveMessage();
        TPWrite "SERVER SAYS" + servermMessage;
        ! parse the server message
        IF ParseJobString(servermMessage, currentJob) THEN
            !exit clause
            IF currentJob.job_Shape = "exit" THEN
                EXIT;
            ENDIF
        ENDIF
        !Send message to server
        clinetMessage:="Time to die";
        RobotClienSendMessage(clinetMessage);

        !disconnect and close
        IF currentJob.job_Shape = "exit" THEN
            RobotClientCloseAndDisconnect;
        ENDIF
        move_to_shape(currentJob.job_Shape);
    ENDPROC

ENDMODULE
