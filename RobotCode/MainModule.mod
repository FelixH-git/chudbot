MODULE MainModule

    VAR ShapeJob currentJob;
    VAR string servermMessage;
    VAR string clinetMessage;
    VAR bool connected := FALSE;

    PROC main()
        
        ! 1. ACTUALLY CONNECT TO PYTHON FIRST
        RobotAsClientConnect;

        WHILE TRUE DO
            servermMessage := RobotClientReciveMessage();
            
            IF ParseJobString(servermMessage, currentJob) THEN
                
                IF currentJob.job_Shape = "exit" THEN 
                    clinetMessage := "time TO DIE";
                    RobotClienSendMessage(clinetMessage);
                    RobotClientCloseAndDisconnect;
                    EXIT;
                ENDIF
                
                ! 1. Inherit safe downward wrist angle from the Aruco area
                shape := pHome; 
                
                ! 2. Apply the AI Coordinates to the X and Y
                shape.trans.x := currentJob.x_val;
                shape.trans.y := currentJob.y_val;
                
                ! 3. Set the pick height (Z). Negative numbers mean it will pick from higher up
                shape.trans.z := 0.0; 
                ! 4. Apply the AI rotation (spinning around the tool's Z-axis)
                ! If the robot rotates the wrong way (CW instead of CCW), change this to -currentJob.rz_val
                shape := RelTool(shape, 0, 0, 0 \Rz:=currentJob.rz_val);
                
                PrintRobtarget(shape);
                
                ! 4. Call the sorting module
                move_to_shape currentJob.job_Shape;
                
            ELSE
                IF servermMessage <> "" THEN
                    TPWrite "parse failed " + servermMessage; 
                ENDIF
            ENDIF

        ENDWHILE
        
    ENDPROC
    
    PROC PrintRobtarget(robtarget p)
        TPWrite "X: " + NumToStr(p.trans.x, 2);
        TPWrite "Y: " + NumToStr(p.trans.y, 2);
        TPWrite "Z: " + NumToStr(p.trans.z, 2);
    ENDPROC
    
    proc suckOn()
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
