<<<<<<< Updated upstream:MainModule.mod
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

=======
MODULE MainModule

    VAR ShapeJob currentJob;
    VAR string servermMessage;
    VAR string clinetMessage;
    VAR bool connected := FALSE;

    PROC main()
        
!        ! 1. ACTUALLY CONNECT TO PYTHON FIRST
        RobotAsClientConnect;

!        MoveL pCircle, v1000, fine, tsuck\WObj:=wobjAiPlaceTray;
!        MoveL pSquare, v1000, fine, tsuck\WObj:=wobjAiPlaceTray;
!        MoveL pStar, v1000, fine, tsuck\WObj:=wobjAiPlaceTray;
!        MoveL pHexagon, v1000, fine, tsuck\WObj:=wobjAiPlaceTray;
!        MoveL pTriangle, v1000, fine, tsuck\WObj:=wobjAiPlaceTray;
        
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
                ! (If you haven't taught pArucoHover yet, use pHome for now)
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
    
    
>>>>>>> Stashed changes:RobotCode/MainModule.mod
ENDMODULE