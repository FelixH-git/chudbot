MODULE moveToShape
    PROC move_to_shape(string shape_input, num rz_val)
            VAR robtarget pDest;
            VAR robtarget pDestApp;
            VAR orient held_orient;

            waitforOK("");
            MoveJ phome, v100, fine, tsuck\WObj:=wobjAruco;
            
            ! 1. Safe Approach: Hover 100mm directly above the shape (neutral wrist)
            MoveL RelTool(shape, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAruco;
            
            ! 2. Go down to pick (neutral wrist)
            MoveL shape, v50 , fine, tsuck\WObj:=wobjAruco;
            suckOn;
            WaitTime 1.0;
            
            ! 3. Safe Depart: Lift straight back up 100mm along the tool's Z-axis
            MoveL RelTool(shape, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAruco;
            
            ! 4. Rotate the piece in the air AFTER picking up!
            MoveL RelTool(shape, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAruco;
            
            ! 5. Freeze this exact rotation so the wrist DOES NOT rotate on the way to the tray!
            held_orient := CRobT(\Tool:=tsuck \WObj:=wobjAiPlaceTray).rot;
            
            ! Disable configuration checking so the robot takes the smooth path without twisting Axis 6
            ConfJ \Off;
            ConfL \Off;

            ! Move to tray approach keeping the EXACT rotation
            pDestApp := pApporach;
            pDestApp.rot := held_orient;
            MoveJ pDestApp, v100, fine, tsuck\WObj:=wobjAiPlaceTray;

            ! Select the target slot
            IF shape_input = "Star" THEN
                pDest := pStar;
            ELSEIF shape_input = "Square" THEN
                pDest := pSquare;
            ELSEIF shape_input = "Hexagon" THEN
                pDest := pHexagon;
            ELSEIF shape_input = "Circle" THEN
                pDest := pCircle;
            ELSEIF shape_input = "Triangle" THEN
                pDest := pTriangle;
            ELSE
                TPWrite "Unknown shape: " + shape_input;
                ConfJ \On;
                ConfL \On;
                RETURN;
            ENDIF

            ! Apply the exact rotation from the air to the tray slot
            pDest.rot := held_orient;

            ! Place into slot maintaining the rotation
            MoveL RelTool(pDest, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            MoveL pDest, v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1.0;
            suckOff;
            MoveL RelTool(pDest, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            
            ! Restore configuration checking
            ConfJ \On;
            ConfL \On;

            ! Move back to safe approach over tray
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
    ENDPROC
    
ENDMODULE
