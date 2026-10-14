MODULE moveToShape
    PROC move_to_shape(string shape_input, num rz_val)
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
            
            ! Move to tray approach with rotation applied
            MoveJ RelTool(pApporach, 0, 0, 0 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;

            ! Sorting logic (place into tray slot with rotation)
            IF shape_input = "Star" THEN
                MoveL RelTool(pStar, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL RelTool(pStar, 0, 0, 0 \Rz:=rz_val), v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pStar, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Square" THEN
                MoveL RelTool(pSquare, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL RelTool(pSquare, 0, 0, 0 \Rz:=rz_val), v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pSquare, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Hexagon" THEN
                MoveL RelTool(pHexagon, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL RelTool(pHexagon, 0, 0, 0 \Rz:=rz_val), v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pHexagon, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Circle" THEN
                MoveL RelTool(pCircle, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL RelTool(pCircle, 0, 0, 0 \Rz:=rz_val), v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pCircle, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Triangle" THEN
                MoveL RelTool(pTriangle, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL RelTool(pTriangle, 0, 0, 0 \Rz:=rz_val), v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pTriangle, 0, 0, -100 \Rz:=rz_val), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ENDIF
            
            ! Move back to safe approach over tray
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
    ENDPROC
    
ENDMODULE
