MODULE moveToShape
    PROC move_to_shape(string shape_input)
            waitforOK("");
            MoveJ phome, v100, fine, tsuck\WObj:=wobjAruco;
            
            ! 1. Safe Approach: Hover 100mm directly above the shape along the tool's Z-axis
            MoveL RelTool(shape, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAruco;
            
            ! 2. Go down to pick
            MoveL shape, v50 , fine, tsuck\WObj:=wobjAruco;
            suckOn;
            WaitTime 1.0;
            
            ! 3. Safe Depart: Lift straight back up 100mm along the tool's Z-axis
            MoveL RelTool(shape, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAruco;
            
            ! Move to tray approach
            MoveJ pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;

            ! Sorting logic
            IF shape_input = "Star" THEN
                MoveL RelTool(pStar, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL pStar, v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pStar, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Square" THEN
                MoveL RelTool(pSquare, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL pSquare, v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pSquare, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Hexagon" THEN
                MoveL RelTool(pHexagon, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL pHexagon, v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pHexagon, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Circle" THEN
                MoveL RelTool(pCircle, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL pCircle, v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pCircle, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ELSEIF shape_input = "Triangle" THEN
                MoveL RelTool(pTriangle, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
                MoveL pTriangle, v50 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime 1.0;
                suckOff;
                MoveL RelTool(pTriangle, 0, 0, -100), v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            ENDIF
            
            ! Move back to safe approach over tray
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
    ENDPROC
    
ENDMODULE
