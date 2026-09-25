MODULE moveToShape
    PROC move_to_shape(string shape_input)
            ! Fallback if empty shape input is provided
            IF shape_input = "" THEN
                shape_input := "pcircle";
            ENDIF

            waitforOK(shape_input);
            MoveL phome, v100, fine, tsuck\WObj:=wobj1;
            MoveL shape, v100 , fine, tsuck\WObj:=wobjAruco;
            suckOn;
            WaitTime(1);
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;

            !movers 
            IF shape_input = "pstar" OR shape_input = "Star" OR shape_input = "star" THEN
                MoveL pStar, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "psquare" OR shape_input = "Square" OR shape_input = "square" THEN
                MoveL pSquare, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "phexa" OR shape_input = "Hexagon" OR shape_input = "hexagon" OR shape_input = "phex" THEN
                MoveL pHexagon, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "pcircle" OR shape_input = "Circle" OR shape_input = "circle" THEN
                MoveL pCircle, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "ptriangle" OR shape_input = "Triangle" OR shape_input = "triangle" THEN
                MoveL pTriangle, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            
        ENDPROC
ENDMODULE
