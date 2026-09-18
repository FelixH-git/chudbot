MODULE moveToShape
    PROC move_to_shape(string shape_input)
            waitforOK("");
            MoveL phome, v100, fine, tsuck\WObj:=wobj1;
            MoveL shape, v100 , fine, tsuck\WObj:=wobjAruco;
            suckOn;
            WaitTime(1);
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;

            
            !movers 
            IF shape_input = "pstar" THEN
                MoveL pStar, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "psquare" THEN
                MoveL pSquare, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "phexa" THEN
                MoveL pHexagon, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "pcircle" THEN
                MoveL pCircle, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            IF shape_input = "ptriangle" THEN
                MoveL pTriangle, v100 , fine, tsuck\WObj:=wobjAiPlaceTray;
                WaitTime(1);
                suckOff;
            ENDIF
            MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            
        ENDPROC
ENDMODULE