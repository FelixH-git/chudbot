MODULE moveToShape

    PROC move_to_shape(string shape_input)
        VAR bool isStar := FALSE;
        VAR bool isSquare := FALSE;
        VAR bool isHexa := FALSE;
        VAR bool isCircle := FALSE;
        VAR bool isTriangle := FALSE;

        ! Check shape matching (supports p-prefix or direct shape names)
        IF shape_input = "pstar" OR shape_input = "Star" OR shape_input = "star" THEN
            isStar := TRUE;
        ELSEIF shape_input = "psquare" OR shape_input = "Square" OR shape_input = "square" THEN
            isSquare := TRUE;
        ELSEIF shape_input = "phexa" OR shape_input = "Hexagon" OR shape_input = "hexagon" OR shape_input = "phex" THEN
            isHexa := TRUE;
        ELSEIF shape_input = "pcircle" OR shape_input = "Circle" OR shape_input = "circle" THEN
            isCircle := TRUE;
        ELSEIF shape_input = "ptriangle" OR shape_input = "Triangle" OR shape_input = "triangle" THEN
            isTriangle := TRUE;
        ELSE
            TPWrite "Unknown shape requested: " + shape_input;
            RETURN;
        ENDIF

        ! Optional operator confirm
        waitforOK(shape_input);

        ! Pick sequence at detected shape target (wobjAruco)
        MoveL phome, v100, fine, tsuck\WObj:=wobj1;
        MoveL shape, v100, fine, tsuck\WObj:=wobjAruco;
        suckOn;
        WaitTime 1;
        MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;

        ! Place sequence into corresponding tray slot (wobjAiPlaceTray)
        IF isStar THEN
            MoveL pStar, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ELSEIF isSquare THEN
            MoveL pSquare, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ELSEIF isHexa THEN
            MoveL pHexagon, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ELSEIF isCircle THEN
            MoveL pCircle, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ELSEIF isTriangle THEN
            MoveL pTriangle, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
            WaitTime 1;
            suckOff;
        ENDIF

        ! Retract back up to approach and home positions
        MoveL pApporach, v100, fine, tsuck\WObj:=wobjAiPlaceTray;
        MoveL phome, v100, fine, tsuck\WObj:=wobj1;
    ENDPROC

ENDMODULE