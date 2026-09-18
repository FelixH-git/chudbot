MODULE GripperIO
    
! Only this module is allowed to touch physical inputs and outputs
    PROC suckOn()
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