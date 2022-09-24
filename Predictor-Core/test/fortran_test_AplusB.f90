program main
use ML_predictor
use iso_c_binding, only: C_ptr

    implicit none
    ! Predictor
    TYPE(C_ptr) :: pd

    ! 2D data float (Support up to 6d)
    REAL(kind=4), dimension(2,3)  :: arr_a = reshape((/0.0, 1.1, 2.2, 3.3, 4.4, 5.5/), (/2,3/))
    INTEGER(kind=4), dimension(6) :: arr_b = (/5, 4, 3, 2, 1, 0/)
    REAL(kind=8), dimension(3,2)  :: arr_c = 0.0d0


    ! Create predictor from *.pb

    pd = C_createPredictor("test/models/simple_graph_tf2.pb")
    
    call C_PredictorRegisterInputNode(pd, "input_a")
    call C_PredictorRegisterInputNode(pd, "input_b")
    call C_PredictorRegisterOutputNode(pd, "result")

    ! Set the input data number and data
    call C_PredictorSetDataCount(pd, 3);
    call C_PredictorSetNodeData(pd, "input_a", arr_a, 6)
    call C_PredictorSetNodeData(pd, "input_b", arr_b, 6)

    ! Run the model 
    call C_PredictorRun(pd)

    ! Get the model output into 
    call C_PredictorGetNodeData(pd, "result", arr_c, 6)

    ! Print the output
    write (*, *) "1D Running Result:" 
    write (*, *) arr_c

    call C_deletePredictor(pd)

    
end program main



