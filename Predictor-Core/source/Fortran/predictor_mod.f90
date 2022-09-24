module ML_predictor

use, intrinsic :: ISO_C_Binding, only: C_int, C_double, C_float, C_char, C_size_t, C_null_char
use, intrinsic :: ISO_C_Binding, only: C_ptr, C_NULL_ptr

implicit none


interface 

! function C_strlen(s) result(size_s) bind(C,name="strlen")
!     import C_ptr, C_size_t
!     integer(C_size_t) :: size_s
!     type(C_ptr), value, intent(in) :: s  !character(len=*), intent(in)
! end function C_strlen

function C_createPredictorFromPB_Cstring (pb_name) result(self) bind(C, name="createPredictorFromPB")
    import
    CHARACTER(kind=C_char) :: pb_name(*)
    type(C_ptr) :: self
end function

function C_createPredictorFromSavedModel_Cstring (model_dir, tags) result(self) bind(C, name="createPredictorFromSavedModel")
    import
    CHARACTER(kind=C_char) :: model_dir(*)
    CHARACTER(kind=C_char) :: tags(*)
    type(C_ptr) :: self
end function

subroutine C_deletePredictor (self) bind(C, name="deletePredictor")
    import
    type(C_ptr), value :: self
end subroutine
    
subroutine C_PredictorSetDataCount (self, data_count) bind(C, name="PredictorSetDataCount")
    import 
    type(C_ptr), value :: self
    INTEGER(C_int), value :: data_count
end subroutine

subroutine C_PredictorRegisterInputNode_Cstring (self, node_name) bind(C, name="PredictorRegisterInputNode")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
end subroutine

subroutine C_PredictorRegisterOutputNode_Cstring (self, node_name) bind(C, name="PredictorRegisterOutputNode")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
end subroutine

subroutine C_PredictorSetNodeDataDouble_Cstring (self, node_name, p_data, data_length) bind(C, name="PredictorSetNodeDataDouble")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    ! REAL(C_double) :: p_data(0:data_length)
    REAL(C_double) :: p_data(*)

end subroutine

subroutine C_PredictorSetNodeDataFloat_Cstring (self, node_name, p_data, data_length) bind(C, name="PredictorSetNodeDataFloat")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    REAL(C_float) :: p_data(*)
end subroutine

subroutine C_PredictorSetNodeDataInt_Cstring (self, node_name, p_data, data_length) bind(C, name="PredictorSetNodeDataInt")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    INTEGER(C_int) :: p_data(*)
end subroutine

subroutine C_PredictorRun (self) bind(C, name="PredictorRun")
    import 
    type(C_ptr), value :: self
end subroutine

subroutine C_PredictorGetNodeDataDouble_Cstring (self, node_name, p_data, data_length) bind(C, name="PredictorGetNodeDataDouble")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    REAL(C_double) :: p_data(0:data_length)
end subroutine

subroutine C_PredictorGetNodeDataFloat_Cstring (self, node_name, p_data, data_length) bind(C, name="PredictorGetNodeDataFloat")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    REAL(C_float) :: p_data(0:data_length)
end subroutine

subroutine C_PredictorGetNodeDataInt_Cstring (self, node_name, p_data, data_length) bind(C, name="PredictorGetNodeDataInt")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    INTEGER(C_int) :: p_data(0:data_length)
end subroutine

!---------------------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeDouble_Cstring (self, node_name, p_data, data_length) &
    bind(C, name="PredictorSetNodeDataTransposeDouble")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    ! REAL(C_double) :: p_data(0:data_length)
    REAL(C_double) :: p_data(*)

end subroutine

subroutine C_PredictorSetNodeDataTransposeFloat_Cstring (self, node_name, p_data, data_length) &
    bind(C, name="PredictorSetNodeDataTransposeFloat")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    REAL(C_float) :: p_data(*)
end subroutine

subroutine C_PredictorSetNodeDataTransposeInt_Cstring (self, node_name, p_data, data_length) &
    bind(C, name="PredictorSetNodeDataTransposeInt")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    INTEGER(C_int) :: p_data(*)
end subroutine


subroutine C_PredictorGetNodeDataTransposeDouble_Cstring (self, node_name, p_data, data_length) &
    bind(C, name="PredictorGetNodeDataTransposeDouble")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    REAL(C_double) :: p_data(0:data_length)
end subroutine

subroutine C_PredictorGetNodeDataTransposeFloat_Cstring (self, node_name, p_data, data_length) &
    bind(C, name="PredictorGetNodeDataTransposeFloat")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    REAL(C_float) :: p_data(0:data_length)
end subroutine

subroutine C_PredictorGetNodeDataTransposeInt_Cstring (self, node_name, p_data, data_length) &
    bind(C, name="PredictorGetNodeDataTransposeInt")
    import 
    type(C_ptr), value :: self
    CHARACTER(kind=C_char) :: node_name(*)
    INTEGER(C_int), value :: data_length
    INTEGER(C_int) :: p_data(0:data_length)
end subroutine



end interface

interface C_createPredictor
module procedure  C_createPredictorFromPB_Fstring, C_createPredictorFromSavedModel_Fstring
end interface C_createPredictor


interface C_PredictorSetNodeData
module procedure  C_PredictorSetNodeDataInt_Fstring_1d, &
                  C_PredictorSetNodeDataFloat_Fstring_1d, &
                  C_PredictorSetNodeDataDouble_Fstring_1d, &
                  C_PredictorSetNodeDataInt_Fstring_2d, &
                  C_PredictorSetNodeDataFloat_Fstring_2d, &
                  C_PredictorSetNodeDataDouble_Fstring_2d, &
                  C_PredictorSetNodeDataInt_Fstring_3d, &
                  C_PredictorSetNodeDataFloat_Fstring_3d, &
                  C_PredictorSetNodeDataDouble_Fstring_3d, &
                  C_PredictorSetNodeDataInt_Fstring_4d, &
                  C_PredictorSetNodeDataFloat_Fstring_4d, &
                  C_PredictorSetNodeDataDouble_Fstring_4d, &
                  C_PredictorSetNodeDataInt_Fstring_5d, &
                  C_PredictorSetNodeDataFloat_Fstring_5d, &
                  C_PredictorSetNodeDataDouble_Fstring_5d, &
                  C_PredictorSetNodeDataInt_Fstring_6d, &
                  C_PredictorSetNodeDataFloat_Fstring_6d, &
                  C_PredictorSetNodeDataDouble_Fstring_6d
end interface C_PredictorSetNodeData



interface C_PredictorGetNodeData
module procedure  C_PredictorGetNodeDataInt_Fstring_1d, &
                  C_PredictorGetNodeDataFloat_Fstring_1d, &
                  C_PredictorGetNodeDataDouble_Fstring_1d, &
                  C_PredictorGetNodeDataInt_Fstring_2d, &
                  C_PredictorGetNodeDataFloat_Fstring_2d, &
                  C_PredictorGetNodeDataDouble_Fstring_2d, &
                  C_PredictorGetNodeDataInt_Fstring_3d, &
                  C_PredictorGetNodeDataFloat_Fstring_3d, &
                  C_PredictorGetNodeDataDouble_Fstring_3d, &
                  C_PredictorGetNodeDataInt_Fstring_4d, &
                  C_PredictorGetNodeDataFloat_Fstring_4d, &
                  C_PredictorGetNodeDataDouble_Fstring_4d, &
                  C_PredictorGetNodeDataInt_Fstring_5d, &
                  C_PredictorGetNodeDataFloat_Fstring_5d, &
                  C_PredictorGetNodeDataDouble_Fstring_5d, &
                  C_PredictorGetNodeDataInt_Fstring_6d, &
                  C_PredictorGetNodeDataFloat_Fstring_6d, &
                  C_PredictorGetNodeDataDouble_Fstring_6d
end interface C_PredictorGetNodeData


interface C_PredictorSetNodeDataTranspose
module procedure  C_PredictorSetNodeDataTransposeInt_Fstring_1d, &
                  C_PredictorSetNodeDataTransposeFloat_Fstring_1d, &
                  C_PredictorSetNodeDataTransposeDouble_Fstring_1d, &
                  C_PredictorSetNodeDataTransposeInt_Fstring_2d, &
                  C_PredictorSetNodeDataTransposeFloat_Fstring_2d, &
                  C_PredictorSetNodeDataTransposeDouble_Fstring_2d, &
                  C_PredictorSetNodeDataTransposeInt_Fstring_3d, &
                  C_PredictorSetNodeDataTransposeFloat_Fstring_3d, &
                  C_PredictorSetNodeDataTransposeDouble_Fstring_3d, &
                  C_PredictorSetNodeDataTransposeInt_Fstring_4d, &
                  C_PredictorSetNodeDataTransposeFloat_Fstring_4d, &
                  C_PredictorSetNodeDataTransposeDouble_Fstring_4d, &
                  C_PredictorSetNodeDataTransposeInt_Fstring_5d, &
                  C_PredictorSetNodeDataTransposeFloat_Fstring_5d, &
                  C_PredictorSetNodeDataTransposeDouble_Fstring_5d, &
                  C_PredictorSetNodeDataTransposeInt_Fstring_6d, &
                  C_PredictorSetNodeDataTransposeFloat_Fstring_6d, &
                  C_PredictorSetNodeDataTransposeDouble_Fstring_6d
end interface C_PredictorSetNodeDataTranspose



interface C_PredictorGetNodeDataTranspose
module procedure  C_PredictorGetNodeDataTransposeInt_Fstring_1d, &
                  C_PredictorGetNodeDataTransposeFloat_Fstring_1d, &
                  C_PredictorGetNodeDataTransposeDouble_Fstring_1d, &
                  C_PredictorGetNodeDataTransposeInt_Fstring_2d, &
                  C_PredictorGetNodeDataTransposeFloat_Fstring_2d, &
                  C_PredictorGetNodeDataTransposeDouble_Fstring_2d, &
                  C_PredictorGetNodeDataTransposeInt_Fstring_3d, &
                  C_PredictorGetNodeDataTransposeFloat_Fstring_3d, &
                  C_PredictorGetNodeDataTransposeDouble_Fstring_3d, &
                  C_PredictorGetNodeDataTransposeInt_Fstring_4d, &
                  C_PredictorGetNodeDataTransposeFloat_Fstring_4d, &
                  C_PredictorGetNodeDataTransposeDouble_Fstring_4d, &
                  C_PredictorGetNodeDataTransposeInt_Fstring_5d, &
                  C_PredictorGetNodeDataTransposeFloat_Fstring_5d, &
                  C_PredictorGetNodeDataTransposeDouble_Fstring_5d, &
                  C_PredictorGetNodeDataTransposeInt_Fstring_6d, &
                  C_PredictorGetNodeDataTransposeFloat_Fstring_6d, &
                  C_PredictorGetNodeDataTransposeDouble_Fstring_6d
end interface C_PredictorGetNodeDataTranspose

contains

!-------------------------------------------------------------------------------------------------

function Fstring_to_Cstring(Fstring) result(Cstring)
    character(len=*) :: Fstring
    character(kind=c_char), dimension(len(Fstring)+1) :: Cstring
    integer :: i
    do i = 1, len(Fstring)
        Cstring(i) = Fstring(i:i)
    end do
    Cstring(len(Fstring)+1) = C_null_char
end function

function C_createPredictorFromPB_Fstring (pb_name) result(self) 
    type(C_ptr) :: self
    character(len=*), intent(in) :: pb_name
    character(kind=c_char), dimension(len(pb_name)+1) :: pb_name_cstring
    pb_name_cstring = Fstring_to_Cstring(pb_name)
    self = C_createPredictorFromPB_Cstring(pb_name_cstring)
end function

function C_createPredictorFromSavedModel_Fstring (model_dir, tags) result(self)
    character(len=*), intent(in) :: model_dir
    character(kind=c_char), dimension(len(model_dir)+1) :: model_dir_cstring
    character(len=*), intent(in) :: tags
    character(kind=c_char), dimension(len(tags)+1) :: tags_cstring
    type(C_ptr) :: self
    model_dir_cstring = Fstring_to_Cstring(model_dir)
    tags_cstring = Fstring_to_Cstring(tags)
    self = C_createPredictorFromSavedModel_Cstring(model_dir_cstring, tags_cstring)
end function

subroutine C_PredictorRegisterInputNode (self, node_name)
    type(C_ptr), value :: self
    character(len=*), intent(in) :: node_name
    character(kind=c_char), dimension(len(node_name)+1) :: node_name_cstring
    node_name_cstring = Fstring_to_Cstring(node_name)
    call C_PredictorRegisterInputNode_Cstring(self, node_name_cstring)
end subroutine

subroutine C_PredictorRegisterOutputNode (self, node_name)
    type(C_ptr), value :: self
    character(len=*), intent(in) :: node_name
    character(kind=c_char), dimension(len(node_name)+1) :: node_name_cstring
    node_name_cstring = Fstring_to_Cstring(node_name)
    call C_PredictorRegisterOutputNode_Cstring(self, node_name_cstring)
end subroutine


!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataInt_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataInt_Fstring_1d


subroutine C_PredictorSetNodeDataFloat_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataFloat_Fstring_1d


subroutine C_PredictorSetNodeDataDouble_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataDouble_Fstring_1d



subroutine C_PredictorGetNodeDataInt_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataInt_Fstring_1d


subroutine C_PredictorGetNodeDataFloat_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataFloat_Fstring_1d

subroutine C_PredictorGetNodeDataDouble_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataDouble_Fstring_1d



!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataInt_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataInt_Fstring_2d


subroutine C_PredictorSetNodeDataFloat_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataFloat_Fstring_2d


subroutine C_PredictorSetNodeDataDouble_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataDouble_Fstring_2d



subroutine C_PredictorGetNodeDataInt_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataInt_Fstring_2d


subroutine C_PredictorGetNodeDataFloat_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataFloat_Fstring_2d

subroutine C_PredictorGetNodeDataDouble_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataDouble_Fstring_2d


!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataInt_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataInt_Fstring_3d


subroutine C_PredictorSetNodeDataFloat_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataFloat_Fstring_3d


subroutine C_PredictorSetNodeDataDouble_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataDouble_Fstring_3d



subroutine C_PredictorGetNodeDataInt_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataInt_Fstring_3d


subroutine C_PredictorGetNodeDataFloat_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataFloat_Fstring_3d

subroutine C_PredictorGetNodeDataDouble_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataDouble_Fstring_3d

!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataInt_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataInt_Fstring_4d


subroutine C_PredictorSetNodeDataFloat_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataFloat_Fstring_4d


subroutine C_PredictorSetNodeDataDouble_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataDouble_Fstring_4d



subroutine C_PredictorGetNodeDataInt_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataInt_Fstring_4d


subroutine C_PredictorGetNodeDataFloat_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataFloat_Fstring_4d

subroutine C_PredictorGetNodeDataDouble_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataDouble_Fstring_4d

!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataInt_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataInt_Fstring_5d


subroutine C_PredictorSetNodeDataFloat_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataFloat_Fstring_5d


subroutine C_PredictorSetNodeDataDouble_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataDouble_Fstring_5d



subroutine C_PredictorGetNodeDataInt_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataInt_Fstring_5d


subroutine C_PredictorGetNodeDataFloat_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataFloat_Fstring_5d

subroutine C_PredictorGetNodeDataDouble_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataDouble_Fstring_5d

!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataInt_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataInt_Fstring_6d


subroutine C_PredictorSetNodeDataFloat_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataFloat_Fstring_6d


subroutine C_PredictorSetNodeDataDouble_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataDouble_Fstring_6d



subroutine C_PredictorGetNodeDataInt_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataInt_Fstring_6d


subroutine C_PredictorGetNodeDataFloat_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataFloat_Fstring_6d

subroutine C_PredictorGetNodeDataDouble_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataDouble_Fstring_6d


!----------------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeInt_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeInt_Fstring_1d


subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_1d


subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_1d



subroutine C_PredictorGetNodeDataTransposeInt_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeInt_Fstring_1d


subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_1d

subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_1d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_1d



!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeInt_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeInt_Fstring_2d


subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_2d


subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_2d



subroutine C_PredictorGetNodeDataTransposeInt_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeInt_Fstring_2d


subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_2d

subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_2d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_2d


!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeInt_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeInt_Fstring_3d


subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_3d


subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_3d



subroutine C_PredictorGetNodeDataTransposeInt_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeInt_Fstring_3d


subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_3d

subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_3d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_3d

!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeInt_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeInt_Fstring_4d


subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_4d


subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_4d



subroutine C_PredictorGetNodeDataTransposeInt_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeInt_Fstring_4d


subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_4d

subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_4d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_4d

!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeInt_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeInt_Fstring_5d


subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_5d


subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_5d



subroutine C_PredictorGetNodeDataTransposeInt_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeInt_Fstring_5d


subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_5d

subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_5d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_5d

!-------------------------------------------------------------------------------------------------

subroutine C_PredictorSetNodeDataTransposeInt_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeInt_Fstring_6d


subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeFloat_Fstring_6d


subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorSetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorSetNodeDataTransposeDouble_Fstring_6d



subroutine C_PredictorGetNodeDataTransposeInt_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    INTEGER(C_int) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeInt_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeInt_Fstring_6d


subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_float) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeFloat_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeFloat_Fstring_6d

subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_6d (self, f_node_name, p_data, data_length) 
    type(C_ptr) :: self
    INTEGER(C_int) :: data_length
    REAL(C_double) :: p_data(:,:,:,:,:,:)
    character(len=*), intent(in) :: f_node_name
    character(kind=c_char), dimension(len(f_node_name)+1) :: c_node_name
    c_node_name = Fstring_to_Cstring(f_node_name)
    call C_PredictorGetNodeDataTransposeDouble_Cstring(self, c_node_name, p_data, data_length)
end subroutine C_PredictorGetNodeDataTransposeDouble_Fstring_6d






















    
end module ML_predictor
