module mlvars
  !
  use iso_c_binding, only: c_ptr, C_char
  !
  implicit none
  ! PRIVATE
  ! ML predictor
  ! type(c_ptr),save :: pd_ml
  ! character(kind=C_char, len=:), save, allocatable :: in_node_name, out_node_name
  TYPE :: pd_pack
    type(c_ptr) :: pd_ml
    character(kind=C_char, len=:), allocatable :: in_node_name, out_node_name
    !
    ! ML Arrays
    real,allocatable,dimension(:,:,:) :: ml_input_q1, ml_input_q2, ml_output
    real,allocatable,dimension(:,:) :: ml_input_container, ml_output_container

    ! ML kwSST Arrays
    real,allocatable,dimension(:,:,:) :: R_weight, Ndy, Ret, vist_rans
    ! Temp arrays stored in predictor
    real,allocatable,dimension(:,:,:) :: U_tmp, V_tmp, W_tmp, k_tmp, omega_tmp, kmean_tmp


    ! Status Vars
    integer :: n_pd_created = 0, n_arr_created = 0

  end type pd_pack

  type(pd_pack), allocatable, dimension(:) :: pd_pack_arr



  !
  !
  contains
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is used to allocate common array.                 |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 02-05-2022  | Created by Weishuo Liu                              |
  !+-------------------------------------------------------------------+
  subroutine AllocatePredictors(npredictor)
    integer :: npredictor
    allocate(pd_pack_arr(npredictor))
    print*, 'allocated pd_pack_arr mlvars.F90', npredictor
  end subroutine AllocatePredictors

  subroutine CheckPredictorsSize()
    print*, 'Predictor size mlvars.F90', size(pd_pack_arr)
  end subroutine CheckPredictorsSize

  subroutine Init_ML_Predictor(n_pd_arr)
    !
    use ML_Predictor
    ! use iso_c_binding, only: C_ptr
    use ISO_C_Binding, only: C_char

    implicit none
    integer, intent(in) :: n_pd_arr
    integer :: n_ml_options = 99
    character(kind=C_char, len=256) :: buffer_string
    character(kind=C_char, len=:), allocatable :: file_name, tags_name

    if ( pd_pack_arr(n_pd_arr)%n_pd_created /= 0) then
      print*, 'Predictor has already been created'
      return
    end if

    open(unit=1220, file="ML_inputs/ML_Predictor.in",status="OLD",action="READ")
    read(1220,*)  
    read(1220,*)  n_ml_options
    ! print*, "n_ml_options is: ", n_ml_options

    read(1220,*)  
    read(1220,'(a256)')  buffer_string
    print*, "buffer_string is: ", trim(buffer_string)
    file_name = trim(buffer_string)
    ! print*, "file_name is: ", file_name, "   len(file_name) is: ", len(file_name)


    read(1220,*)  
    read(1220,'(a256)')  buffer_string
    tags_name = trim(buffer_string)
    ! print*, "tags_name is: ", tags_name, "   len(tags_name) is: ", len(tags_name)


    read(1220,*)  
    read(1220,'(a256)')  buffer_string
    pd_pack_arr(n_pd_arr)%in_node_name = trim(buffer_string)
    ! print*, "in_node_name is: ", in_node_name, "   len(in_node_name) is: ", len(in_node_name)



    read(1220,*)  
    read(1220,'(a256)')  buffer_string
    pd_pack_arr(n_pd_arr)%out_node_name = trim(buffer_string)
    ! print*, "out_node_name is: ", out_node_name, "   len(out_node_name) is: ", len(out_node_name)
    
    ! f_no, face, ist, iend, jst, jend,  neighb, subface, orient
    close(unit=1220)

    if ( n_ml_options == 0) then
        print*, "n_ml_options is: ", n_ml_options
        print*, "Reading models from PB graph:"
        print*, "file_name is: ", file_name
        print*, "in_node_name is: ", pd_pack_arr(n_pd_arr)%in_node_name
        print*, "out_node_name is: ", pd_pack_arr(n_pd_arr)%out_node_name
        print*, "Creating ML predictor:"
        pd_pack_arr(n_pd_arr)%pd_ml = C_createPredictor(file_name)
        call C_PredictorRegisterInputNode(pd_pack_arr(n_pd_arr)%pd_ml, pd_pack_arr(n_pd_arr)%in_node_name)
        call C_PredictorRegisterOutputNode(pd_pack_arr(n_pd_arr)%pd_ml, pd_pack_arr(n_pd_arr)%out_node_name)   
        print*, "ML predictor created."

    else if (n_ml_options == 1) then
        print*, "n_ml_options is: ", n_ml_options
        print*, "Reading models from  SavedModel format:"
        print*, "file_name is: ", file_name
        print*, "tags_name is: ", tags_name
        print*, "in_node_name is: ", pd_pack_arr(n_pd_arr)%in_node_name
        print*, "out_node_name is: ", pd_pack_arr(n_pd_arr)%out_node_name
        print*, "Creating ML predictor:"
        pd_pack_arr(n_pd_arr)%pd_ml = C_createPredictor(file_name, tags_name)
        call C_PredictorRegisterInputNode(pd_pack_arr(n_pd_arr)%pd_ml, pd_pack_arr(n_pd_arr)%in_node_name)
        call C_PredictorRegisterOutputNode(pd_pack_arr(n_pd_arr)%pd_ml, pd_pack_arr(n_pd_arr)%out_node_name)   
        print*, "ML predictor created."

    else
        print*, "unsupported n_ml_options, the value is: ", n_ml_options
        stop
    end if

    pd_pack_arr(n_pd_arr)%n_pd_created = 1

  end subroutine Init_ML_Predictor

  subroutine Finalize_ML_Predictor(n_pd_arr)
      use ML_Predictor
      ! use iso_c_binding, only: C_ptr
      implicit none
      integer, intent(in) :: n_pd_arr
      if ( pd_pack_arr(n_pd_arr)%n_pd_created == 0) then
          print*, 'Predictor has not been created, no need to finalize'
          return
      end if
      call C_deletePredictor(pd_pack_arr(n_pd_arr)%pd_ml)
      
  end subroutine Finalize_ML_Predictor

  subroutine Allocate_ML_Arrays(n_pd_arr, jdim, kdim, idim)
      use ML_Predictor
      ! use iso_c_binding, only: C_ptr
      implicit none
      integer, intent(in) :: n_pd_arr
      integer, intent(in) :: jdim, kdim, idim
      integer :: lallo
      integer :: n_data

      if ( pd_pack_arr(n_pd_arr)%n_arr_created /= 0) then
          print*, 'ML Arrays has already been created'
          return
      end if

      ! n_data = (jdim + 1) * (kdim + 1) * (idim + 1) 
      ! n_data = (jdim - 1) * (kdim - 1) * (idim - 1)
      n_data = jdim  * kdim  * idim 

      allocate(pd_pack_arr(n_pd_arr)%ml_input_q1(1:jdim,1:kdim,1:idim), &
               pd_pack_arr(n_pd_arr)%ml_input_q2(1:jdim,1:kdim,1:idim), &
               pd_pack_arr(n_pd_arr)%ml_output(1:jdim,1:kdim,1:idim),stat=lallo)
      if(lallo.ne.0) stop ' !! error at allocating ml_input_q1, ml_input_q2, ml_output'
        !  ML Arrays LWS Added

      ! R_weight, Ndy, Ret
      allocate(pd_pack_arr(n_pd_arr)%R_weight(1:jdim,1:kdim,1:idim),   &
               pd_pack_arr(n_pd_arr)%Ndy(1:jdim,1:kdim,1:idim),        &
               pd_pack_arr(n_pd_arr)%Ret(1:jdim,1:kdim,1:idim),        &
               pd_pack_arr(n_pd_arr)%vist_rans(1:jdim,1:kdim,1:idim), stat=lallo)
      if(lallo.ne.0) stop ' !! error at allocating R_weight, Ndy, Ret, vist_rans'
      ! Temp arrays
      allocate(pd_pack_arr(n_pd_arr)%U_tmp(1:jdim,1:kdim,1:idim),        &
               pd_pack_arr(n_pd_arr)%V_tmp(1:jdim,1:kdim,1:idim),        &
               pd_pack_arr(n_pd_arr)%W_tmp(1:jdim,1:kdim,1:idim),        &
               pd_pack_arr(n_pd_arr)%k_tmp(1:jdim,1:kdim,1:idim),        &
               pd_pack_arr(n_pd_arr)%omega_tmp(1:jdim,1:kdim,1:idim),    &
               pd_pack_arr(n_pd_arr)%kmean_tmp(1:jdim,1:kdim,1:idim),  stat=lallo)
      if(lallo.ne.0) stop ' !! error at allocating U_tmp, V_tmp, W_tmp, k_tmp, omega_tmp, kmean_tmp'
      !  ML Arrays LWS Added
      allocate(pd_pack_arr(n_pd_arr)%ml_input_container(2, n_data), &
                pd_pack_arr(n_pd_arr)%ml_output_container(1, n_data) ,stat=lallo)
      if(lallo.ne.0) stop ' !! error at allocating ml_input_container, ml_output_container'
      
      pd_pack_arr(n_pd_arr)%R_weight=1.0
      pd_pack_arr(n_pd_arr)%Ndy=1.0
      pd_pack_arr(n_pd_arr)%Ret=1.0
      pd_pack_arr(n_pd_arr)%vist_rans=1.0

      pd_pack_arr(n_pd_arr)%U_tmp = 1.0
      pd_pack_arr(n_pd_arr)%V_tmp = 1.0
      pd_pack_arr(n_pd_arr)%W_tmp = 1.0
      pd_pack_arr(n_pd_arr)%k_tmp = 1.0
      pd_pack_arr(n_pd_arr)%omega_tmp = 1.0
      pd_pack_arr(n_pd_arr)%kmean_tmp = 1.0

      
      pd_pack_arr(n_pd_arr)%ml_input_q1=1.0
      pd_pack_arr(n_pd_arr)%ml_input_q2=1.0
      pd_pack_arr(n_pd_arr)%ml_output=1.0
      pd_pack_arr(n_pd_arr)%ml_input_container=1.0
      pd_pack_arr(n_pd_arr)%ml_output_container=1.0

      call C_PredictorSetDataCount(pd_pack_arr(n_pd_arr)%pd_ml, n_data);
      print *, 'test set predictor input data with empty container, in_node: ', pd_pack_arr(n_pd_arr)%in_node_name
      call C_PredictorSetNodeData(pd_pack_arr(n_pd_arr)%pd_ml, &
                                  pd_pack_arr(n_pd_arr)%in_node_name, &
                                  pd_pack_arr(n_pd_arr)%ml_input_container, &
                                  size(pd_pack_arr(n_pd_arr)%ml_input_container))
      print *, 'test run predictor with empty container...'
      call C_PredictorRun(pd_pack_arr(n_pd_arr)%pd_ml)
      print *, 'test get predictor input data with empty container, out_node: ', pd_pack_arr(n_pd_arr)%out_node_name
      call C_PredictorGetNodeData(pd_pack_arr(n_pd_arr)%pd_ml,&
                                  pd_pack_arr(n_pd_arr)%out_node_name, &
                                  pd_pack_arr(n_pd_arr)%ml_output_container, &
                                  size(pd_pack_arr(n_pd_arr)%ml_output_container))
      print *, 'test output: ', pd_pack_arr(n_pd_arr)%ml_output_container(1, 1),pd_pack_arr(n_pd_arr)%ml_output_container(1, 5), &
                                pd_pack_arr(n_pd_arr)%ml_output_container(1, n_data)

      pd_pack_arr(n_pd_arr)%n_arr_created = 1

  end subroutine Allocate_ML_Arrays

  subroutine Deallocate_ML_Arrays(n_pd_arr)
      use ML_Predictor
      ! use iso_c_binding, only: C_ptr
      implicit none
      integer, intent(in) :: n_pd_arr

      if ( pd_pack_arr(n_pd_arr)%n_arr_created == 0) then
          print*, 'ML Arrays has not been created, no need to deallocate'
          return
      end if

      deallocate(pd_pack_arr(n_pd_arr)%R_weight)
      deallocate(pd_pack_arr(n_pd_arr)%Ndy)
      deallocate(pd_pack_arr(n_pd_arr)%Ret)
      deallocate(pd_pack_arr(n_pd_arr)%vist_rans)

      deallocate(pd_pack_arr(n_pd_arr)%U_tmp)
      deallocate(pd_pack_arr(n_pd_arr)%V_tmp)
      deallocate(pd_pack_arr(n_pd_arr)%W_tmp)
      deallocate(pd_pack_arr(n_pd_arr)%k_tmp)
      deallocate(pd_pack_arr(n_pd_arr)%omega_tmp)
      deallocate(pd_pack_arr(n_pd_arr)%kmean_tmp)


      deallocate(pd_pack_arr(n_pd_arr)%ml_input_q1)
      deallocate(pd_pack_arr(n_pd_arr)%ml_input_q2)
      deallocate(pd_pack_arr(n_pd_arr)%ml_output)
      deallocate(pd_pack_arr(n_pd_arr)%ml_input_container)
      deallocate(pd_pack_arr(n_pd_arr)%ml_output_container)
      
  end subroutine Deallocate_ML_Arrays

  subroutine updateEddyViscosity(n_pd_arr, jdim, kdim, idim, nummem, q, zksav, smin, vist3d, Mach, Reue)
      use ML_Predictor
      ! use iso_c_binding, only: C_ptr
      implicit none
      integer, intent(in) :: n_pd_arr
      integer, intent(in) :: jdim, kdim, idim, nummem
      real, intent(in) :: Mach, Reue
      real, intent(in) :: q(jdim,kdim,idim,5)
      real, intent(in) :: zksav(jdim,kdim,idim,nummem)
      real, intent(in) :: smin(jdim-1,kdim-1,idim-1)
      real, intent(inout) :: vist3d(jdim,kdim,idim)

      ! real :: U_tmp(jdim,kdim,idim), V_tmp(jdim,kdim,idim), W_tmp(jdim,kdim,idim)
      ! real :: k_tmp(jdim,kdim,idim), omega_tmp(jdim,kdim,idim), kmean_tmp(jdim,kdim,idim)
      ! real :: vist_rans(jdim,kdim,idim)

      integer :: i, j, k, i_data

      real :: nu_inf
      real :: small_denominator

      if ( pd_pack_arr(n_pd_arr)%n_pd_created == 0) then
          print*, 'ML Predictor has not been created, please initiate the ML predictor'
          return
      end if

      if ( pd_pack_arr(n_pd_arr)%n_arr_created == 0) then
          print*, 'ML Arrays has not been created, please allocate the arrays with proper dimension'
          return
      end if

      
      nu_inf = Mach/Reue
      ! print*, 'Reue is ', Reue
      small_denominator = 1e-7

      ! print*, 'mlvars line 301 '

      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        pd_pack_arr(n_pd_arr)%U_tmp(j,k,i) = q(j,k,i,2)
        pd_pack_arr(n_pd_arr)%V_tmp(j,k,i) = q(j,k,i,3)
        pd_pack_arr(n_pd_arr)%W_tmp(j,k,i) = q(j,k,i,4)
        pd_pack_arr(n_pd_arr)%omega_tmp(j,k,i) = zksav(j,k,i,1)
        pd_pack_arr(n_pd_arr)%k_tmp(j,k,i) = zksav(j,k,i,2)
        ! pd_pack_arr(n_pd_arr)%Ndy(j,k,i) = Mach * sqrt(pd_pack_arr(n_pd_arr)%k_tmp(j,k,i)) &
        !                                   * smin(j,k,i) / (50.0 * nu_inf)
      enddo
      enddo
      enddo

      ! print*, 'mlvars line 317 '

      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        pd_pack_arr(n_pd_arr)%vist_rans(j,k,i) = vist3d(j,k,i)
      enddo
      enddo
      enddo

      ! print*, 'mlvars line 327 '

      

      pd_pack_arr(n_pd_arr)%kmean_tmp = 0.5 * (pd_pack_arr(n_pd_arr)%U_tmp**2 + &
                                               pd_pack_arr(n_pd_arr)%V_tmp**2 + &
                                               pd_pack_arr(n_pd_arr)%W_tmp**2)
      pd_pack_arr(n_pd_arr)%Ret = pd_pack_arr(n_pd_arr)%k_tmp / &
                                  (50.0 * pd_pack_arr(n_pd_arr)%omega_tmp + small_denominator)
      pd_pack_arr(n_pd_arr)%ml_input_q1 = 25.0 * pd_pack_arr(n_pd_arr)%k_tmp / &
                                          (pd_pack_arr(n_pd_arr)%kmean_tmp +  &
                                           25.0 * pd_pack_arr(n_pd_arr)%k_tmp + small_denominator)

      pd_pack_arr(n_pd_arr)%ml_input_q2 = pd_pack_arr(n_pd_arr)%k_tmp / &
                                         (pd_pack_arr(n_pd_arr)%k_tmp +  &
                                         50.0 * pd_pack_arr(n_pd_arr)%omega_tmp + small_denominator)


                          
      ! print*, 'mlvars line 346 '
      i_data = 1
      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        pd_pack_arr(n_pd_arr)%ml_input_container(1, i_data) = pd_pack_arr(n_pd_arr)%ml_input_q1(j,k,i)
        pd_pack_arr(n_pd_arr)%ml_input_container(2, i_data) = pd_pack_arr(n_pd_arr)%ml_input_q2(j,k,i)
        i_data = i_data + 1
      enddo
      enddo
      enddo

      ! print*, 'mlvars line 358 '

      call C_PredictorSetNodeData(pd_pack_arr(n_pd_arr)%pd_ml, &
                                  pd_pack_arr(n_pd_arr)%in_node_name, &
                                  pd_pack_arr(n_pd_arr)%ml_input_container, &
                                  size(pd_pack_arr(n_pd_arr)%ml_input_container))

      call C_PredictorRun(pd_pack_arr(n_pd_arr)%pd_ml)

      call C_PredictorGetNodeData(pd_pack_arr(n_pd_arr)%pd_ml,&
                                  pd_pack_arr(n_pd_arr)%out_node_name, &
                                  pd_pack_arr(n_pd_arr)%ml_output_container, &
                                  size(pd_pack_arr(n_pd_arr)%ml_output_container))

                                  
      ! print*, 'mlvars line 373 '
      i_data = 1
      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        pd_pack_arr(n_pd_arr)%ml_output(j,k,i) = pd_pack_arr(n_pd_arr)%ml_output_container(1, i_data)
        i_data = i_data + 1
      enddo
      enddo
      enddo
      ! print*, 'mlvars line 383 '

      pd_pack_arr(n_pd_arr)%ml_output = max(pd_pack_arr(n_pd_arr)%ml_output, 0.0)
      pd_pack_arr(n_pd_arr)%ml_output = min(pd_pack_arr(n_pd_arr)%ml_output, 0.95)

      ! print*, 'mlvars line 388 '


      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        vist3d(j,k,i) = 30.0 * pd_pack_arr(n_pd_arr)%Ret(j,k,i) &
                        * (pd_pack_arr(n_pd_arr)%ml_output(j,k,i) &
                          / (1.0 - pd_pack_arr(n_pd_arr)%ml_output(j,k,i)))
      enddo
      enddo
      enddo
      ! print*, 'mlvars line 400 '

      ! vist3d = 0.0
      
      
  end subroutine updateEddyViscosity

  subroutine exchangeEddyViscosity(n_pd_arr, jdim, kdim, idim, vist3d)
      implicit none
      integer, intent(in) :: n_pd_arr
      integer, intent(in) :: jdim, kdim, idim
      real, intent(inout) :: vist3d(jdim,kdim,idim)
      integer :: i, j, k

      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        vist3d(j,k,i) = pd_pack_arr(n_pd_arr)%vist_rans(j,k,i)
      enddo
      enddo
      enddo
  end subroutine exchangeEddyViscosity

    subroutine initvist3d_ml(n_pd_arr, jdim, kdim, idim, vist3d)
      implicit none
      integer, intent(in) :: n_pd_arr
      integer, intent(in) :: jdim, kdim, idim
      real, intent(in) :: vist3d(jdim,kdim,idim)
      integer :: i, j, k
      do i=1,idim
      do k=1,kdim
      do j=1,jdim
        pd_pack_arr(n_pd_arr)%vist_rans(j,k,i) = vist3d(j,k,i)
      enddo
      enddo
      enddo
  end subroutine initvist3d_ml
  !
end module mlvars
!+---------------------------------------------------------------------+
!| The end of the module commarray.                                    |
!+---------------------------------------------------------------------+