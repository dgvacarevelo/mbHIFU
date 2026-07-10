!>
!! @file
!! @brief Contains module @ref m_bubbles_el "m_bubbles_EL"

#:include 'macros.fpp'

!> @brief Tracks Lagrangian bubbles and couples their dynamics to the Eulerian flow via volume averaging
module m_bubbles_EL

    use m_global_parameters
    use m_mpi_proxy
    use m_bubbles_EL_kernels
    use m_bubbles
    use m_variables_conversion
    use m_compile_specific
    use m_boundary_common
    use m_helper_basic
    use m_sim_helpers
    use m_helper
    use m_constants, only: time_stepper_rk1, time_stepper_rk2, time_stepper_rk3, precision_single

    implicit none

    ! (nBub)
    integer, allocatable, dimension(:,:) :: lag_id      !< Global and local IDs
    real(wp), allocatable, dimension(:)  :: bub_R0      !< Initial bubble radius
    real(wp), allocatable, dimension(:)  :: Rmax_stats  !< Maximum radius
    real(wp), allocatable, dimension(:)  :: Rmin_stats  !< Minimum radius
    $:GPU_DECLARE(create='[lag_id, bub_R0, Rmax_stats, Rmin_stats]')

    real(wp), allocatable, dimension(:) :: gas_mg      !< Bubble's gas mass
    real(wp), allocatable, dimension(:) :: gas_betaT   !< heatflux model (Preston et al., 2007)
    real(wp), allocatable, dimension(:) :: gas_betaC   !< massflux model (Preston et al., 2007)
    real(wp), allocatable, dimension(:) :: bub_dphidt  !< subgrid velocity potential (Maeda & Colonius, 2018)
    $:GPU_DECLARE(create='[gas_mg, gas_betaT, gas_betaC, bub_dphidt]')

    ! (nBub, 1 -> actual val or 2 -> temp val)
    real(wp), allocatable, dimension(:,:) :: gas_p      !< Pressure in the bubble
    real(wp), allocatable, dimension(:,:) :: gas_mv     !< Vapor mass in the bubble
    real(wp), allocatable, dimension(:,:) :: intfc_rad  !< Bubble radius
    real(wp), allocatable, dimension(:,:) :: intfc_vel  !< Velocity of the bubble interface
    real(wp), allocatable, dimension(:,:) :: intfc_ac   !< Acceleration of the bubble interface
    $:GPU_DECLARE(create='[gas_p, gas_mv, intfc_rad, intfc_vel, intfc_ac]')
    ! (nBub, 1-> x or 2->y or 3 ->z, 1 -> actual or 2 -> temporal val)
    real(wp), allocatable, dimension(:,:,:) :: mtn_pos      !< Bubble's position
    real(wp), allocatable, dimension(:,:,:) :: mtn_posPrev  !< Bubble's previous position
    real(wp), allocatable, dimension(:,:,:) :: mtn_vel      !< Bubble's velocity
    real(wp), allocatable, dimension(:,:,:) :: mtn_s        !< Bubble's computational cell position in real format
    $:GPU_DECLARE(create='[mtn_pos, mtn_posPrev, mtn_vel, mtn_s]')
    ! (nBub, 1-> x or 2->y or 3 ->z, time-stage)
    real(wp), allocatable, dimension(:,:)   :: intfc_draddt  !< Time derivative of bubble's radius
    real(wp), allocatable, dimension(:,:)   :: intfc_dveldt  !< Time derivative of bubble's interface velocity
    real(wp), allocatable, dimension(:,:)   :: gas_dpdt      !< Time derivative of gas pressure
    real(wp), allocatable, dimension(:,:)   :: gas_dmvdt     !< Time derivative of the vapor mass in the bubble
    real(wp), allocatable, dimension(:,:,:) :: mtn_dposdt    !< Time derivative of the bubble's position
    real(wp), allocatable, dimension(:,:,:) :: mtn_dveldt    !< Time derivative of the bubble's velocity
    $:GPU_DECLARE(create='[intfc_draddt, intfc_dveldt, gas_dpdt, gas_dmvdt, mtn_dposdt, mtn_dveldt]')

    real(wp), allocatable, dimension(:)   :: bub_interact  !< Scattered pressure from each bubble
    real(wp), allocatable, dimension(:,:) :: bub_int_ids   !< Ids of the neighboring bubbles for pout interaction
    ! real(wp), allocatable, dimension(:) :: bub_lambda_c !< Mean inter-bubble distance (p' white noise) real(wp), allocatable,
    ! dimension(:, :) :: bub_rnd_phase !< Random phases (1:num_noise) per bubble (p' white noise)
    $:GPU_DECLARE(create='[bub_interact, bub_int_ids]')

    real(wp), allocatable, dimension(:,:) :: moments_bubs  !< Moments of volume, qac, qvis, qth_pos, qth_neg
    real(wp), allocatable, dimension(:)   :: acPw_bubs     !< Acoustic power of the bubbles (HIFU)
    real(wp), allocatable, dimension(:,:) :: mrmtnt_shell  !< Lipid shell indicator (Marmotant model)
    real(wp), allocatable, dimension(:)   :: mrmtnt_Rbuck  !< Buckling radius (Marmotant model)
    real(wp), allocatable, dimension(:)   :: mrmtnt_Rrupt  !< Rupture radius (Marmotant model)
    real(wp), allocatable, dimension(:)   :: bub_qvis      !< Time-averaged viscous intensity (HIFU)
    real(wp), allocatable, dimension(:)   :: bub_qth       !< Time-averaged thermal intensity (HIFU)
    real(wp), allocatable, dimension(:)   :: bub_hifu_rad  !< Time-averaged radius
    real(wp), allocatable, dimension(:)   :: bub_rho       !< Density at infinity per bub
    $:GPU_DECLARE(create='[mrmtnt_shell, mrmtnt_Rbuck, mrmtnt_Rrupt, bub_qvis, bub_qth, bub_hifu_rad, moments_bubs, acPw_bubs, bub_rho]')

    integer, private :: lag_num_ts  !< Number of time stages in the time-stepping scheme
    $:GPU_DECLARE(create='[lag_num_ts]')

    integer  :: nBubs                   !< Number of bubbles in the local domain
    real(wp) :: Rmax_glb, Rmin_glb      !< Maximum and minimum bubbe size in the local domain
    real(wp) :: Rmean_glb, lag_vol_glb  !< Mean size and sum of all bubbles' volume
    !> Projection of the lagrangian particles in the Eulerian framework
    type(vector_field) :: q_beta
    integer            :: q_beta_idx  !< Size of the q_beta vector field
    $:GPU_DECLARE(create='[nBubs, Rmax_glb, Rmin_glb, Rmean_glb, lag_vol_glb, q_beta, q_beta_idx]')

contains

    !> Initializes the lagrangian subgrid bubble solver
    !! @param q_cons_vf Initial conservative variables
    impure subroutine s_initialize_bubbles_EL_module(q_cons_vf, bc_type)

        type(scalar_field), dimension(sys_size), intent(inout)      :: q_cons_vf
        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        integer                                                     :: nBubs_glb, i, int_var

        ! Setting number of time-stages for selected time-stepping scheme

        lag_num_ts = time_stepper

        ! Allocate space for the Eulerian fields needed to map the effect of the bubbles
        if (lag_params%solver_approach == 1) then
            ! One-way coupling
            q_beta_idx = 3
        else if (lag_params%solver_approach == 2) then
            ! Two-way coupling
            q_beta_idx = 4
            if (p == 0) then
                ! Subgrid noise model for 2D approximation
                q_beta_idx = 6
            end if
        else
            call s_mpi_abort('Please check the lag_params%solver_approach input')
        end if

        $:GPU_UPDATE(device='[lag_num_ts, q_beta_idx]')

        @:ALLOCATE(q_beta%vf(1:q_beta_idx))

        do i = 1, q_beta_idx
            @:ALLOCATE(q_beta%vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
        end do

        @:ACC_SETUP_VFs(q_beta)

        ! Allocating space for lagrangian variables
        nBubs_glb = lag_params%nBubs_glb

        @:ALLOCATE(lag_id(1:nBubs_glb, 1:2))
        @:ALLOCATE(bub_R0(1:nBubs_glb))
        @:ALLOCATE(Rmax_stats(1:nBubs_glb))
        @:ALLOCATE(Rmin_stats(1:nBubs_glb))
        @:ALLOCATE(gas_mg(1:nBubs_glb))
        @:ALLOCATE(gas_betaT(1:nBubs_glb))
        @:ALLOCATE(gas_betaC(1:nBubs_glb))
        @:ALLOCATE(bub_dphidt(1:nBubs_glb))
        @:ALLOCATE(gas_p(1:nBubs_glb, 1:2))
        @:ALLOCATE(gas_mv(1:nBubs_glb, 1:2))
        @:ALLOCATE(intfc_rad(1:nBubs_glb, 1:2))
        @:ALLOCATE(intfc_vel(1:nBubs_glb, 1:2))
        @:ALLOCATE(intfc_ac(1:nBubs_glb, 1:2))
        @:ALLOCATE(mtn_pos(1:nBubs_glb, 1:3, 1:2))
        @:ALLOCATE(mtn_posPrev(1:nBubs_glb, 1:3, 1:2))
        @:ALLOCATE(mtn_vel(1:nBubs_glb, 1:3, 1:2))
        @:ALLOCATE(mtn_s(1:nBubs_glb, 1:3, 1:2))
        @:ALLOCATE(intfc_draddt(1:nBubs_glb, 1:lag_num_ts))
        @:ALLOCATE(intfc_dveldt(1:nBubs_glb, 1:lag_num_ts))
        @:ALLOCATE(gas_dpdt(1:nBubs_glb, 1:lag_num_ts))
        @:ALLOCATE(gas_dmvdt(1:nBubs_glb, 1:lag_num_ts))
        @:ALLOCATE(mtn_dposdt(1:nBubs_glb, 1:3, 1:lag_num_ts))
        @:ALLOCATE(mtn_dveldt(1:nBubs_glb, 1:3, 1:lag_num_ts))
        ! Marmotant model
        @:ALLOCATE(mrmtnt_shell(1:nBubs_glb, 1:2))
        @:ALLOCATE(mrmtnt_Rbuck(1:nBubs_glb))
        @:ALLOCATE(mrmtnt_Rrupt(1:nBubs_glb))

        ! hifu
        @:ALLOCATE(bub_qvis(1:nBubs_glb))
        @:ALLOCATE(bub_qth(1:nBubs_glb))
        @:ALLOCATE(bub_hifu_rad(1:nBubs_glb))
        @:ALLOCATE(bub_rho(1:nBubs_glb))

        if (hifu_params%moments) then
            @:ALLOCATE(moments_bubs(1:5, 1:4))
        end if
        if (hifu_params%power_balance) then
            @:ALLOCATE(acPw_bubs(1:3))
        end if

        ! Interbubble interaction 1: emitted Pout, 2: sum of Pouts from volume of influence (self-inclusive)
        @:ALLOCATE(bub_interact(1:nBubs_glb))
        ! 1: number of interacting bubbles (self-inclusive), 2:nBubs_glb+1: IDs in the volume of influence (self-inclusive)
        if (lag_params%pressure_corrector .and. any(lag_params%interaction_model == (/2, 3/))) then
            int_var = min(max_bub_int, nBubs_glb + 1)
            @:ALLOCATE(bub_int_ids(1:nBubs_glb, 1:int_var))
        end if
        !@:ALLOCATE(bub_lambda_c(1:nBubs_glb))
        !@:ALLOCATE(bub_rnd_phase(1:nBubs_glb, 1:num_noise))

        if (adap_dt .and. f_is_default(adap_dt_tol)) adap_dt_tol = dflt_adap_dt_tol

        ! call s_initialize_bubbles_EL_kernels()

        ! Starting bubbles
        call s_read_input_bubbles(q_cons_vf, bc_type)

    end subroutine s_initialize_bubbles_EL_module

    !> The purpose of this procedure is to start lagrange bubble parameters applying nondimensionalization if needed
    ! impure subroutine s_start_lagrange_inputs()

    ! integer :: id_bubbles, id_host real(wp) :: rho0, c0, T0, x0, p0

    ! id_bubbles = num_fluids id_host = num_fluids - 1

    ! !Reference values rho0 = lag_params%rho0 c0 = lag_params%c0 T0 = lag_params%T0 x0 = lag_params%x0 p0 = rho0*c0*c0

    ! !Update inputs Tw = lag_params%Thost/T0 pv = fluid_pp(id_host)%pv/p0 gamma_v = fluid_pp(id_host)%gamma_v gamma_n =
    ! fluid_pp(id_bubbles)%gamma_v k_vl = fluid_pp(id_host)%k_v*(T0/(x0*rho0*c0*c0*c0)) k_nl =
    ! fluid_pp(id_bubbles)%k_v*(T0/(x0*rho0*c0*c0*c0)) cp_v = fluid_pp(id_host)%cp_v*(T0/(c0*c0)) cp_n =
    ! fluid_pp(id_bubbles)%cp_v*(T0/(c0*c0)) R_v = (R_uni/fluid_pp(id_host)%M_v)*(T0/(c0*c0)) R_n =
    ! (R_uni/fluid_pp(id_bubbles)%M_v)*(T0/(c0*c0)) lag_params%diffcoefvap = lag_params%diffcoefvap/(x0*c0) ss =
    ! fluid_pp(id_host)%ss/(rho0*x0*c0*c0) mul0 = fluid_pp(id_host)%mul0/(rho0*x0*c0)

    ! !Marmotant model lag_params%ss0_ctdBub = lag_params%ss0_ctdBub/(rho0*x0*c0*c0) lag_params%srfDilVsc_ctdBub =
    ! lag_params%srfDilVsc_ctdBub/(rho0*x0*x0*c0) lag_params%srfElast_ctdBub = lag_params%srfElast_ctdBub/(rho0*x0*c0*c0)

    ! ! Parameters used in bubble_model Web = 1._wp/ss Re_inv = mul0

    ! if (polytropic) then Ca = (p0-pv)/(rho0*c0*c0) gamma_m = gamma_n if (thermal == 2) gamma_m = 1._wp ! Isothermal end if

    ! ! Need improvements to accept polytropic gas compression, isothermal and adiabatic thermal models, and ! the Gilmore and RP
    ! bubble models. ! polytropic = .false. ! Forcing no polytropic model ! thermal = 3 ! Forcing constant transfer coefficient
    ! model based on Preston et al., 2007 ! If Keller-Miksis model is not selected, then no radial motion

    !     !GPU vars get updated in initialize_gpu_vars

    ! end subroutine s_start_lagrange_inputs

    !> The purpose of this procedure is to obtain the initial bubbles' information
    !! @param q_cons_vf Conservative variables
    impure subroutine s_read_input_bubbles(q_cons_vf, bc_type)

        type(scalar_field), dimension(sys_size), intent(inout)      :: q_cons_vf
        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        real(wp), dimension(8)                                      :: inputBubble
        real(wp)                                                    :: qtime
        integer                                                     :: id, bub_id, save_count
        integer                                                     :: i, ios
        logical                                                     :: file_exist, indomain, read_flag
        real(wp)                                                    :: safeStop, tmp_val
        character(LEN=path_len + 2*name_len)                        :: path_D_dir, file_loc

        ! Initialize number of particles

        bub_id = 0
        id = 0
        safeStop = 0._wp

        ! Read the input lag_bubble file or restart point
        if (cfl_dt) then
            save_count = n_start
            qtime = n_start*t_save
        else
            save_count = t_step_start
            qtime = t_step_start*dt
        end if

        ! Read input file in the middle of a pure Euler simulation
        write (file_loc, '(a,i0,a)') 'lag_bubbles_', save_count, '.dat'
        file_loc = trim(case_dir) // '/restart_data' // trim(mpiiofs) // trim(file_loc)
        inquire (file=trim(file_loc), exist=file_exist)
        read_flag = .true.
        lag_params%initial_corrector = .true.
        if (file_exist) then
            read_flag = .false.
            lag_params%initial_corrector = .false.
        end if

        if (read_flag) then
            if (proc_rank == 0) print *, 'Reading lagrange bubbles input file.'
            call s_mpi_barrier()
            inquire (file='input/lag_bubbles.dat', exist=file_exist)
            if (file_exist) then
                open (94, file='input/lag_bubbles.dat', form='formatted', iostat=ios)
                do while (ios == 0)
                    read (94, *, iostat=ios) (inputBubble(i), i=1, 8)
                    if (ios /= 0) cycle
                    indomain = particle_in_domain_physical(inputBubble(1:3))
                    id = id + 1
                    if (indomain) then
                        bub_id = bub_id + 1
                        if (bub_id > lag_params%nBubs_glb) then
                            safeStop = 1._wp*bub_id
                        else
                            call s_add_bubbles(inputBubble, q_cons_vf, bub_id)
                            lag_id(bub_id, 1) = id  ! global ID
                            lag_id(bub_id, 2) = bub_id  ! local ID
                            nBubs = bub_id  ! local number of bubbles
                        end if
                    end if
                end do
                close (94)
            else
                call s_mpi_abort("Initialize the lagrange bubbles in input/lag_bubbles.dat")
            end if
        else
            if (proc_rank == 0) print *, 'Restarting lagrange bubbles at save_count: ', save_count
            call s_mpi_barrier()
            call s_restart_bubbles(bub_id, save_count)
        end if

        print '("Lagrange bubbles running, in proc ", I8, " number: ", I8, " / ", I8)', proc_rank, bub_id, id

        call s_mpi_barrier()
        if (num_procs > 1) then
            call s_mpi_allreduce_max(safeStop, tmp_val)
            safeStop = tmp_val
        end if

        if (int(safeStop) > lag_params%nBubs_glb) then
            if (proc_rank == 0) print '("Maximum number of bubbbles per processor is: ", I8)', int(safeStop)
            call s_mpi_abort('Current number of bubbles is larger than nBubs_glb.')
        else
            safeStop = nBubs
            if (num_procs > 1) then
                call s_mpi_allreduce_max(safeStop, tmp_val)
                safeStop = tmp_val
            end if
            if (proc_rank == 0) print '("Maximum number of bubbbles per processor is: ", I8)', int(safeStop)
        end if

        $:GPU_UPDATE(device='[bubbles_lagrange, lag_params]')

        $:GPU_UPDATE(device='[lag_id, bub_R0, Rmax_stats, Rmin_stats, gas_mg, gas_betaT, gas_betaC, bub_dphidt, gas_p, gas_mv, &
                     & intfc_rad, intfc_vel, mtn_pos, mtn_posPrev, mtn_vel, mtn_s, intfc_draddt, intfc_dveldt, gas_dpdt, &
                     & gas_dmvdt, mtn_dposdt, mtn_dveldt, nBubs]')

        $:GPU_UPDATE(device='[intfc_ac, mrmtnt_shell, mrmtnt_Rbuck, mrmtnt_Rrupt, bub_qvis, bub_qth, bub_hifu_rad, bub_interact]')

        Rmax_glb = min(dflt_real, -dflt_real)
        Rmin_glb = max(dflt_real, -dflt_real)
        Rmean_glb = 0._wp
        lag_vol_glb = 0._wp
        $:GPU_UPDATE(device='[Rmax_glb, Rmin_glb, Rmean_glb, lag_vol_glb]')

        $:GPU_UPDATE(device='[dx, dy, dz, x_cb, x_cc, y_cb, y_cc, z_cb, z_cc]')

        $:GPU_UPDATE(device='[ss_init, dil_vsc, el_vsc]')

        ! Populate temporal variables
        call s_transfer_data_to_tmp
        call s_start_bubble_interaction
        call s_smear_voidfraction(bc_type)

        if (read_flag) then
            ! Create ./D directory
            write (path_D_dir, '(A,I0,A,I0)') trim(case_dir) // '/D'
            call my_inquire(path_D_dir, file_exist)
            if (.not. file_exist) call s_create_directory(trim(path_D_dir))
            call s_write_restart_lag_bubbles(save_count)  ! Needed for post_processing
        end if

        call s_calculate_lag_bubble_stats()
        if (lag_params%write_bubbles) call s_write_lag_particles(qtime, replace=.false.)
        call s_write_void_evol(qtime, replace=.false.)

    end subroutine s_read_input_bubbles

    !> Add a new bubble from input data for a fresh start
    impure subroutine s_add_bubbles(inputBubble, q_cons_vf, bub_id)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        real(wp), dimension(8), intent(in)                  :: inputBubble
        integer, intent(in)                                 :: bub_id
        integer                                             :: i
        real(wp)                                            :: pliq, volparticle, concvap, totalmass, kparticle, cpparticle
        real(wp)                                            :: omegaN_local, PeG, PeT, rhol, qv, gamma, pi_inf, dynP
        integer, dimension(3)                               :: cell
        real(wp), dimension(2)                              :: Re
        real(wp)                                            :: massflag, heatflag, Re_trans, Im_trans, Web_mod

        massflag = 0._wp
        heatflag = 0._wp
        if (lag_params%massTransfer_model) massflag = 1._wp
        if (lag_params%heatTransfer_model) heatflag = 1._wp

        bub_R0(bub_id) = inputBubble(7)
        Rmax_stats(bub_id) = min(dflt_real, -dflt_real)
        Rmin_stats(bub_id) = max(dflt_real, -dflt_real)
        bub_dphidt(bub_id) = 0._wp
        intfc_rad(bub_id, 1) = inputBubble(7)
        intfc_vel(bub_id, 1) = inputBubble(8)
        intfc_ac(bub_id, 1) = 0._wp
        mtn_pos(bub_id,1:3,1) = inputBubble(1:3)
        mtn_posPrev(bub_id,1:3,1) = mtn_pos(bub_id,1:3,1)
        mtn_vel(bub_id,1:3,1) = inputBubble(4:6)
        bub_qvis(bub_id) = 0._wp
        bub_qth(bub_id) = 0._wp
        bub_hifu_rad(bub_id) = 0._wp
        bub_interact(bub_id) = 0._wp

        if (cyl_coord .and. p == 0) then
            mtn_pos(bub_id, 2, 1) = sqrt(mtn_pos(bub_id, 2, 1)**2._wp + mtn_pos(bub_id, 3, 1)**2._wp)
            ! Storing azimuthal angle (-Pi to Pi)) into the third coordinate variable
            mtn_pos(bub_id, 3, 1) = atan2(inputBubble(3), inputBubble(2))
            ! mtn_posPrev(bub_id,1:3,1) = mtn_pos(bub_id,1:3,1) ! Need 3D coords for hifu heat solver
        end if

        cell = -buff_size
        call s_locate_cell(mtn_pos(bub_id,1:3,1), cell, mtn_s(bub_id,1:3,1))

        ! Check if the bubble is located in the ghost cell of a symmetric, or wall boundary
        if ((any(bc_x%beg == (/BC_REFLECTIVE, BC_CHAR_SLIP_WALL, BC_SLIP_WALL, &
            & BC_NO_SLIP_WALL/)) .and. cell(1) < 0) .or. (any(bc_x%end == (/BC_REFLECTIVE, BC_CHAR_SLIP_WALL, BC_SLIP_WALL, &
            & BC_NO_SLIP_WALL/)) .and. cell(1) > m) .or. (any(bc_y%beg == (/BC_REFLECTIVE, BC_CHAR_SLIP_WALL, BC_SLIP_WALL, &
            & BC_NO_SLIP_WALL/)) .and. cell(2) < 0) .or. (any(bc_y%end == (/BC_REFLECTIVE, BC_CHAR_SLIP_WALL, BC_SLIP_WALL, &
            & BC_NO_SLIP_WALL/)) .and. cell(2) > n)) then
            call s_mpi_abort("Lagrange bubble is in the ghost cells of a symmetric or wall boundary.")
        end if

        if (p > 0) then
            if ((any(bc_z%beg == (/BC_REFLECTIVE, BC_CHAR_SLIP_WALL, BC_SLIP_WALL, &
                & BC_NO_SLIP_WALL/)) .and. cell(3) < 0) .or. (any(bc_z%end == (/BC_REFLECTIVE, BC_CHAR_SLIP_WALL, BC_SLIP_WALL, &
                & BC_NO_SLIP_WALL/)) .and. cell(3) > p)) then
                call s_mpi_abort("Lagrange bubble is in the ghost cells of a symmetric or wall boundary.")
            end if
        end if

        call s_convert_to_mixture_variables(q_cons_vf, cell(1), cell(2), cell(3), rhol, gamma, pi_inf, qv, Re)
        dynP = 0._wp
        do i = 1, num_dims
            dynP = dynP + 0.5_wp*q_cons_vf(eqn_idx%cont%end + i)%sf(cell(1), cell(2), cell(3))**2/rhol
        end do
        if (.not. f_is_default(acoustic_bc_params%Pbase)) then
            pliq = acoustic_bc_params%Pbase
        else
            pliq = (q_cons_vf(eqn_idx%E)%sf(cell(1), cell(2), cell(3)) - dynP - pi_inf)/gamma
        end if
        if (pliq < 0) print *, "Negative pressure", proc_rank, q_cons_vf(eqn_idx%E)%sf(cell(1), cell(2), cell(3)), pi_inf, gamma, &
            & pliq, cell, dynP

        ! Activate or deactivate the mass model
        massflag = 0._wp
        ! if (lag_params%coatedBub_model .or. lag_params%massTransfer_model) then
        if (lag_params%massTransfer_model) then
            ! Assume vapor and gas is present in bubble
            massflag = 1._wp
        end if

        ! Marmotant model parameters
        mrmtnt_shell(bub_id, 1) = 0._wp
        if (lag_params%coatedBub_model) mrmtnt_shell(bub_id, 1) = 1._wp
        mrmtnt_Rbuck(bub_id) = mrmtnt_shell(bub_id, 1)*bub_R0(bub_id)/sqrt(1._wp + ss_init/el_vsc)
        mrmtnt_Rrupt(bub_id) = mrmtnt_Rbuck(bub_id)*sqrt(1._wp + ss/el_vsc)

        ! Initial particle pressure
        gas_p(bub_id, 1) = pliq + 2._wp*(1._wp/Web)/bub_R0(bub_id)
        if (lag_params%coatedBub_model) then
            gas_p(bub_id, 1) = pliq + 2._wp*(ss_init)/bub_R0(bub_id)
            ! print *, 'Rbuck and Rrupt', mrmtnt_Rbuck(bub_id), mrmtnt_Rrupt(bub_id), bub_id
        end if
        if (pv*(massflag) > gas_p(bub_id, 1)) then
            print *, proc_rank, gas_p(bub_id, 1), pv*(massflag), pliq
            call s_mpi_abort("Lagrange bubble initially located in a region with pressure below the vapor pressure.")
        end if

        ! Initial particle mass
        volparticle = 4._wp/3._wp*pi*bub_R0(bub_id)**3._wp  ! volume
        gas_mv(bub_id, 1) = pv*volparticle*(1._wp/(R_v*Tw))*(massflag)  ! vapermass
        gas_mg(bub_id) = (gas_p(bub_id, 1) - pv*(massflag))*volparticle*(1._wp/(R_g*Tw))  ! gasmass
        if (gas_mg(bub_id) <= 0._wp) then
            print *, gas_p(bub_id, 1), pv, massflag, R_v, Tw, volparticle, R_g, bub_R0(bub_id)
            call s_mpi_abort("The initial mass of gas inside the bubble is negative. Check the initial conditions.")
        end if
        totalmass = gas_mg(bub_id) + gas_mv(bub_id, 1)  ! totalmass

        ! Bubble natural frequency
        concvap = gas_mv(bub_id, 1)/(gas_mv(bub_id, 1) + gas_mg(bub_id))
        omegaN_local = (3._wp*(gas_p(bub_id, 1) - pv*(massflag)) + 4._wp*(1._wp/Web)/bub_R0(bub_id))/rhol
        if (lag_params%coatedBub_model) then
            omegaN_local = (3._wp*(gas_p(bub_id, 1) - pv*(massflag)) + 4._wp*(ss_init)/bub_R0(bub_id))/rhol
        end if
        omegaN_local = sqrt(omegaN_local/bub_R0(bub_id)**2._wp)

        cpparticle = concvap*cp_v + (1._wp - concvap)*cp_g
        kparticle = concvap*k_vl + (1._wp - concvap)*k_gl

        ! Mass and heat transfer coefficients (based on Preston 2007)
        PeT = totalmass/volparticle*cpparticle*bub_R0(bub_id)**2._wp*omegaN_local/kparticle
        call s_transcoeff(1._wp, PeT, Re_trans, Im_trans)
        gas_betaT(bub_id) = Re_trans*kparticle

        PeG = bub_R0(bub_id)**2._wp*omegaN_local/vd
        call s_transcoeff(1._wp, PeG, Re_trans, Im_trans)
        gas_betaC(bub_id) = Re_trans*vd

        if (polytropic) then
            gas_p(bub_id, 2) = gas_p(bub_id, 1)
        else
            if (gas_betaT(bub_id) /= gas_betaT(bub_id) .or. gas_betaC(bub_id) /= gas_betaC(bub_id)) then
                print *, bub_id, gas_betaT(bub_id), gas_betaC(bub_id)
                call s_mpi_abort("NaN mass and heat transfer coefficients")
            end if
        end if

    end subroutine s_add_bubbles

    subroutine s_initial_pressure_correction(q_prim_vf, bc_type)

        type(scalar_field), dimension(sys_size), intent(in)         :: q_prim_vf
        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        real(wp)                                                    :: pinf, aux1, aux2, massflag, volparticle
        real(wp)                                                    :: concvap, totalmass, kparticle, cpparticle
        real(wp)                                                    :: omegaN, PeG, PeT, cson, rhol, Re_trans, Im_trans
        real(wp)                                                    :: gamma, pi_inf, qv, myRcell
        real(wp), dimension(eqn_idx%cont%end)                       :: myalpha_rho, myalpha
        real(wp), dimension(2)                                      :: Re
        integer, dimension(3)                                       :: cell
        integer                                                     :: i, k
        complex(wp)                                                 :: imag, trans, c1, c2, c3
        real(wp)                                                    :: qtime
        integer                                                     :: save_count

        lag_params%initial_corrector = .false.

        if (lag_params%cluster_type /= 1) then
            if (proc_rank == 0) print *, 'Performing s_initial_pressure_correction '

            $:GPU_PARALLEL_LOOP(private='[k, myalpha_rho, myalpha, Re, cell]')
            do k = 1, nBubs
                ! Obtaining driving pressure
                call s_get_pinf(k, q_prim_vf, 1, pinf, cell, aux1, aux2, myRcell)

                ! Obtain liquid density and computing speed of sound from pinf
                $:GPU_LOOP(parallelism='[seq]')
                do i = 1, num_fluids
                    myalpha_rho(i) = q_prim_vf(i)%sf(cell(1), cell(2), cell(3))
                    myalpha(i) = q_prim_vf(eqn_idx%E + i)%sf(cell(1), cell(2), cell(3))
                end do
                call s_convert_species_to_mixture_variables_acc(rhol, gamma, pi_inf, qv, myalpha, myalpha_rho, Re)
                call s_compute_cson_from_pinf(q_prim_vf, pinf, cell, rhol, gamma, pi_inf, cson)

                ! Activate or deactivate the mass model
                massflag = 0._wp
                if (lag_params%massTransfer_model) then
                    ! Assume vapor and gas is present in bubble
                    massflag = 1._wp
                end if

                ! Marmotant model parameters: no need correction

                ! Initial particle pressure
                gas_p(k, 1) = pinf + 2._wp*(1._wp/Web)/bub_R0(k)
                if (lag_params%coatedBub_model) then
                    gas_p(k, 1) = pinf + 2._wp*(ss_init)/bub_R0(k)
                end if
                if (polytropic) gas_p(k, 2) = gas_p(k, 1)

                ! Initial particle mass
                volparticle = 4._wp/3._wp*pi*bub_R0(k)**3._wp  ! volume
                gas_mv(k, 1) = pv*volparticle*(1._wp/(R_v*Tw))*(massflag)  ! vapermass
                gas_mg(k) = (gas_p(k, 1) - pv*(massflag))*volparticle*(1._wp/(R_g*Tw))  ! gasmass
                totalmass = gas_mg(k) + gas_mv(k, 1)  ! totalmass

                ! Bubble natural frequency
                concvap = gas_mv(k, 1)/(gas_mv(k, 1) + gas_mg(k))
                omegaN = (3._wp*(gas_p(k, 1) - pv*(massflag)) + 4._wp*(1._wp/Web)/bub_R0(k))/rhol
                if (lag_params%coatedBub_model) then
                    omegaN = (3._wp*(gas_p(k, 1) - pv*(massflag)) + 4._wp*(ss_init)/bub_R0(k))/rhol
                end if

                omegaN = sqrt(omegaN/bub_R0(k)**2._wp)
                cpparticle = concvap*cp_v + (1._wp - concvap)*cp_g
                kparticle = concvap*k_vl + (1._wp - concvap)*k_gl

                ! Mass and heat transfer coefficients (based on Preston 2007)
                PeT = totalmass/volparticle*cpparticle*bub_R0(k)**2._wp*omegaN/kparticle
                imag = (0._wp, 1._wp)
                c1 = imag*PeT
                c2 = sqrt(c1)
                c3 = (exp(c2) - exp(-c2))/(exp(c2) + exp(-c2))  ! tanh(c2)
                trans = ((c2/c3 - 1._wp)**(-1) - 3._wp/c1)**(-1)  ! transfer function
                Re_trans = trans
                Im_trans = aimag(trans)
                gas_betaT(k) = Re_trans*kparticle

                PeG = bub_R0(k)**2._wp*omegaN/vd
                c1 = imag*PeG
                c2 = sqrt(c1)
                c3 = (exp(c2) - exp(-c2))/(exp(c2) + exp(-c2))  ! tanh(c2)
                trans = ((c2/c3 - 1._wp)**(-1) - 3._wp/c1)**(-1)  ! transfer function
                Re_trans = trans
                Im_trans = aimag(trans)
                gas_betaC(k) = Re_trans*vd

                bub_interact(k) = pinf
            end do
            $:END_GPU_PARALLEL_LOOP()

            call s_transfer_data_to_tmp
            $:GPU_UPDATE(host='[gas_p, gas_mv, gas_mg, gas_betaT, gas_betaC, bub_interact]')

            call s_smear_voidfraction(bc_type)

            ! Replace files
            if (cfl_dt) then
                save_count = n_start
                qtime = n_start*t_save
            else
                save_count = t_step_start
                qtime = t_step_start*dt
            end if

            call s_calculate_lag_bubble_stats()
            if (lag_params%write_bubbles) call s_write_lag_particles(qtime, replace=.true.)
            call s_write_restart_lag_bubbles(save_count)  ! Needed for post_processing
            call s_write_void_evol(qtime, replace=.true.)

            ! call s_mpi_barrier()
        end if

    end subroutine s_initial_pressure_correction

    !> Restore bubble data from a restart file
    impure subroutine s_restart_bubbles(bub_id, save_count)

        integer, intent(inout)               :: bub_id, save_count
        character(LEN=path_len + 2*name_len) :: file_loc
        real(wp)                             :: file_time, file_dt
        integer                              :: file_num_procs, file_tot_part, tot_part

#ifdef MFC_MPI
        real(wp), dimension(lag_io_vars)       :: inputvals
        integer, dimension(MPI_STATUS_SIZE)    :: status
        integer(kind=MPI_OFFSET_KIND)          :: disp
        integer                                :: view
        integer, dimension(3)                  :: cell
        logical                                :: indomain, particle_file, file_exist
        integer, dimension(2)                  :: gsizes, lsizes, start_idx_part
        integer                                :: ifile, ierr, tot_data, id
        integer                                :: i
        integer, dimension(:), allocatable     :: proc_bubble_counts
        real(wp), dimension(1:1,1:lag_io_vars) :: dummy

        dummy = 0._wp

        ! Construct file path
        write (file_loc, '(A,I0,A)') 'lag_bubbles_', save_count, '.dat'
        file_loc = trim(case_dir) // '/restart_data' // trim(mpiiofs) // trim(file_loc)

        ! Check if file exists
        inquire (FILE=trim(file_loc), EXIST=file_exist)
        if (.not. file_exist) then
            call s_mpi_abort('Restart file ' // trim(file_loc) // ' does not exist!')
        end if

        if (.not. parallel_io) return

        if (proc_rank == 0) then
            call MPI_FILE_OPEN(MPI_COMM_SELF, file_loc, MPI_MODE_RDONLY, mpi_info_int, ifile, ierr)

            call MPI_FILE_READ(ifile, file_tot_part, 1, MPI_INTEGER, status, ierr)
            call MPI_FILE_READ(ifile, file_time, 1, mpi_p, status, ierr)
            call MPI_FILE_READ(ifile, file_dt, 1, mpi_p, status, ierr)
            call MPI_FILE_READ(ifile, file_num_procs, 1, MPI_INTEGER, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)
        end if

        call MPI_BCAST(file_tot_part, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(file_time, 1, mpi_p, 0, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(file_dt, 1, mpi_p, 0, MPI_COMM_WORLD, ierr)
        call MPI_BCAST(file_num_procs, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

        allocate (proc_bubble_counts(file_num_procs))

        if (proc_rank == 0) then
            call MPI_FILE_OPEN(MPI_COMM_SELF, file_loc, MPI_MODE_RDONLY, mpi_info_int, ifile, ierr)

            ! Skip to processor counts position
            disp = int(sizeof(file_tot_part) + 2*sizeof(file_time) + sizeof(file_num_procs), MPI_OFFSET_KIND)
            call MPI_FILE_SEEK(ifile, disp, MPI_SEEK_SET, ierr)
            call MPI_FILE_READ(ifile, proc_bubble_counts, file_num_procs, MPI_INTEGER, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)
        end if

        call MPI_BCAST(proc_bubble_counts, file_num_procs, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

        ! Set time variables from file
        mytime = file_time
        dt = file_dt

        bub_id = proc_bubble_counts(proc_rank + 1)

        start_idx_part(1) = 0
        do i = 1, proc_rank
            start_idx_part(1) = start_idx_part(1) + proc_bubble_counts(i)
        end do

        start_idx_part(2) = 0
        lsizes(1) = bub_id
        lsizes(2) = lag_io_vars

        gsizes(1) = file_tot_part
        gsizes(2) = lag_io_vars

        if (bub_id > 0) then
            allocate (MPI_IO_DATA_lag_bubbles(bub_id,1:lag_io_vars))

            call MPI_TYPE_CREATE_SUBARRAY(2, gsizes, lsizes, start_idx_part, MPI_ORDER_FORTRAN, mpi_p, view, ierr)
            call MPI_TYPE_COMMIT(view, ierr)

            call MPI_FILE_OPEN(MPI_COMM_WORLD, file_loc, MPI_MODE_RDONLY, mpi_info_int, ifile, ierr)

            ! Skip extended header
            disp = int(sizeof(file_tot_part) + 2*sizeof(file_time) + sizeof(file_num_procs) &
                       & + file_num_procs*sizeof(proc_bubble_counts(1)), MPI_OFFSET_KIND)
            call MPI_FILE_SET_VIEW(ifile, disp, mpi_p, view, 'native', mpi_info_int, ierr)

            call MPI_FILE_READ_ALL(ifile, MPI_IO_DATA_lag_bubbles, lag_io_vars*bub_id, mpi_p, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)
            call MPI_TYPE_FREE(view, ierr)

            nBubs = bub_id

            do i = 1, bub_id
                lag_id(i, 1) = int(MPI_IO_DATA_lag_bubbles(i, 1))
                lag_id(i, 2) = i
                mtn_pos(i,1:3,1) = MPI_IO_DATA_lag_bubbles(i,2:4)
                mtn_posPrev(i,1:3,1) = MPI_IO_DATA_lag_bubbles(i,5:7)
                mtn_vel(i,1:3,1) = MPI_IO_DATA_lag_bubbles(i,8:10)
                intfc_rad(i, 1) = MPI_IO_DATA_lag_bubbles(i, 11)
                intfc_vel(i, 1) = MPI_IO_DATA_lag_bubbles(i, 12)
                bub_R0(i) = MPI_IO_DATA_lag_bubbles(i, 13)
                Rmax_stats(i) = MPI_IO_DATA_lag_bubbles(i, 14)
                Rmin_stats(i) = MPI_IO_DATA_lag_bubbles(i, 15)
                bub_dphidt(i) = MPI_IO_DATA_lag_bubbles(i, 16)
                gas_p(i, 1) = MPI_IO_DATA_lag_bubbles(i, 17)
                gas_mv(i, 1) = MPI_IO_DATA_lag_bubbles(i, 18)
                gas_mg(i) = MPI_IO_DATA_lag_bubbles(i, 19)
                gas_betaT(i) = MPI_IO_DATA_lag_bubbles(i, 20)
                gas_betaC(i) = MPI_IO_DATA_lag_bubbles(i, 21)
                ! Marmotant
                mrmtnt_shell(bub_id, 1) = MPI_IO_DATA_lag_bubbles(i, 22)
                mrmtnt_Rbuck(bub_id) = MPI_IO_DATA_lag_bubbles(i, 23)
                mrmtnt_Rrupt(bub_id) = MPI_IO_DATA_lag_bubbles(i, 24)
                ! hifu
                bub_qvis(bub_id) = MPI_IO_DATA_lag_bubbles(i, 25)
                bub_qth(bub_id) = MPI_IO_DATA_lag_bubbles(i, 26)
                intfc_ac(bub_id, 1) = MPI_IO_DATA_lag_bubbles(i, 27)
                bub_hifu_rad(bub_id) = MPI_IO_DATA_lag_bubbles(i, 28)

                bub_interact(bub_id) = 0._wp
                if (polytropic) then
                    gas_p(bub_id, 2) = gas_p(bub_id, 1)
                    gas_p(bub_id, 1) = pv + (gas_p(bub_id, 2) - pv)*(bub_R0(bub_id)/intfc_rad(bub_id, 1))**(3._wp*gam_m)
                end if
                cell = -buff_size
                call s_locate_cell(mtn_pos(i,1:3,1), cell, mtn_s(i,1:3,1))
            end do

            deallocate (MPI_IO_DATA_lag_bubbles)
        else
            nBubs = 0

            call MPI_TYPE_CONTIGUOUS(0, mpi_p, view, ierr)
            call MPI_TYPE_COMMIT(view, ierr)

            call MPI_FILE_OPEN(MPI_COMM_WORLD, file_loc, MPI_MODE_RDONLY, mpi_info_int, ifile, ierr)

            ! Skip extended header
            disp = int(sizeof(file_tot_part) + 2*sizeof(file_time) + sizeof(file_num_procs) &
                       & + file_num_procs*sizeof(proc_bubble_counts(1)), MPI_OFFSET_KIND)
            call MPI_FILE_SET_VIEW(ifile, disp, mpi_p, view, 'native', mpi_info_int, ierr)

            call MPI_FILE_READ_ALL(ifile, dummy, 0, mpi_p, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)
            call MPI_TYPE_FREE(view, ierr)
        end if

        if (proc_rank == 0) then
            write (*, '(A,I0,A,I0)') 'Read ', file_tot_part, ' particles from restart file at t_step = ', save_count
            write (*, '(A,E15.7,A,E15.7)') 'Restart time = ', mytime, ', dt = ', dt
        end if

        deallocate (proc_bubble_counts)
#endif

    end subroutine s_restart_bubbles

    !> 3D modeling of bubble interaction and 2D approximation. 3D: Obtain the identifiers of the bubbles within the volume of
    !! influence. Assume that the bubbles are smaller than the grid size always. (no need to re-run during simulation). 2D: Compute
    !! the local number density of each bubble for the p'_cell model as white noise. It is the number of bubbles per volume of
    !! mixture in the physical domain (not buffers).
    subroutine s_start_bubble_interaction()

        integer                :: i, j, k
        integer                :: nb_local
        real(wp)               :: xb_smear, xe_smear
        real(wp)               :: yb_smear, ye_smear
        real(wp)               :: zb_smear, ze_smear
        real(wp), dimension(3) :: scoord
        integer, dimension(3)  :: cell
        integer                :: smear_idx
        real(wp)               :: num_rn1, num_rn2, num_rn
        real(wp)               :: st_dev_rn, mean_rn
        real(wp)               :: safeStop, tmp_val

        if (.not. lag_params%pressure_corrector) return

        if (proc_rank == 0) print '("Influence volume (bubble interaction) in # of surrounding cells is ", I0)', &
            & lag_params%influence
        ! mean_rn = 0.5_wp*pi st_dev_rn = 1._wp

        safeStop = 0._wp
        $:GPU_PARALLEL_LOOP(private='[j, cell, scoord]', reduction='[[safeStop]]',reductionOp='[MAX]',copy='[safeStop]')
        do j = 1, nBubs
            ! Is the bubble in the physical domain? if (particle_in_domain_physical(mtn_pos(j, 1:3, 1))) then Find the cell location
            scoord = mtn_s(j,1:3,1)
            cell(:) = int(scoord(:))
            $:GPU_LOOP(parallelism='[seq]')
            do i = 1, num_dims
                if (scoord(i) < 0._wp) cell(i) = cell(i) - 1
            end do

            ! Define smearing boundaries Assuming that the cell is always larger than the bubble, then the smearing volume is
            ! constant (influence+1+influence)x(3+1+3).

            smear_idx = cell(1) - lag_params%influence - 1
            if (smear_idx < -buff_size - 1) then
                do while (smear_idx < -buff_size - 1)
                    smear_idx = smear_idx + 1
                end do
            end if
            xb_smear = x_cb(smear_idx)

            smear_idx = cell(1) + lag_params%influence
            if (smear_idx > m + buff_size) then
                do while (smear_idx > m + buff_size)
                    smear_idx = smear_idx - 1
                end do
            end if
            xe_smear = x_cb(smear_idx)

            smear_idx = cell(2) - lag_params%influence - 1
            if (smear_idx < -buff_size - 1 .and. .not. cyl_coord) then
                do while (smear_idx < -buff_size - 1)
                    smear_idx = smear_idx + 1
                end do
            end if
            if (smear_idx < -1 .and. cyl_coord) smear_idx = -1
            yb_smear = y_cb(smear_idx)

            smear_idx = cell(2) + lag_params%influence
            if (smear_idx > n + buff_size) then
                do while (smear_idx > n + buff_size)
                    smear_idx = smear_idx - 1
                end do
            end if
            ye_smear = y_cb(smear_idx)

            if (p > 0) then
                smear_idx = cell(3) - lag_params%influence - 1
                if (smear_idx < -buff_size - 1) then
                    do while (smear_idx < -buff_size - 1)
                        smear_idx = smear_idx + 1
                    end do
                end if
                zb_smear = z_cb(smear_idx)

                smear_idx = cell(3) + lag_params%influence
                if (smear_idx > p + buff_size) then
                    do while (smear_idx > p + buff_size)
                        smear_idx = smear_idx - 1
                    end do
                end if
                ze_smear = z_cb(smear_idx)
            end if

            if (any(lag_params%interaction_model == (/2, 3/)) .and. p == 0) then
                yb_smear = mtn_posPrev(j, 2, 1) - abs(xe_smear - xb_smear)
                ye_smear = mtn_posPrev(j, 2, 1) + abs(xe_smear - xb_smear)

                zb_smear = mtn_posPrev(j, 3, 1) - abs(xe_smear - xb_smear)
                ze_smear = mtn_posPrev(j, 3, 1) + abs(xe_smear - xb_smear)
            end if

            ! Find bubbles inside the boundaries
            nb_local = 0
            if (any(lag_params%interaction_model == (/2, 3/))) then
                $:GPU_LOOP(parallelism='[seq]')
                do k = 1, nBubs
                    if ((mtn_posPrev(k, 1, 1) < xe_smear) .and. (mtn_posPrev(k, 1, 1) >= xb_smear) .and. (mtn_posPrev(k, 2, &
                        & 1) < ye_smear) .and. (mtn_posPrev(k, 2, 1) >= yb_smear) .and. (mtn_posPrev(k, 3, &
                        & 1) < ze_smear) .and. (mtn_posPrev(k, 3, 1) >= zb_smear)) then
                        if (k /= j) then
                            nb_local = nb_local + 1
                            if (nb_local <= max_bub_int) then
                                bub_int_ids(j, nb_local + 1) = k
                            end if
                            safeStop = max(safeStop, 1._wp*nb_local)
                        end if
                    end if
                end do
            else
                $:GPU_LOOP(parallelism='[seq]')
                do k = 1, nBubs
                    if ((mtn_pos(k, 1, 1) < xe_smear) .and. (mtn_pos(k, 1, 1) >= xb_smear) .and. (mtn_pos(k, 2, &
                        & 1) < ye_smear) .and. (mtn_pos(k, 2, 1) >= yb_smear)) then
                        if (p > 0) then
                            ! if ((mtn_pos(k, 3, 1) < ze_smear) .and. (mtn_pos(k, 3, 1) >= zb_smear)) then nb_local = nb_local + 1
                            ! if (nb_local <= max_bub_int) then !bub_int_ids(j, nb_local + 1) = k end if safeStop = max(safeStop,
                            ! 1._wp*nb_local) end if
                        else
                            nb_local = nb_local + 1
                        end if
                    end if
                end do
            end if

            if (any(lag_params%interaction_model == (/2, 3/))) then
                ! Total number of interacting bubbles
                bub_int_ids(j, 1) = nb_local
            else
                ! Compute and update the mean inter-bubble distance lambda_c bub_lambda_c(j) = 1._wp/(nb_local**(1._wp/3._wp))
            end if

            ! ! Populate random phases (1:num_noise) ! Should they remain constant at every calculation? or should they be replaced
            ! every t_step? i = 1 do while (.true.)

            ! call random_number(num_rn1) num_rn1 = 1._wp - num_rn1 call random_number(num_rn2) num_rn2 = 1._wp - num_rn2

            !     num_rn = st_dev_rn*sqrt(-2._wp*log(num_rn1))*cos(2._wp*pi*num_rn2) + mean_rn

            ! if (num_rn >= 0._wp .and. num_rn <= 2._wp*pi) then bub_rnd_phase(j, i) = num_rn if (i == num_noise) exit i = i + 1 end
            ! if

            ! end do

            ! if (lag_id(j, 1) == 1) then print*, 'bub_lambda_c', bub_lambda_c(j) ! print*, 'bub_rnd_phase', bub_rnd_phase(j,
            ! 1:num_noise) end if
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (num_procs > 1) then
            call s_mpi_allreduce_max(safeStop, tmp_val)
            safeStop = tmp_val
        end if

        if (proc_rank == 0) print '("Maximum number of interacting bubbles is ", I0)', int(safeStop)

        if (safeStop > max_bub_int) then
            call s_mpi_abort('Failed getting interacting bubbles.')
        end if

        if (any(lag_params%interaction_model == (/2, 3/))) then
            $:GPU_UPDATE(host='[bub_int_ids]')
            if (lag_params%nBubs_glb < 100) then
                do j = 1, nBubs
                    if (bub_int_ids(j, 1) /= 0) then
                        print '(" (proc: ", I3, ") Bubble ", I5, " interacts with ", I5, " bubbles.")', proc_rank, j, &
                            & int(bub_int_ids(j, 1))
                    end if
                end do
            end if
        end if

    end subroutine s_start_bubble_interaction

    !> Contains the bubble dynamics subroutines.
    subroutine s_compute_bubble_EL_dynamics(q_prim_vf, stage)

        type(scalar_field), dimension(sys_size), intent(inout) :: q_prim_vf
        integer, intent(in)                                    :: stage
        real(wp)                                               :: myVapFlux
        real(wp)                                               :: preterm1, term2, paux, pint, Romega, term1_fac
        real(wp)                                               :: myR_m, mygamma_m, myPb, myMass_g, myMass_v
        real(wp)                                               :: myR, myV, myBeta_c, myBeta_t, myR0, myPbdot, myMvdot
        real(wp)                                               :: myPinf, aux1, aux2, myCson, myRho
        real(wp)                                               :: gamma, pi_inf, qv
        real(wp)                                               :: Rb, myConc_v, myPout, myInt, myShell, myRbuck, myRrupt, myAc
        real(wp)                                               :: myQth, myQvis, myRcell, myRmean, myKe, myVolmean

        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: myalpha_rho, myalpha
        #:else
            real(wp), dimension(num_fluids) :: myalpha_rho, myalpha
        #:endif
        real(wp), dimension(2) :: Re
        integer, dimension(3) :: cell
        real(wp) :: myTzPcell, myNoise_constant, myLambda_c, myloc, mydk, myPnoise, myLag_time
        real(wp), dimension(num_noise) :: myPhase
        integer :: adap_dt_stop_max, adap_dt_stop                     !< Fail-safe exit if max iteration count reached
        real(wp) :: dmalf, dmntait, dmBtait, dm_bub_adv_src, dm_divu  !< Dummy variables for unified subgrid bubble subroutines
        integer :: i, k, l
        real(wp), dimension(1:4) :: mom_vol, mom_qvis, mom_qth_p, mom_qth_n, mom_ke
        real(wp) :: fxb_Rc
        integer :: total_ids, bub_idx
        logical :: flg_bub_in_cv
        real(wp) :: acPw_qvis, acPw_qth, acPW_nbubs, acPw_ke
        real(wp) :: sum_qvis, sum_qth

        call nvtxStartRange("LAGRANGE-BUBBLE-DYNAMICS")

        sum_qvis = 0._wp; sum_qth = 0._wp

        if (hifu_params%moments) then
            mom_vol(1:4) = 0._wp; mom_qvis(1:4) = 0._wp; mom_ke(1:4) = 0._wp
            mom_qth_p(1:4) = 0._wp; mom_qth_n(1:4) = 0._wp
        end if
        if (hifu_params%power_balance) then
            acPw_qvis = 0._wp; acPw_qth = 0._wp
            acPW_nbubs = 0._wp; acPw_ke = 0._wp
        end if

        ! Subgrid p_inf model based on Maeda and Colonius (2018).
        if (lag_params%pressure_corrector .and. .not. adap_dt) then
            call s_calculate_scattered_pressure(q_prim_vf)
        end if

        ! Radial motion
        adap_dt_stop_max = 0
        $:GPU_PARALLEL_LOOP(private='[k, i, myalpha_rho, myalpha, Re, cell, myVapFlux, preterm1, term2, paux, pint, Romega, &
                            & term1_fac, myR_m, mygamma_m, myPb, myMass_g, myMass_v, myR, myV, myBeta_c, myBeta_t, myR0, myPbdot, &
                            & myMvdot, myPinf, aux1, aux2, myCson, myRho, gamma, pi_inf, qv, dmalf, dmntait, dmBtait, myAc, &
                            & myShell, myRbuck, myRrupt, dm_bub_adv_src, dm_divu, adap_dt_stop, fxb_Rc]', &
                            & reduction='[[adap_dt_stop_max], [mom_vol(1:4), mom_ke(1:4), mom_qvis(1:4), mom_qth_p(1:4), &
                            & mom_qth_n(1:4)], [acPw_qvis, acPw_qth, acPW_nbubs, acPw_ke, sum_qvis, sum_qth]]', &
                            & reductionOp='[MAX, +, +]', copy='[adap_dt_stop_max, mom_vol(1:4), mom_qvis(1:4), mom_qth_p(1:4), &
                            & mom_qth_n(1:4), acPw_qvis, acPw_qth, acPW_nbubs, acPw_ke, sum_qvis, sum_qth, mom_ke(1:4)]', copyin='[stage]')
        do k = 1, nBubs
            ! Keller-Miksis model

            ! Current bubble state
            myPb = gas_p(k, 2)
            myMass_g = gas_mg(k)
            myMass_v = gas_mv(k, 2)
            myR = intfc_rad(k, 2)
            myV = intfc_vel(k, 2)
            myBeta_c = gas_betaC(k)
            myBeta_t = gas_betaT(k)
            myR0 = bub_R0(k)
            myShell = mrmtnt_shell(k, 2)
            myRbuck = mrmtnt_Rbuck(k)
            myRrupt = mrmtnt_Rrupt(k)
            if (myR > myRrupt) myShell = 0._wp
            myLag_time = mytime - dt
            myPout = 0._wp  ! Self-scaterred pressure
            myInt = 0._wp  ! Interaction term from surrounding bubbles
            if (lag_params%pressure_corrector .and. .not. adap_dt) then
                ! if (any(lag_params%interaction_model == (/1, 3/)) .and. .not. adap_dt) then
                if (any(lag_params%interaction_model == (/1, 3/))) then
                    myPout = bub_interact(k)
                end if
                if (any(lag_params%interaction_model == (/2, 3/))) then
                    myInt = bub_interact(k)
                end if
            end if

            ! Vapor and heat fluxes
            if (.not. polytropic) then
                call s_vflux(myR, myV, myPb, myMass_v, k, myVapFlux, myMass_g, myBeta_c, myR_m, mygamma_m, myShell)
                myPbdot = f_bpres_dot(myVapFlux, myR, myV, myPb, myMass_v, k, myBeta_t, myR_m, mygamma_m, myShell)
                myMvdot = 4._wp*pi*myR**2._wp*myVapFlux
            else
                myVapFlux = 0._wp; myPbdot = 0._wp; myMvdot = 0._wp
            end if

            ! Retrieving driving pressure
            call s_get_pinf(k, q_prim_vf, 1, myPinf, cell, aux1, aux2, myRcell)

            ! Obtain liquid density and computing speed of sound from pinf
            call s_compute_species_fraction(q_prim_vf, cell(1), cell(2), cell(3), myalpha_rho, myalpha)
            call s_convert_species_to_mixture_variables_acc(myRho, gamma, pi_inf, qv, myalpha, myalpha_rho, Re)
            if (lag_params%pressure_corrector .and. any(lag_params%interaction_model == (/1, 3/)) .and. .not. adap_dt) then
                ! Kazuki's model to adjust Pinf

                myPinf = myPinf + myPout
                ! if (p==0) then ! White noise for 2D reduced model (myTzPcell is myPinf) myLambda_c = bub_lambda_c(k) myloc =
                ! mtn_s(k, 3, 2) !myPhase = bub_rnd_phase(k, 1:num_noise) call s_white_noise_constants(k, myLambda_c, q_prim_vf,
                ! cell, myPinf, myNoise_constant, mydk) call s_compute_cson_from_pinf(q_prim_vf, myPinf, cell, myRho, gamma, pi_inf,
                ! myCson) myPnoise = f_pres_stochastic(myPinf, myNoise_constant, myLambda_c, mydk, myloc, myLag_time, myCson) myPinf
                ! = myPinf + myPnoise*lag_params%pnoise_scale end if
            end if

            call s_compute_cson_from_pinf(q_prim_vf, myPinf, cell, myRho, gamma, pi_inf, myCson)
            bub_rho(k) = myRho  ! Used for moments of KE

            ! Adaptive time stepping
            if (adap_dt) then
                if (stage == 3) myLag_time = mytime - 0.5_wp*dt

                call s_advance_step(myRho, myPinf, myR, myV, myR0, myPb, myPbdot, dmalf, dmntait, dmBtait, dm_bub_adv_src, &
                                    & dm_divu, k, myMass_v, myMass_g, myBeta_c, myBeta_t, myCson, myInt, myShell, myRbuck, &
                                    & myRrupt, myRcell, myNoise_constant, myLambda_c, mydk, myloc, myLag_time, myAc, myQvis, &
                                    & myQth, myKe, myRmean, myVolmean, adap_dt_stop)

                ! Update bubble state
                intfc_rad(k, 1) = myR
                intfc_vel(k, 1) = myV
                intfc_ac(k, 1) = myAc
                gas_p(k, 1) = myPb
                if (polytropic) gas_p(k, 1) = pv + (myPb - pv)*(bub_R0(k)/myR)**(3._wp*gam_m)
                gas_mv(k, 1) = myMass_v
                mrmtnt_shell(k, 1) = myShell
                if (hifu_params%sampling) then
                    bub_qvis(k) = bub_qvis(k) + myQvis  !> Viscous damping of the bubble (Watts*second)
                    bub_qth(k) = bub_qth(k) + myQth     !> Thermal damping of the bubble (Watts*second)
                    bub_hifu_rad(k) = bub_hifu_rad(k) + myRmean !> Mean radius (m*second)
                    sum_qvis = sum_qvis + bub_qvis(k)
                    sum_qth = sum_qth + bub_qth(k)
                    if (hifu_params%moments) then
                        fxb_Rc = (mtn_pos(k, 1, 1) - hifu_params%cloud_center(1))/hifu_params%R_cloud

                        $:GPU_LOOP(parallelism='[seq]')
                        do i = 1, 4
                            mom_vol(i) = mom_vol(i) + (myVolmean/(0.5_wp*dt))*(fxb_Rc)**(i - 1)
                            mom_ke(i) = mom_ke(i) + (myKe/(0.5_wp*dt))*(fxb_Rc)**(i - 1)
                            mom_qvis(i) = mom_qvis(i) + (myQvis/(0.5_wp*dt))*(fxb_Rc)**(i - 1)
                            if (myQth < 0._wp) then
                                mom_qth_p(i) = mom_qth_p(i) + (myQth/(0.5_wp*dt))*(fxb_Rc)**(i - 1)
                            else
                                mom_qth_n(i) = mom_qth_n(i) + (myQth*(0.5_wp*dt))*(fxb_Rc)**(i - 1)
                            end if
                        end do
                    end if
                    if (hifu_params%power_balance) then
                        flg_bub_in_cv = f_bub_in_cv(mtn_pos(k,1:3,1))
                        if (flg_bub_in_cv) then
                            acPw_qvis = acPw_qvis + myQvis/(0.5_wp*dt)  ! (Watts)
                            acPw_qth = acPw_qth + myQth/(0.5_wp*dt)  ! (Watts)
                            acPW_nbubs = acPW_nbubs + 1._wp
                            acPw_ke = acPw_ke + myKe/(0.5_wp*dt)  ! (Watts)
                        end if
                    end if
                end if
            else
                ! Radial acceleration from bubble models
                intfc_dveldt(k, stage) = f_rddot(myRho, myPinf, myR, myV, myR0, myPb, myPbdot, dmalf, dmntait, dmBtait, &
                             & dm_bub_adv_src, dm_divu, myCson, myInt, myShell, myRbuck, myRcell)
                intfc_draddt(k, stage) = myV
                gas_dmvdt(k, stage) = myMvdot
                gas_dpdt(k, stage) = myPbdot
                mrmtnt_shell(k, 2) = myShell
            end if

            bub_interact(k) = myPinf  ! Need the pressure seen by the bubble to be part of the printed outputs
            ! Strang Splitting: P radiated by each bubble is internally calculated since it vary per substep, then it is not
            ! included in this var. No adap_dt: This term includes the radiated pressure.

            adap_dt_stop_max = max(adap_dt_stop_max, adap_dt_stop)
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (adap_dt .and. adap_dt_stop_max > 0) call s_mpi_abort("Adaptive time stepping failed to converge.")

        if (hifu_params%sampling .and. adap_dt) then
            if (hifu_params%moments) then
                if (stage == 3) then
                    do i = 1, 4
                        moments_bubs(1, i) = moments_bubs(1, i) + mom_qvis(i)
                        moments_bubs(2, i) = moments_bubs(2, i) + mom_qth_p(i)
                        moments_bubs(3, i) = moments_bubs(3, i) + mom_qth_n(i)
                        moments_bubs(4, i) = moments_bubs(4, i) + mom_vol(i)
                        moments_bubs(5, i) = moments_bubs(5, i) + mom_ke(i)
                    end do

                    do i = 1, 5
                        call s_write_moments(moments_bubs(i,1:4), idx=i)
                    end do
                else
                    do i = 1, 4
                        moments_bubs(1, i) = mom_qvis(i)
                        moments_bubs(2, i) = mom_qth_p(i)
                        moments_bubs(3, i) = mom_qth_n(i)
                        moments_bubs(4, i) = mom_vol(i)
                        moments_bubs(5, i) = mom_ke(i)
                    end do
                end if
            end if
            if (hifu_params%power_balance) then
                if (stage == 3) then
                    acPw_bubs(1) = 0.5_wp*acPw_bubs(1) + 0.5_wp*acPw_qvis
                    acPw_bubs(2) = 0.5_wp*acPw_bubs(2) + 0.5_wp*acPw_qth
                    acPw_bubs(3) = 0.5_wp*acPw_bubs(3) + 0.5_wp*acPw_ke
                    call s_write_power_balance_bubs(acPw_bubs(1), acPw_bubs(2), acPw_bubs(3), acPW_nbubs, dt)
                else
                    acPw_bubs(1) = acPw_qvis
                    acPw_bubs(2) = acPw_qth
                    acPw_bubs(3) = acPw_ke
                end if
            end if
            if (stage == 3) call s_sum_qbub(sum_qvis, sum_qth)
        end if

        ! Bubbles remain in a fixed position
        $:GPU_PARALLEL_LOOP(collapse=2, private='[k, l]', copyin='[stage]')
        do k = 1, nBubs
            do l = 1, 3
                mtn_dposdt(k, l, stage) = 0._wp
                mtn_dveldt(k, l, stage) = 0._wp
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        call nvtxEndRange

    end subroutine s_compute_bubble_EL_dynamics

    subroutine s_sum_qbub(sum_qvis, sum_qth)

        real(wp), intent(inout) :: sum_qvis, sum_qth
        real(wp)                :: var_glb, sum_nBubs
        character(len=512)      :: line

        sum_nBubs = nBubs*1._wp

        if (num_procs > 1) then
            call s_mpi_allreduce_sum(sum_qvis, var_glb)
            sum_qvis = var_glb

            call s_mpi_allreduce_sum(sum_qth, var_glb)
            sum_qth = var_glb

            call s_mpi_allreduce_sum(sum_nBubs, var_glb)
            sum_nBubs = var_glb
        end if

        if (proc_rank == 0) then
            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + dt, dt, &
                   & sum_qvis, sum_qth, sum_nBubs, 0._wp, 0._wp
            write (98, '(A)') trim(line)
        end if

    end subroutine s_sum_qbub

    function f_bub_in_cv(pos_part)

        $:GPU_ROUTINE(parallelism='[seq]')
        real(wp), dimension(3), intent(in) :: pos_part
        logical                            :: f_bub_in_cv

        f_bub_in_cv = ((pos_part(1) < hifu_params%cv_xe) .and. (pos_part(1) >= hifu_params%cv_xb) .and. (pos_part(2) &
                       & < hifu_params%cv_ye) .and. (pos_part(2) >= hifu_params%cv_yb) .and. (pos_part(3) < hifu_params%cv_ze) &
                       & .and. (pos_part(3) >= hifu_params%cv_zb))

    end function f_bub_in_cv

    subroutine s_write_power_balance_bubs(acPw_qvis, acPw_qth, acPw_ke, acPW_nbubs, hdid)

        real(wp), intent(inout) :: acPw_qvis, acPw_qth, acPw_ke, acPW_nbubs
        real(wp)                :: hdid
        real(wp)                :: var_glb
        integer                 :: i
        character(len=512)      :: line

        if (num_procs > 1) then
            call s_mpi_allreduce_sum(acPw_qvis, var_glb)
            acPw_qvis = var_glb

            call s_mpi_allreduce_sum(acPw_qth, var_glb)
            acPw_qth = var_glb

            call s_mpi_allreduce_sum(acPw_ke, var_glb)
            acPw_ke = var_glb

            call s_mpi_allreduce_sum(acPW_nbubs, var_glb)
            acPW_nbubs = var_glb
        end if

        if (proc_rank == 0) then
            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + hdid, hdid, &
                   & acPW_nbubs, acPw_qvis, acPw_qth, acPw_ke
            write (86, '(A)') trim(line)
        end if

    end subroutine s_write_power_balance_bubs

    !> Compute the Lagrangian bubble source terms and add them to the RHS
    subroutine s_compute_bubbles_EL_source(q_cons_vf, q_prim_vf, rhs_vf, bc_type)

        type(scalar_field), dimension(sys_size), intent(inout)      :: q_cons_vf
        type(scalar_field), dimension(sys_size), intent(inout)      :: q_prim_vf
        type(scalar_field), dimension(sys_size), intent(inout)      :: rhs_vf
        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        integer                                                     :: i, j, k, l

        if (.not. adap_dt) call s_smear_voidfraction(bc_type)

        if (lag_params%solver_approach == 2) then
            ! (q / (1 - beta)) * d(beta)/dt source
            if (p == 0 .and. .not. lag_params%newModel_2D) then
                $:GPU_PARALLEL_LOOP(private='[i, j, k, l]', collapse=4)
                do k = 0, p
                    do j = 0, n
                        do i = 0, m
                            do l = 1, eqn_idx%E
                                if (q_beta%vf(1)%sf(i, j, k) > (1._wp - lag_params%valmaxvoid)) then
                                    rhs_vf(l)%sf(i, j, k) = rhs_vf(l)%sf(i, j, k) + q_cons_vf(l)%sf(i, j, k)*(q_beta%vf(2)%sf(i, &
                                           & j, k) + q_beta%vf(5)%sf(i, j, k))
                                end if
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            else
                $:GPU_PARALLEL_LOOP(private='[i, j, k, l]', collapse=4)
                do k = 0, p
                    do j = 0, n
                        do i = 0, m
                            do l = 1, eqn_idx%E
                                if (q_beta%vf(1)%sf(i, j, k) > (1._wp - lag_params%valmaxvoid)) then
                                    rhs_vf(l)%sf(i, j, k) = rhs_vf(l)%sf(i, j, k) + q_cons_vf(l)%sf(i, j, k)/q_beta%vf(1)%sf(i, &
                                           & j, k)*q_beta%vf(2)%sf(i, j, k)
                                end if
                            end do
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end if

            do l = 1, num_dims
                call s_gradient_dir(q_prim_vf(eqn_idx%E)%sf, q_beta%vf(3)%sf, l)

                ! (q / (1 - beta)) * d(beta)/dt source
                $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
                do k = 0, p
                    do j = 0, n
                        do i = 0, m
                            if (q_beta%vf(1)%sf(i, j, k) > (1._wp - lag_params%valmaxvoid)) then
                                rhs_vf(eqn_idx%cont%end + l)%sf(i, j, k) = rhs_vf(eqn_idx%cont%end + l)%sf(i, j, &
                                       & k) - (1._wp - q_beta%vf(1)%sf(i, j, k))/q_beta%vf(1)%sf(i, j, k)*q_beta%vf(3)%sf(i, j, k)
                            end if
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()

                ! source in energy
                $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
                do k = idwbuff(3)%beg, idwbuff(3)%end
                    do j = idwbuff(2)%beg, idwbuff(2)%end
                        do i = idwbuff(1)%beg, idwbuff(1)%end
                            q_beta%vf(3)%sf(i, j, k) = q_prim_vf(eqn_idx%E)%sf(i, j, k)*q_prim_vf(eqn_idx%cont%end + l)%sf(i, j, k)
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()

                call s_gradient_dir(q_beta%vf(3)%sf, q_beta%vf(4)%sf, l)

                ! (beta / (1 - beta)) * d(Pu)/dl source
                $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
                do k = 0, p
                    do j = 0, n
                        do i = 0, m
                            if (q_beta%vf(1)%sf(i, j, k) > (1._wp - lag_params%valmaxvoid)) then
                                rhs_vf(eqn_idx%E)%sf(i, j, k) = rhs_vf(eqn_idx%E)%sf(i, j, k) - q_beta%vf(4)%sf(i, j, &
                                       & k)*(1._wp - q_beta%vf(1)%sf(i, j, k))/q_beta%vf(1)%sf(i, j, k)
                            end if
                        end do
                    end do
                end do
                $:END_GPU_PARALLEL_LOOP()
            end do
        end if

    end subroutine s_compute_bubbles_EL_source

    !> Compute the speed of sound from a given driving pressure
    subroutine s_compute_cson_from_pinf(q_prim_vf, pinf, cell, rhol, gamma, pi_inf, cson)

        $:GPU_ROUTINE(function_name='s_compute_cson_from_pinf', parallelism='[seq]', cray_inline=True)

        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        real(wp), intent(in)                                :: pinf, rhol, gamma, pi_inf
        integer, dimension(3), intent(in)                   :: cell
        real(wp), intent(out)                               :: cson
        real(wp)                                            :: E, H
        #:if not MFC_CASE_OPTIMIZATION and USING_AMD
            real(wp), dimension(3) :: vel
        #:else
            real(wp), dimension(num_dims) :: vel
        #:endif
        integer :: i

        vel(:) = 0._wp
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_dims
            vel(i) = q_prim_vf(i + eqn_idx%cont%end)%sf(cell(1), cell(2), cell(3))
        end do
        E = gamma*pinf + pi_inf + 0.5_wp*rhol*dot_product(vel, vel)
        H = (E + pinf)/rhol
        cson = sqrt((H - 0.5_wp*dot_product(vel, vel))/gamma)

    end subroutine s_compute_cson_from_pinf

    !> Smear the bubble effects onto the Eulerian grid
    subroutine s_smear_voidfraction(bc_type)

        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        integer                                                     :: i, j, k, l, nVar

        call nvtxStartRange("BUBBLES-LAGRANGE-KERNELS")

        $:GPU_PARALLEL_LOOP(private='[i, j, k, l]', collapse=4)
        do i = 1, q_beta_idx
            do l = idwbuff(3)%beg, idwbuff(3)%end
                do k = idwbuff(2)%beg, idwbuff(2)%end
                    do j = idwbuff(1)%beg, idwbuff(1)%end
                        q_beta%vf(i)%sf(j, k, l) = 0._wp
                    end do
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (lag_params%newModel_2D) then
            call s_smoothfunction(nBubs, intfc_rad, intfc_vel, mtn_s, mtn_posPrev, q_beta)
        else
            call s_smoothfunction(nBubs, intfc_rad, intfc_vel, mtn_s, mtn_pos, q_beta)
        end if

        ! Add effect of bubbles across processors
        if (num_procs > 1) then
            nVar = 1
            if (lag_params%solver_approach == 2) then
                nVar = 2
                if (p == 0) nVar = 3
            end if

            call s_populate_EL_buffers(q_beta, bc_type, nVar)
        end if

        ! Store 1-beta
        $:GPU_PARALLEL_LOOP(private='[j, k, l]', collapse=3)
        do l = idwbuff(3)%beg, idwbuff(3)%end
            do k = idwbuff(2)%beg, idwbuff(2)%end
                do j = idwbuff(1)%beg, idwbuff(1)%end
                    q_beta%vf(1)%sf(j, k, l) = 1._wp - q_beta%vf(1)%sf(j, k, l)
                    ! Limiting void fraction given max value
                    q_beta%vf(1)%sf(j, k, l) = max(q_beta%vf(1)%sf(j, k, l), 1._wp - lag_params%valmaxvoid)
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        call nvtxEndRange

    end subroutine s_smear_voidfraction

    subroutine s_calculate_scattered_pressure(q_prim_vf)

        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        integer, dimension(3)                               :: cell
        real(wp)                                            :: myR0, myR, myV, myPb, myShell, myRbuck, myRrupt, myDist, myA
        real(wp)                                            :: Pcell, Rcell, Pw, myRho, myPout, sumPout
        real(wp)                                            :: preterm1, term2, aux, denom, c1, c2, myInt
        integer                                             :: bub_idx, total_ids
        integer                                             :: i, k

        if (any(lag_params%interaction_model == (/1, 3/))) then  ! Kazuki's model (DV version)
            $:GPU_PARALLEL_LOOP(private='[k, cell]')
            do k = 1, nBubs
                ! Current bubble state
                myR0 = bub_R0(k)
                myR = intfc_rad(k, 2)
                myV = intfc_vel(k, 2)
                myPb = gas_p(k, 2)
                myShell = mrmtnt_shell(k, 2)
                myRbuck = mrmtnt_Rbuck(k)
                myRrupt = mrmtnt_Rrupt(k)
                if (myR > myRrupt) myShell = 0._wp

                ! ! Calculate velocity potentials (valid for one bubble per cell) call s_get_pinf(k, q_prim_vf, 2, Pcell, cell,
                ! preterm1, term2, Rcell)

                ! ! Obtain liquid density and computing speed of sound from myPinf myRho = 0._wp $:GPU_LOOP(parallelism='[seq]') do
                ! i = 1, num_fluids myRho = myRho + q_prim_vf(i)%sf(cell(1), cell(2), cell(3)) end do

                ! aux = Rcell**3._wp - myR**3._wp c2 = (3._wp/2._wp)*(myR**3._wp)*(1._wp - myR/Rcell)/aux c1 =
                ! (3._wp/2._wp)*(myR*(Rcell**2._wp - myR**2._wp))/aux

                ! Pw = f_cpbw_KM(myR0, myR, myV, myPb, myShell, myRbuck) Pw = Pw/myRho - 0.5_wp*myV**2._wp bub_dphidt(k) =
                ! (Pcell/myRho - Pw) - c2*myV**2._wp ! Accounting for the potential induced by the bubble averaged over the control
                ! volume ! Note that this is based on the incompressible flow assumption near the bubble. bub_dphidt(k) =
                ! bub_dphidt(k)/(1._wp - c1)

                ! ! Scattered pressure myPout = myRho*(c1*bub_dphidt(k) - c2*myV**2._wp)

                ! !Update emitted Pout bub_interact(k) = myPout !print*, 'myPout matching:', myPout

                ! Calculate velocity potentials (valid for one bubble per cell)
                call s_get_pinf(k, q_prim_vf, 2, Pcell, cell, preterm1, term2, Rcell)

                ! Obtain liquid density and computing speed of sound from myPinf
                myRho = 0._wp
                $:GPU_LOOP(parallelism='[seq]')
                do i = 1, num_fluids
                    myRho = myRho + q_prim_vf(i)%sf(cell(1), cell(2), cell(3))
                end do

                aux = Rcell**3._wp - myR**3._wp
                c2 = (3._wp/2._wp)*(myR**3._wp)*(1._wp - myR/Rcell)/aux
                c1 = 3._wp/2._wp*(myR*(Rcell**2._wp - myR**2._wp))/aux

                Pw = f_cpbw_KM(myR0, myR, myV, myPb, myShell, myRbuck)
                Pw = Pw + 0.5_wp*myV**2._wp
                bub_dphidt(k) = (Pcell - Pw) + c2*myV**2._wp
                ! Accounting for the potential induced by the bubble averaged over the control volume Note that this is based on the
                ! incompressible flow assumption near the bubble.
                bub_dphidt(k) = bub_dphidt(k)/(1._wp - c1)

                ! Scattered pressure
                myPout = c1*bub_dphidt(k) + c2*myV**2._wp

                ! Update emitted Pout
                bub_interact(k) = myPout
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        if (any(lag_params%interaction_model == (/2, 3/))) then  ! Aditya's model, Pout is going to be I term from eqn 3.19
            $:GPU_PARALLEL_LOOP(private='[k, cell]')
            do k = 1, nBubs
                ! Number of the bubbles in the smearing volume (Self-inclusive)
                total_ids = int(bub_int_ids(k, 1))
                sumPout = 0._wp

                if (total_ids + 1 >= 2) then
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 2, total_ids + 1
                        bub_idx = bub_int_ids(k, i)
                        ! Current interacting bubble state
                        myR = intfc_rad(bub_idx, 2)
                        myV = intfc_vel(bub_idx, 2)
                        myA = intfc_ac(bub_idx, 2)
                        myDist = (mtn_posPrev(bub_idx, 1, 2) - mtn_posPrev(k, 1, 2))**2._wp + (mtn_posPrev(bub_idx, 2, &
                                  & 2) - mtn_posPrev(k, 2, 2))**2._wp + (mtn_posPrev(bub_idx, 3, 2) - mtn_posPrev(k, 3, 2))**2._wp
                        myDist = sqrt(myDist)

                        ! if (bub_idx /= k .and. .not. f_approx_equal(myDist, 0._wp)) then  ! non-inclusive for Aditya's model
                        myInt = (2._wp*myR*myV**2._wp + myA*myR**2._wp)/myDist
                        if (myInt /= myInt) then
                            print *, myR, myV, myA, myDist, 'Bub', k, 'with bub', bub_idx
                        else
                            sumPout = sumPout - myInt
                        end if

                        ! if (k==50) print*, 'bub-50:', k, bub_int_ids(k, 1), bub_idx, sumPout, myR, myV, myA, myDist if (k==50)
                        ! print*, 'bub-50: bub_int_ids', bub_int_ids(k, i)
                    end do
                end if

                ! I term: sum over bubbles
                bub_interact(k) = sumPout
                ! print*, 'sum Pout:', bub_interact(k)
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        ! call s_mpi_barrier()

    end subroutine s_calculate_scattered_pressure

    !> Compute the bubble driving pressure p_inf
    subroutine s_get_pinf(bub_id, q_prim_vf, ptype, f_pinfl, cell, preterm1, term2, Romega)

        $:GPU_ROUTINE(function_name='s_get_pinf',parallelism='[seq]', cray_inline=True)

        integer, intent(in)                                 :: bub_id, ptype
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        real(wp), intent(out)                               :: f_pinfl
        integer, dimension(3), intent(out)                  :: cell
        real(wp), intent(out), optional                     :: preterm1, term2, Romega
        real(wp), dimension(3)                              :: scoord, psi
        real(wp)                                            :: dc, vol, aux, chardist, dist_cc
        real(wp)                                            :: volgas, term1, Rbeq, denom
        real(wp)                                            :: charvol, charpres, charvol2, charpres2, charbeta
        integer, dimension(3)                               :: cellaux
        integer                                             :: i, j, k
        integer                                             :: smearGrid, smearGridz
        logical                                             :: celloutside, condition

        scoord = mtn_s(bub_id,1:3,2)
        f_pinfl = 0._wp

        !> Find current bubble cell
        cell(:) = int(scoord(:))
        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_dims
            if (scoord(i) < 0._wp) cell(i) = cell(i) - 1
        end do

        if ((lag_params%cluster_type == 1)) then
            !> Getting p_cell in terms of only the current cell by interpolation
            !> Getting the cell volulme as Omega
            if (p > 0) then
                vol = dx(cell(1))*dy(cell(2))*dz(cell(3))
            else
                if (cyl_coord) then
                    vol = dx(cell(1))*dy(cell(2))*y_cc(cell(2))*2._wp*pi
                else
                    vol = dx(cell(1))*dy(cell(2))*lag_params%charwidth
                end if
            end if

            !> Obtain bilinear interpolation coefficients, based on the current location of the bubble.
            psi(1) = (scoord(1) - real(cell(1)))*dx(cell(1)) + x_cb(cell(1) - 1)
            if (cell(1) == (m + buff_size)) then
                cell(1) = cell(1) - 1
                psi(1) = 1._wp
            else if (cell(1) == (-buff_size)) then
                psi(1) = 0._wp
            else
                if (psi(1) < x_cc(cell(1))) cell(1) = cell(1) - 1
                psi(1) = abs((psi(1) - x_cc(cell(1)))/(x_cc(cell(1) + 1) - x_cc(cell(1))))
            end if

            psi(2) = (scoord(2) - real(cell(2)))*dy(cell(2)) + y_cb(cell(2) - 1)
            if (cell(2) == (n + buff_size)) then
                cell(2) = cell(2) - 1
                psi(2) = 1._wp
            else if (cell(2) == (-buff_size)) then
                psi(2) = 0._wp
            else
                if (psi(2) < y_cc(cell(2))) cell(2) = cell(2) - 1
                psi(2) = abs((psi(2) - y_cc(cell(2)))/(y_cc(cell(2) + 1) - y_cc(cell(2))))
            end if

            if (p > 0) then
                psi(3) = (scoord(3) - real(cell(3)))*dz(cell(3)) + z_cb(cell(3) - 1)
                if (cell(3) == (p + buff_size)) then
                    cell(3) = cell(3) - 1
                    psi(3) = 1._wp
                else if (cell(3) == (-buff_size)) then
                    psi(3) = 0._wp
                else
                    if (psi(3) < z_cc(cell(3))) cell(3) = cell(3) - 1
                    psi(3) = abs((psi(3) - z_cc(cell(3)))/(z_cc(cell(3) + 1) - z_cc(cell(3))))
                end if
            else
                psi(3) = 0._wp
            end if

            !> Perform bilinear interpolation
            if (p == 0) then  ! 2D
                f_pinfl = q_prim_vf(eqn_idx%E)%sf(cell(1), cell(2), cell(3))*(1._wp - psi(1))*(1._wp - psi(2))
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1) + 1, cell(2), cell(3))*psi(1)*(1._wp - psi(2))
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1) + 1, cell(2) + 1, cell(3))*psi(1)*psi(2)
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1), cell(2) + 1, cell(3))*(1._wp - psi(1))*psi(2)
            else  ! 3D
                f_pinfl = q_prim_vf(eqn_idx%E)%sf(cell(1), cell(2), cell(3))*(1._wp - psi(1))*(1._wp - psi(2))*(1._wp - psi(3))
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1) + 1, cell(2), cell(3))*psi(1)*(1._wp - psi(2))*(1._wp - psi(3))
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1) + 1, cell(2) + 1, cell(3))*psi(1)*psi(2)*(1._wp - psi(3))
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1), cell(2) + 1, cell(3))*(1._wp - psi(1))*psi(2)*(1._wp - psi(3))
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1), cell(2), cell(3) + 1)*(1._wp - psi(1))*(1._wp - psi(2))*psi(3)
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1) + 1, cell(2), cell(3) + 1)*psi(1)*(1._wp - psi(2))*psi(3)
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1) + 1, cell(2) + 1, cell(3) + 1)*psi(1)*psi(2)*psi(3)
                f_pinfl = f_pinfl + q_prim_vf(eqn_idx%E)%sf(cell(1), cell(2) + 1, cell(3) + 1)*(1._wp - psi(1))*psi(2)*psi(3)
            end if

            ! R_Omega
            dc = (3._wp*vol/(4._wp*pi))**(1._wp/3._wp)
        else if (lag_params%cluster_type >= 2) then
            ! Bubble dynamic closure from Maeda and Colonius (2018)

            ! Include the cell that contains the bubble (mapCells+1+mapCells)
            smearGrid = mapCells - (-mapCells) + 1
            smearGridz = smearGrid
            if (p == 0) smearGridz = 1

            charvol = 0._wp
            charpres = 0._wp
            charvol2 = 0._wp
            charpres2 = 0._wp
            vol = 0._wp
            charbeta = 0._wp

            $:GPU_LOOP(parallelism='[seq]')
            do i = 1, smearGrid
                $:GPU_LOOP(parallelism='[seq]')
                do j = 1, smearGrid
                    $:GPU_LOOP(parallelism='[seq]')
                    do k = 1, smearGridz
                        cellaux(1) = cell(1) + i - (mapCells + 1)
                        cellaux(2) = cell(2) + j - (mapCells + 1)
                        cellaux(3) = cell(3) + k - (mapCells + 1)
                        if (p == 0) cellaux(3) = 0

                        !> check if the current cell is outside the computational domain or not (including ghost cells)
                        celloutside = .false.
                        if (num_dims == 2) then
                            if ((cellaux(1) < -buff_size) .or. (cellaux(2) < -buff_size)) then
                                celloutside = .true.
                            end if
                            if ((cellaux(2) > n + buff_size) .or. (cellaux(1) > m + buff_size)) then
                                celloutside = .true.
                            end if
                        else
                            if ((cellaux(3) < -buff_size) .or. (cellaux(1) < -buff_size) .or. (cellaux(2) < -buff_size)) then
                                celloutside = .true.
                            end if

                            if ((cellaux(3) > p + buff_size) .or. (cellaux(2) > n + buff_size) .or. (cellaux(1) > m + buff_size)) &
                                & then
                                celloutside = .true.
                            end if
                        end if

                        if (lag_params%interaction_model == 2 .and. .not. celloutside) then
                            ! Liquid pressure from the cells around a virtual sphere of radius K*chardist that surrounds the bubble
                            if (p > 0) then
                                chardist = sqrt(dx(cell(1))*dy(cell(2))*dz(cell(3)))
                                dist_cc = sqrt((x_cc(cell(1)) - x_cc(cellaux(1)))**2._wp + (y_cc(cell(2)) - y_cc(cellaux(2))) &
                                               & **2._wp + (z_cc(cell(3)) - z_cc(cellaux(3)))**2._wp)
                                ! condition = abs(dist_cc-lag_params%scaleVirtualSphere*chardist) < 1.8_wp*chardist
                                condition = (cellaux(1) == cell(1) - mapCells .or. cellaux(1) == cell(1) + mapCells &
                                             & .or. cellaux(2) == cell(2) - mapCells .or. cellaux(2) == cell(2) + mapCells &
                                             & .or. cellaux(3) == cell(3) - mapCells .or. cellaux(3) == cell(3) + mapCells)
                                if (.not. condition) celloutside = .true.
                            else
                                chardist = sqrt(dx(cell(1))*dy(cell(2)))
                                dist_cc = sqrt((x_cc(cell(1)) - x_cc(cellaux(1)))**2._wp + (y_cc(cell(2)) - y_cc(cellaux(2))) &
                                               & **2._wp)
                                ! condition = abs(dist_cc-lag_params%scaleVirtualSphere*chardist) < 1.5_wp*chardist
                                condition = (cellaux(1) == cell(1) - mapCells .or. cellaux(1) == cell(1) + mapCells &
                                             & .or. cellaux(2) == cell(2) - mapCells .or. cellaux(2) == cell(2) + mapCells)
                                if (.not. condition) celloutside = .true.
                            end if
                        end if

                        if (.not. celloutside) then
                            !> Obtaining the cell volulme
                            if (p > 0) then
                                vol = dx(cellaux(1))*dy(cellaux(2))*dz(cellaux(3))
                            else
                                if (cyl_coord) then
                                    vol = dx(cellaux(1))*dy(cellaux(2))*y_cc(cellaux(2))*2._wp*pi
                                else
                                    vol = dx(cellaux(1))*dy(cellaux(2))*lag_params%charwidth
                                end if
                            end if

                            !> Update values
                            charvol = charvol + vol
                            charbeta = charbeta + q_beta%vf(1)%sf(cellaux(1), cellaux(2), cellaux(3))
                            charpres = charpres + q_prim_vf(eqn_idx%E)%sf(cellaux(1), cellaux(2), cellaux(3))*vol
                            charvol2 = charvol2 + vol*q_beta%vf(1)%sf(cellaux(1), cellaux(2), cellaux(3))
                            charpres2 = charpres2 + q_prim_vf(eqn_idx%E)%sf(cellaux(1), cellaux(2), &
                                                              & cellaux(3))*vol*q_beta%vf(1)%sf(cellaux(1), cellaux(2), cellaux(3))
                        end if
                    end do
                end do
            end do

            f_pinfl = charpres2/charvol2
            if (lag_params%interaction_model == 2) f_pinfl = charpres/charvol
            vol = charvol
            dc = (3._wp*abs(vol)/(4._wp*pi))**(1._wp/3._wp)
        end if

        ! Control volume radius
        Romega = dc

    end subroutine s_get_pinf

    !> The purpose of this procedure is to calculate and store the time-averaged heat sources from the lagrange bubbles. The heat
    !! sources model the viscous and thermal damping of the bubbles valid with the hifu solver.
    subroutine s_compute_bubble_heat_sources_HIFU(hdid)

        real(wp), intent(in)     :: hdid
        real(wp)                 :: fpb_h, fmass_n_h, fmass_v_h, fR_h, fV_h, fbeta_t_h, fshell_h, frho
        real(wp)                 :: conc_v_h, R_m_h, gamma_m_h, T_bar_h, grad_T_h, heatflux_h, fR0_h
        integer                  :: k, i
        integer                  :: abortFlag, abortFlag_max
        real(wp), dimension(1:4) :: mom_vol, mom_qvis, mom_qth_p, mom_qth_n, mom_ke
        real(wp)                 :: fxb_Rc, fqvis, fqth, fVol, fke
        logical                  :: flg_bub_in_cv
        real(wp)                 :: acPw_qvis, acPw_qth, acPW_nbubs, acPw_ke
        real(wp)                 :: sum_qvis, sum_qth

        sum_qvis = 0._wp; sum_qth = 0._wp

        if (hifu_params%moments) then
            mom_vol(1:4) = 0._wp; mom_qvis(1:4) = 0._wp; mom_ke(1:4) = 0._wp
            mom_qth_p(1:4) = 0._wp; mom_qth_n(1:4) = 0._wp
        end if

        if (hifu_params%power_balance) then
            acPw_qvis = 0._wp; acPw_qth = 0._wp
            acPW_nbubs = 0._wp; acPw_ke = 0._wp
        end if

#ifdef MFC_DEBUG
        if (proc_rank == 0) print *, 'Computing bubble heat sources', mytime, hdid
#endif
        abortFlag_max = 0
        $:GPU_PARALLEL_LOOP(private='[k]',reduction='[[abortFlag_max], [acPw_qvis, acPw_qth, acPW_nbubs, sum_qvis, &
                            & sum_qth], [mom_vol(1:4), mom_ke(1:4), mom_qvis(1:4), mom_qth_p(1:4), mom_qth_n(1:4)]]', &
                            & reductionOp='[MAX, +, +]',copy='[abortFlag_max, mom_vol(1:4), mom_ke(1:4), mom_qvis(1:4), &
                            & mom_qth_p(1:4), mom_qth_n(1:4), acPw_qvis, acPw_qth, acPW_nbubs, sum_qvis, sum_qth]')
        do k = 1, nBubs
            abortFlag = 0
            !> Current bubble state (no temporal values)
            fpb_h = gas_p(k, 1)
            fmass_n_h = gas_mg(k)
            fmass_v_h = gas_mv(k, 1)
            fR0_h = bub_R0(k)
            fR_h = intfc_rad(k, 1)
            fV_h = intfc_vel(k, 1)
            fbeta_t_h = gas_betaT(k)
            fshell_h = mrmtnt_shell(k, 1)
            frho = bub_rho(k)
            if (hifu_params%moments) fxb_Rc = (mtn_pos(k, 1, 1) - hifu_params%cloud_center(1))/hifu_params%R_cloud

            ! Mixture properties in the bubble
            conc_v_h = 0._wp
            if (lag_params%massTransfer_model .and. (fshell_h == 0._wp)) then
                conc_v_h = 1._wp/(1._wp + (R_v/R_g)*(fpb_h/pv - 1._wp))
            end if
            R_m_h = fmass_n_h*R_g + fmass_v_h*R_v
            gamma_m_h = conc_v_h*gam_v + (1._wp - conc_v_h)*gam_g

            !> Viscous damping of the bubble (Watts)
            fqvis = (4._wp*pi*fR_h**2._wp)*(4._wp*mu_l*(fV_h**2._wp)/(fR_h))
            bub_qvis(k) = bub_qvis(k) + hdid*fqvis

            !> Thermal damping of the bubble (Watts)
            if (.not. polytropic) then
                T_bar_h = fpb_h*(4._wp/3._wp*pi*fR_h**3._wp)/R_m_h
                grad_T_h = -fbeta_t_h*(T_bar_h - Tw)
                if (lag_params%heatTransfer_model .and. (fshell_h == 0._wp)) then
                    heatflux_h = (gamma_m_h - 1._wp)/gamma_m_h*grad_T_h/fR_h
                end if
            else
                T_bar_h = Tw*(fR0_h/fR_h)**(3._wp*(gam_m - 1._wp))  ! Polytropic temp
                heatflux_h = conc_v_h*k_vl + (1._wp - conc_v_h)*k_gl
                heatflux_h = 3._wp*(1._wp - gam_m)*T_bar_h/fR_h
            end if
            fqth = heatflux_h*4._wp*pi*fR_h**2._wp
            bub_qth(k) = bub_qth(k) + hdid*fqth

            sum_qvis = sum_qvis + bub_qvis(k)
            sum_qth = sum_qth + bub_qth(k)

            ! Mean radius
            bub_hifu_rad(k) = bub_hifu_rad(k) + hdid*fR_h

            fVol = (4._wp/3._wp)*pi*fR_h**3._wp
            fke = 2._wp*pi*frho*fR_h**3._wp*fV_h**2._wp

            ! Checking for NaNs and negative qvis
            if (bub_qvis(k) /= bub_qvis(k) .or. bub_qth(k) /= bub_qth(k) .or. bub_qvis(k) < 0._wp) then
                print *, 'Bubble intensity is NaN', k, bub_qvis(k), bub_qth(k), hdid
                print *, 'Viscous damping', fR_h, mu_l, fV_h
                print *, 'Thermal damping', heatflux_h, fR_h
                abortFlag = 1
            end if

            abortFlag_max = max(abortFlag_max, abortFlag)

            if (hifu_params%moments) then
                fxb_Rc = (mtn_pos(k, 1, 1) - hifu_params%cloud_center(1))/hifu_params%R_cloud

                $:GPU_LOOP(parallelism='[seq]')
                do i = 1, 4
                    mom_vol(i) = mom_vol(i) + fVol*(fxb_Rc)**(i - 1)
                    mom_qvis(i) = mom_qvis(i) + fqvis*(fxb_Rc)**(i - 1)
                    mom_ke(i) = mom_ke(i) + fke*(fxb_Rc)**(i - 1)
                    if (fqth < 0._wp) then
                        mom_qth_p(i) = mom_qth_p(i) + fqth*(fxb_Rc)**(i - 1)
                    else
                        mom_qth_n(i) = mom_qth_n(i) + fqth*(fxb_Rc)**(i - 1)
                    end if
                end do
            end if
            if (hifu_params%power_balance) then
                flg_bub_in_cv = f_bub_in_cv(mtn_pos(k,1:3,1))
                if (flg_bub_in_cv) then
                    acPw_qvis = acPw_qvis + fqvis  ! (Watts)
                    acPw_qth = acPw_qth + fqth  ! (Watts)
                    acPW_nbubs = acPW_nbubs + 1._wp
                    acPw_ke = 0._wp  ! Kinetic energy of the bubble, can be added if needed
                end if
            end if
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (abortFlag_max > 0) stop "NaNs in viscous (or thermal) damping of the bubbles"

        call s_sum_qbub(sum_qvis, sum_qth)

        if (hifu_params%moments) then
            call s_write_moments(mom_qvis, idx=1)
            call s_write_moments(mom_qth_p, idx=2)
            call s_write_moments(mom_qth_n, idx=3)
            call s_write_moments(mom_vol, idx=4)
            call s_write_moments(mom_ke, idx=5)
        end if

        if (hifu_params%power_balance) call s_write_power_balance_bubs(acPw_qvis, acPw_qth, acPw_ke, acPW_nbubs, dt)

    end subroutine s_compute_bubble_heat_sources_HIFU

    subroutine s_write_moments(mom_all, idx)

        integer, intent(in)                :: idx
        real(wp), dimension(4), intent(in) :: mom_all
        real(wp)                           :: total, moment1, moment2, moment3
        real(wp)                           :: val_tmp
        character(len=512)                 :: line

        total = mom_all(1); moment1 = mom_all(2)
        moment2 = mom_all(3); moment3 = mom_all(4)

        if (num_procs > 1) then
            val_tmp = total
            call s_mpi_allreduce_sum(val_tmp, total)
            val_tmp = moment1
            call s_mpi_allreduce_sum(val_tmp, moment1)
            val_tmp = moment2
            call s_mpi_allreduce_sum(val_tmp, moment2)
            val_tmp = moment3
            call s_mpi_allreduce_sum(val_tmp, moment3)
        end if

        ! Write the heat statistics to file
        if (proc_rank == 0) then
            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + dt, moment1/total, moment2/total, &
                   & moment3/total, total
            write (97 - idx, '(A)') trim(line)
        end if

    end subroutine s_write_moments

    subroutine s_mean_radius_hifu(t_sampled)

        real(wp), intent(in) :: t_sampled
        integer              :: k

        $:GPU_PARALLEL_LOOP(private='[k]',copyin='[t_sampled]')
        do k = 1, nBubs
            bub_hifu_rad(k) = bub_hifu_rad(k)/t_sampled  ! meters
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_mean_radius_hifu

    ! Compute the first, second, and third moments of the heat source from the bubbles' damping.
    subroutine s_write_heat_stats_bubbles(sampledTime)

        real(wp), intent(in)                 :: sampledTime
        real(wp)                             :: total_heat_vis, heat_moment1_vis, heat_moment2_vis, heat_moment3_vis
        real(wp)                             :: total_heat_th, heat_moment1_th, heat_moment2_th, heat_moment3_th
        real(wp)                             :: total_vol, moment1_vol, moment2_vol, moment3_vol
        real(wp)                             :: val_tmp, fR_h, fqvis, fqth, fxb_Rc, fVol
        integer                              :: i, j, k, l
        logical                              :: file_exist
        character(LEN=path_len + 2*name_len) :: file_loc

        total_heat_vis = 0._wp; heat_moment1_vis = 0._wp
        heat_moment2_vis = 0._wp; heat_moment3_vis = 0._wp

        total_heat_th = 0._wp; heat_moment1_th = 0._wp
        heat_moment2_th = 0._wp; heat_moment3_th = 0._wp

        total_vol = 0._wp; moment1_vol = 0._wp
        moment2_vol = 0._wp; moment3_vol = 0._wp

        $:GPU_PARALLEL_LOOP(private='[k]', reduction='[[total_heat_vis, heat_moment1_vis, heat_moment2_vis, heat_moment3_vis, &
                            & total_heat_th, heat_moment1_th, heat_moment2_th, heat_moment3_th, total_vol, moment1_vol, &
                            & moment2_vol, moment3_vol]]', reductionOp='[MAX]', copy='[total_heat_vis, heat_moment1_vis, &
                            & heat_moment2_vis, heat_moment3_vis, total_heat_th, heat_moment1_th, heat_moment2_th, &
                            & heat_moment3_th, total_vol, moment1_vol, moment2_vol, moment3_vol]')
        do k = 1, nBubs
            fR_h = intfc_rad(k, 1)
            fqvis = bub_qvis(k)
            fqth = bub_qth(k)
            fxb_Rc = (mtn_pos(k, 1, 1) - hifu_params%cloud_center(1))/hifu_params%R_cloud
            fVol = (4._wp/3._wp)*pi*fR_h**3._wp

            total_heat_vis = total_heat_vis + fqvis
            heat_moment1_vis = heat_moment1_vis + fqvis*(fxb_Rc)
            heat_moment2_vis = heat_moment2_vis + fqvis*(fxb_Rc)**2._wp
            heat_moment3_vis = heat_moment3_vis + fqvis*(fxb_Rc)**3._wp

            total_heat_th = total_heat_th + fqth
            heat_moment1_th = heat_moment1_th + fqth*(fxb_Rc)
            heat_moment2_th = heat_moment2_th + fqth*(fxb_Rc)**2._wp
            heat_moment3_th = heat_moment3_th + fqth*(fxb_Rc)**3._wp

            total_vol = total_vol + fVol
            moment1_vol = moment1_vol + fVol*(fxb_Rc)
            moment2_vol = moment2_vol + fVol*(fxb_Rc)**2._wp
            moment3_vol = moment3_vol + fVol*(fxb_Rc)**3._wp
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (num_procs > 1) then
            val_tmp = total_heat_vis
            call s_mpi_allreduce_sum(val_tmp, total_heat_vis)
            val_tmp = heat_moment1_vis
            call s_mpi_allreduce_sum(val_tmp, heat_moment1_vis)
            val_tmp = heat_moment2_vis
            call s_mpi_allreduce_sum(val_tmp, heat_moment2_vis)
            val_tmp = heat_moment3_vis
            call s_mpi_allreduce_sum(val_tmp, heat_moment3_vis)

            val_tmp = total_heat_th
            call s_mpi_allreduce_sum(val_tmp, total_heat_th)
            val_tmp = heat_moment1_th
            call s_mpi_allreduce_sum(val_tmp, heat_moment1_th)
            val_tmp = heat_moment2_th
            call s_mpi_allreduce_sum(val_tmp, heat_moment2_th)
            val_tmp = heat_moment3_th
            call s_mpi_allreduce_sum(val_tmp, heat_moment3_th)

            val_tmp = total_vol
            call s_mpi_allreduce_sum(val_tmp, total_vol)
            val_tmp = moment1_vol
            call s_mpi_allreduce_sum(val_tmp, moment1_vol)
            val_tmp = moment2_vol
            call s_mpi_allreduce_sum(val_tmp, moment2_vol)
            val_tmp = moment3_vol
            call s_mpi_allreduce_sum(val_tmp, moment3_vol)
        end if

        ! Write the heat statistics to file
        if (proc_rank == 0) then
            write (file_loc, '(A,I0,A)') 'moments_qvis.dat'
            file_loc = trim(case_dir) // '/D/' // trim(file_loc)
            inquire (FILE=trim(file_loc), EXIST=file_exist)

            open (11, FILE=trim(file_loc), form='formatted', position='append')

            write (11, '(4X,I24.8,4e24.8)') sampledTime, heat_moment1_vis/total_heat_vis, heat_moment2_vis/total_heat_vis, &
                   & heat_moment3_vis/total_heat_vis, total_heat_vis

            close (11)

            write (file_loc, '(A,I0,A)') 'moments_qth.dat'
            file_loc = trim(case_dir) // '/D/' // trim(file_loc)
            inquire (FILE=trim(file_loc), EXIST=file_exist)

            open (11, FILE=trim(file_loc), form='formatted', position='append')

            write (11, '(4X,I24.8,4e24.8)') sampledTime, heat_moment1_th/total_heat_th, heat_moment2_th/total_heat_th, &
                   & heat_moment3_th/total_heat_th, total_heat_th

            close (11)

            write (file_loc, '(A,I0,A)') 'moments_vol.dat'
            file_loc = trim(case_dir) // '/D/' // trim(file_loc)
            inquire (FILE=trim(file_loc), EXIST=file_exist)

            open (11, FILE=trim(file_loc), form='formatted', position='append')

            write (11, '(4X,I24.8,4e24.8)') sampledTime, moment1_vol/total_vol, moment2_vol/total_vol, moment3_vol/total_vol, &
                   & total_vol

            close (11)
        end if

    end subroutine s_write_heat_stats_bubbles

    !> Update Lagrangian bubble variables using TVD Runge-Kutta time stepping
    impure subroutine s_update_lagrange_tdv_rk(stage)

        integer, intent(in) :: stage
        integer             :: k

        if (time_stepper == time_stepper_rk1) then  ! 1st order TVD RK
            $:GPU_PARALLEL_LOOP(private='[k]')
            do k = 1, nBubs
                ! u{1} = u{n} +  dt * RHS{n}
                intfc_rad(k, 1) = intfc_rad(k, 1) + dt*intfc_draddt(k, 1)
                intfc_vel(k, 1) = intfc_vel(k, 1) + dt*intfc_dveldt(k, 1)
                mtn_pos(k,1:3,1) = mtn_pos(k,1:3,1) + dt*mtn_dposdt(k,1:3,1)
                mtn_vel(k,1:3,1) = mtn_vel(k,1:3,1) + dt*mtn_dveldt(k,1:3,1)
                gas_p(k, 1) = gas_p(k, 1) + dt*gas_dpdt(k, 1)
                gas_mv(k, 1) = gas_mv(k, 1) + dt*gas_dmvdt(k, 1)
                mrmtnt_shell(k, 1) = mrmtnt_shell(k, 2)
                intfc_ac(k, 1) = intfc_dveldt(k, 1)
                if (polytropic) gas_p(k, 1) = pv + (gas_p(k, 2) - pv)*(bub_R0(k)/intfc_rad(k, 1))**(3._wp*gam_m)
            end do
            $:END_GPU_PARALLEL_LOOP()

            call s_transfer_data_to_tmp
            call s_calculate_lag_bubble_stats()
            if (lag_params%write_bubbles) then
                call s_write_lag_particles(mytime, replace=.false.)
            end if
            call s_write_void_evol(mytime, replace=.false.)
        else if (time_stepper == time_stepper_rk2) then  ! 2nd order TVD RK
            if (stage == 1) then
                $:GPU_PARALLEL_LOOP(private='[k]')
                do k = 1, nBubs
                    ! u{1} = u{n} +  dt * RHS{n}
                    intfc_rad(k, 2) = intfc_rad(k, 1) + dt*intfc_draddt(k, 1)
                    intfc_vel(k, 2) = intfc_vel(k, 1) + dt*intfc_dveldt(k, 1)
                    mtn_pos(k,1:3,2) = mtn_pos(k,1:3,1) + dt*mtn_dposdt(k,1:3,1)
                    mtn_vel(k,1:3,2) = mtn_vel(k,1:3,1) + dt*mtn_dveldt(k,1:3,1)
                    gas_p(k, 2) = gas_p(k, 1) + dt*gas_dpdt(k, 1)
                    gas_mv(k, 2) = gas_mv(k, 1) + dt*gas_dmvdt(k, 1)
                    if (.not. polytropic) gas_p(k, 2) = gas_p(k, 1) + dt*gas_dpdt(k, 1)
                end do
                $:END_GPU_PARALLEL_LOOP()
            else if (stage == 2) then
                $:GPU_PARALLEL_LOOP(private='[k]')
                do k = 1, nBubs
                    ! u{1} = u{n} + (1/2) * dt * (RHS{n} + RHS{1})
                    intfc_rad(k, 1) = intfc_rad(k, 1) + dt*(intfc_draddt(k, 1) + intfc_draddt(k, 2))/2._wp
                    intfc_vel(k, 1) = intfc_vel(k, 1) + dt*(intfc_dveldt(k, 1) + intfc_dveldt(k, 2))/2._wp
                    mtn_pos(k,1:3,1) = mtn_pos(k,1:3,1) + dt*(mtn_dposdt(k,1:3,1) + mtn_dposdt(k,1:3,2))/2._wp
                    mtn_vel(k,1:3,1) = mtn_vel(k,1:3,1) + dt*(mtn_dveldt(k,1:3,1) + mtn_dveldt(k,1:3,2))/2._wp
                    gas_p(k, 1) = gas_p(k, 1) + dt*(gas_dpdt(k, 1) + gas_dpdt(k, 2))/2._wp
                    gas_mv(k, 1) = gas_mv(k, 1) + dt*(gas_dmvdt(k, 1) + gas_dmvdt(k, 2))/2._wp
                    if (lag_params%coatedBub_model .and. (mrmtnt_shell(k, 2) == 0._wp)) then
                        if (intfc_rad(k, 1) < mrmtnt_Rrupt(k)) mrmtnt_shell(k, 2) = 1._wp  ! No actual rupture happened during dt
                    end if
                    mrmtnt_shell(k, 1) = mrmtnt_shell(k, 2)
                    intfc_ac(k, 1) = (intfc_dveldt(k, 1) + intfc_dveldt(k, 2))/2._wp
                    if (polytropic) gas_p(k, 1) = pv + (gas_p(k, 2) - pv)*(bub_R0(k)/intfc_rad(k, 1))**(3._wp*gam_m)
                end do
                $:END_GPU_PARALLEL_LOOP()

                call s_transfer_data_to_tmp
                call s_calculate_lag_bubble_stats()
                if (lag_params%write_bubbles) then
                    call s_write_lag_particles(mytime, replace=.false.)
                end if
                call s_write_void_evol(mytime, replace=.false.)
            end if
        else if (time_stepper == time_stepper_rk3) then  ! 3rd order TVD RK
            if (stage == 1) then
                $:GPU_PARALLEL_LOOP(private='[k]')
                do k = 1, nBubs
                    ! u{1} = u{n} +  dt * RHS{n}
                    intfc_rad(k, 2) = intfc_rad(k, 1) + dt*intfc_draddt(k, 1)
                    intfc_vel(k, 2) = intfc_vel(k, 1) + dt*intfc_dveldt(k, 1)
                    mtn_pos(k,1:3,2) = mtn_pos(k,1:3,1) + dt*mtn_dposdt(k,1:3,1)
                    mtn_vel(k,1:3,2) = mtn_vel(k,1:3,1) + dt*mtn_dveldt(k,1:3,1)
                    gas_p(k, 2) = gas_p(k, 1) + dt*gas_dpdt(k, 1)
                    gas_mv(k, 2) = gas_mv(k, 1) + dt*gas_dmvdt(k, 1)
                    if (.not. polytropic) gas_p(k, 2) = gas_p(k, 1) + dt*gas_dpdt(k, 1)
                end do
                $:END_GPU_PARALLEL_LOOP()
            else if (stage == 2) then
                $:GPU_PARALLEL_LOOP(private='[k]')
                do k = 1, nBubs
                    ! u{2} = u{n} + (1/4) * dt * [RHS{n} + RHS{1}]
                    intfc_rad(k, 2) = intfc_rad(k, 1) + dt*(intfc_draddt(k, 1) + intfc_draddt(k, 2))/4._wp
                    intfc_vel(k, 2) = intfc_vel(k, 1) + dt*(intfc_dveldt(k, 1) + intfc_dveldt(k, 2))/4._wp
                    mtn_pos(k,1:3,2) = mtn_pos(k,1:3,1) + dt*(mtn_dposdt(k,1:3,1) + mtn_dposdt(k,1:3,2))/4._wp
                    mtn_vel(k,1:3,2) = mtn_vel(k,1:3,1) + dt*(mtn_dveldt(k,1:3,1) + mtn_dveldt(k,1:3,2))/4._wp
                    gas_p(k, 2) = gas_p(k, 1) + dt*(gas_dpdt(k, 1) + gas_dpdt(k, 2))/4._wp
                    gas_mv(k, 2) = gas_mv(k, 1) + dt*(gas_dmvdt(k, 1) + gas_dmvdt(k, 2))/4._wp
                    if (.not. polytropic) gas_p(k, 2) = gas_p(k, 1) + dt*(gas_dpdt(k, 1) + gas_dpdt(k, 2))/4._wp
                end do
                $:END_GPU_PARALLEL_LOOP()
            else if (stage == 3) then
                $:GPU_PARALLEL_LOOP(private='[k]')
                do k = 1, nBubs
                    ! u{n+1} = u{n} + (2/3) * dt * [(1/4)* RHS{n} + (1/4)* RHS{1} + RHS{2}]
                    intfc_rad(k, 1) = intfc_rad(k, 1) + (2._wp/3._wp)*dt*(intfc_draddt(k, 1)/4._wp + intfc_draddt(k, &
                              & 2)/4._wp + intfc_draddt(k, 3))
                    intfc_vel(k, 1) = intfc_vel(k, 1) + (2._wp/3._wp)*dt*(intfc_dveldt(k, 1)/4._wp + intfc_dveldt(k, &
                              & 2)/4._wp + intfc_dveldt(k, 3))
                    mtn_pos(k,1:3,1) = mtn_pos(k,1:3,1) + (2._wp/3._wp)*dt*(mtn_dposdt(k,1:3,1)/4._wp + mtn_dposdt(k,1:3, &
                            & 2)/4._wp + mtn_dposdt(k,1:3,3))
                    mtn_vel(k,1:3,1) = mtn_vel(k,1:3,1) + (2._wp/3._wp)*dt*(mtn_dveldt(k,1:3,1)/4._wp + mtn_dveldt(k,1:3, &
                            & 2)/4._wp + mtn_dveldt(k,1:3,3))
                    gas_p(k, 1) = gas_p(k, 1) + (2._wp/3._wp)*dt*(gas_dpdt(k, 1)/4._wp + gas_dpdt(k, 2)/4._wp + gas_dpdt(k, 3))
                    gas_mv(k, 1) = gas_mv(k, 1) + (2._wp/3._wp)*dt*(gas_dmvdt(k, 1)/4._wp + gas_dmvdt(k, 2)/4._wp + gas_dmvdt(k, 3))
                    if (lag_params%coatedBub_model .and. (mrmtnt_shell(k, 2) == 0._wp)) then
                        if (intfc_rad(k, 1) < mrmtnt_Rrupt(k)) mrmtnt_shell(k, 2) = 1._wp  ! No actual rupture happened during dt
                    end if
                    mrmtnt_shell(k, 1) = mrmtnt_shell(k, 2)
                    intfc_ac(k, 1) = (2._wp/3._wp)*(intfc_dveldt(k, 1)/4._wp + intfc_dveldt(k, 2)/4._wp + intfc_dveldt(k, 3))
                    if (polytropic) gas_p(k, 1) = pv + (gas_p(k, 2) - pv)*(bub_R0(k)/intfc_rad(k, 1))**(3._wp*gam_m)
                end do
                $:END_GPU_PARALLEL_LOOP()

                call s_transfer_data_to_tmp
                call s_calculate_lag_bubble_stats()
                if (lag_params%write_bubbles) then
                    call s_write_lag_particles(mytime, replace=.false.)
                end if
                call s_write_void_evol(mytime, replace=.false.)
            end if
        end if

    end subroutine s_update_lagrange_tdv_rk

    !> Locate the cell index for a given physical position
    subroutine s_locate_cell(pos, cell, scoord)

        real(wp), dimension(3), intent(in)   :: pos
        real(wp), dimension(3), intent(out)  :: scoord
        integer, dimension(3), intent(inout) :: cell
        integer                              :: i

        do while (pos(1) < x_cb(cell(1) - 1))
            cell(1) = cell(1) - 1
        end do

        do while (pos(1) > x_cb(cell(1)))
            cell(1) = cell(1) + 1
        end do

        do while (pos(2) < y_cb(cell(2) - 1))
            cell(2) = cell(2) - 1
        end do

        do while (pos(2) > y_cb(cell(2)))
            cell(2) = cell(2) + 1
        end do

        if (p > 0) then
            do while (pos(3) < z_cb(cell(3) - 1))
                cell(3) = cell(3) - 1
            end do
            do while (pos(3) > z_cb(cell(3)))
                cell(3) = cell(3) + 1
            end do
        end if

        ! The numbering of the cell of which left boundary is the domain boundary is 0. if comp.coord of the pos is s, the real
        ! coordinate of s is (the coordinate of the left boundary of the Floor(s)-th cell) + (s-(int(s))*(cell-width). In other
        ! words, the coordinate of the center of the cell is x_cc(cell).

        ! coordinates in computational space
        scoord(1) = cell(1) + (pos(1) - x_cb(cell(1) - 1))/dx(cell(1))
        scoord(2) = cell(2) + (pos(2) - y_cb(cell(2) - 1))/dy(cell(2))
        scoord(3) = 0._wp
        if (p > 0) scoord(3) = cell(3) + (pos(3) - z_cb(cell(3) - 1))/dz(cell(3))
        cell(:) = int(scoord(:))
        do i = 1, num_dims
            if (scoord(i) < 0._wp) cell(i) = cell(i) - 1
        end do

    end subroutine s_locate_cell

    !> Transfer data into the temporal variables
    impure subroutine s_transfer_data_to_tmp()

        integer :: k

        $:GPU_PARALLEL_LOOP(private='[k]')
        do k = 1, nBubs
            if (.not. polytropic) gas_p(k, 2) = gas_p(k, 1)
            gas_mv(k, 2) = gas_mv(k, 1)
            intfc_rad(k, 2) = intfc_rad(k, 1)
            intfc_vel(k, 2) = intfc_vel(k, 1)
            intfc_ac(k, 2) = intfc_ac(k, 1)
            mtn_pos(k,1:3,2) = mtn_pos(k,1:3,1)
            mtn_posPrev(k,1:3,2) = mtn_posPrev(k,1:3,1)
            mtn_vel(k,1:3,2) = mtn_vel(k,1:3,1)
            mtn_s(k,1:3,2) = mtn_s(k,1:3,1)
            mrmtnt_shell(k, 2) = mrmtnt_shell(k, 1)
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_transfer_data_to_tmp

    !> The purpose of this procedure is to determine if the global coordinates of the bubbles are present in the current MPI
    !! processor (including ghost cells).
    !! @param pos_part Spatial coordinates of the bubble
    ! pure function particle_in_domain(pos_part, restartFlag)

    ! logical :: particle_in_domain real(wp), dimension(3), intent(in) :: pos_part logical, intent(in) :: restartFlag

    !     real(wp) :: pos_part_radial

    ! ! 2D if (p == 0 .and. cyl_coord .neqv. .true.) then ! Defining a virtual z-axis that has the same dimensions as y-axis !
    ! defined in the input file particle_in_domain = ((pos_part(1) < x_cb(m + buff_size)) .and. (pos_part(1) >= x_cb(-buff_size -
    ! 1)) .and. & (pos_part(2) < y_cb(n + buff_size)) .and. (pos_part(2) >= y_cb(-buff_size - 1)) .and. & (pos_part(3) <
    ! lag_params%charwidth/2._wp) .and. (pos_part(3) >= -lag_params%charwidth/2._wp)) else ! cyl_coord if (restartFlag) then
    ! pos_part_radial = pos_part(2) else pos_part_radial = sqrt(pos_part(2)**2._wp + pos_part(3)**2._wp) end if

    ! particle_in_domain = ((pos_part(1) < x_cb(m + buff_size)) .and. (pos_part(1) >= x_cb(-buff_size - 1)) .and. & (pos_part_radial
    ! < y_cb(n + buff_size)) .and. (pos_part_radial >= max(y_cb(-buff_size - 1), 0._wp))) end if

    ! ! 3D if (p > 0) then particle_in_domain = ((pos_part(1) < x_cb(m + buff_size)) .and. (pos_part(1) >= x_cb(-buff_size - 1))
    ! .and. & (pos_part(2) < y_cb(n + buff_size)) .and. (pos_part(2) >= y_cb(-buff_size - 1)) .and. & (pos_part(3) < z_cb(p +
    ! buff_size)) .and. (pos_part(3) >= z_cb(-buff_size - 1))) end if

    ! ! For symmetric boundary condition if (bc_x%beg == BC_REFLECTIVE) then particle_in_domain = (particle_in_domain .and.
    ! (pos_part(1) >= x_cb(-1))) end if if (bc_x%end == BC_REFLECTIVE) then particle_in_domain = (particle_in_domain .and.
    ! (pos_part(1) < x_cb(m))) end if if (bc_y%beg == BC_REFLECTIVE .and. (.not. cyl_coord)) then particle_in_domain =
    ! (particle_in_domain .and. (pos_part(2) >= y_cb(-1))) end if if (bc_y%end == BC_REFLECTIVE .and. (.not. cyl_coord)) then
    ! particle_in_domain = (particle_in_domain .and. (pos_part(2) < y_cb(n))) end if

    ! if (p > 0) then if (bc_z%beg == BC_REFLECTIVE) then particle_in_domain = (particle_in_domain .and. (pos_part(3) >= z_cb(-1)))
    ! end if if (bc_z%end == BC_REFLECTIVE) then particle_in_domain = (particle_in_domain .and. (pos_part(3) < z_cb(p))) end if end
    ! if

    ! end function particle_in_domain

    !> Determine if a Lagrangian bubble is within the physical domain excluding ghost cells
    function particle_in_domain_physical(pos_part)

        logical                            :: particle_in_domain_physical
        real(wp), dimension(3), intent(in) :: pos_part

        particle_in_domain_physical = ((pos_part(1) < x_cb(m)) .and. (pos_part(1) >= x_cb(-1)) .and. (pos_part(2) < y_cb(n)) &
                                       & .and. (pos_part(2) >= y_cb(-1)))

        if (p > 0) then
            particle_in_domain_physical = (particle_in_domain_physical .and. (pos_part(3) < z_cb(p)) .and. (pos_part(3) &
                                           & >= z_cb(-1)))
        end if

    end function particle_in_domain_physical

    !> Compute the gradient of a scalar field using second-order central differences on a non-uniform grid
    subroutine s_gradient_dir(q, dq, dir)

        real(stp), dimension(idwbuff(1)%beg:,idwbuff(2)%beg:,idwbuff(3)%beg:), intent(inout) :: q, dq
        integer, intent(in)                                                                  :: dir
        integer                                                                              :: i, j, k

        if (dir == 1) then
            ! Gradient in x dir.
            $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
            do k = 0, p
                do j = 0, n
                    do i = 0, m
                        dq(i, j, k) = q(i, j, k)*(dx(i + 1) - dx(i - 1)) + q(i + 1, j, k)*(dx(i) + dx(i - 1)) - q(i - 1, j, &
                           & k)*(dx(i) + dx(i + 1))
                        dq(i, j, k) = dq(i, j, k)/((dx(i) + dx(i - 1))*(dx(i) + dx(i + 1)))
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        else if (dir == 2) then
            ! Gradient in y dir.
            $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
            do k = 0, p
                do j = 0, n
                    do i = 0, m
                        dq(i, j, k) = q(i, j, k)*(dy(j + 1) - dy(j - 1)) + q(i, j + 1, k)*(dy(j) + dy(j - 1)) - q(i, j - 1, &
                           & k)*(dy(j) + dy(j + 1))
                        dq(i, j, k) = dq(i, j, k)/((dy(j) + dy(j - 1))*(dy(j) + dy(j + 1)))
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        else if (dir == 3) then
            ! Gradient in z dir.
            $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
            do k = 0, p
                do j = 0, n
                    do i = 0, m
                        dq(i, j, k) = q(i, j, k)*(dz(k + 1) - dz(k - 1)) + q(i, j, k + 1)*(dz(k) + dz(k - 1)) - q(i, j, &
                           & k - 1)*(dz(k) + dz(k + 1))
                        dq(i, j, k) = dq(i, j, k)/((dz(k) + dz(k - 1))*(dz(k) + dz(k + 1)))
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()
        end if

        ! call s_mpi_barrier()

    end subroutine s_gradient_dir

    !> Subroutine that writes on each time step the changes of the lagrangian bubbles.
    !! @param qtime Current time
    impure subroutine s_write_lag_particles(qtime, replace)

        real(wp), intent(in)                 :: qtime
        logical, intent(in)                  :: replace
        integer                              :: k
        logical                              :: file_exist, write_unique_id
        character(LEN=path_len + 2*name_len) :: file_loc
        character(LEN=25)                    :: FMT
        character(len=512)                   :: line

        write_unique_id = .false.
        if (lag_params%write_only_bub_id /= dflt_int) write_unique_id = .true.

        write (file_loc, '(A,I0,A)') 'lag_bubble_evol_', proc_rank, '.dat'
        file_loc = trim(case_dir) // '/D/' // trim(file_loc)
        inquire (FILE=trim(file_loc), EXIST=file_exist)

        if (.not. file_exist .or. replace .or. write_unique_id) then
            open (11, FILE=trim(file_loc), form='formatted', position='rewind')
            if (hifu .or. lag_params%coatedBub_model) then
                write (11, &
                       & '(A)') &
                       & 'mytime,dt,id,x,y,z,radius,intfc_vel,intfc_acc,p_inf,vap_mass,vap_conc,p_bub,mrmtnt_shell,mrmtnt_Rrupt'
            else
                write (11, '(A)') 'mytime,id,x,y,z,vap_mass,vap_conc,radius,intfc_vel,p_bub'
            end if
        else
            open (11, FILE=trim(file_loc), form='formatted', position='append')
        end if

        $:GPU_UPDATE(host='[intfc_rad, intfc_vel, intfc_ac, bub_interact, gas_mv, gas_p, mrmtnt_shell]')

        ! One specified bubble id only
        if (write_unique_id) then
            k = lag_params%write_only_bub_id
            if (k == lag_id(k, 1)) then
                write (line, &
                       & '(ES24.16,",",ES24.16,",",I0,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",I0,",",ES24.16)') qtime, &
                       & dt, lag_id(k, 1), mtn_pos(k, 1, 1), mtn_pos(k, 2, 1), mtn_pos(k, 3, 1), intfc_rad(k, 1), intfc_vel(k, &
                       & 1), intfc_ac(k, 1), bub_interact(k), gas_mv(k, 1), gas_mv(k, 1)/(gas_mv(k, 1) + gas_mg(k)), gas_p(k, 1), &
                       & int(mrmtnt_shell(k, 1)), mrmtnt_Rrupt(k)
                write (11, '(A)') trim(line)
            end if

            close (11)
            return
        end if

        if (hifu .or. lag_params%coatedBub_model) then
            do k = 1, nBubs
                write (line, &
                       & '(ES24.16,",",ES24.16,",",I0,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",I0,",",ES24.16)') qtime, &
                       & dt, lag_id(k, 1), mtn_pos(k, 1, 1), mtn_pos(k, 2, 1), mtn_pos(k, 3, 1), intfc_rad(k, 1), intfc_vel(k, &
                       & 1), intfc_ac(k, 1), bub_interact(k), gas_mv(k, 1), gas_mv(k, 1)/(gas_mv(k, 1) + gas_mg(k)), gas_p(k, 1), &
                       & int(mrmtnt_shell(k, 1)), mrmtnt_Rrupt(k)
                write (11, '(A)') trim(line)
            end do

            close (11)
            return
        end if

        if (precision == 1) then
            FMT = "(F16.8,I14,8F16.8)"
        else
            FMT = "(F24.16,I14,8F24.16)"
        end if

        ! Cycle through list
        do k = 1, nBubs
            write (11, FMT) qtime, lag_id(k, 1), mtn_pos(k, 1, 1), mtn_pos(k, 2, 1), mtn_pos(k, 3, 1), gas_mv(k, 1), gas_mv(k, &
                   & 1)/(gas_mv(k, 1) + gas_mg(k)), intfc_rad(k, 1), intfc_vel(k, 1), gas_p(k, 1)
        end do

        close (11)

    end subroutine s_write_lag_particles

    !> Subroutine that writes some useful statistics related to the volume fraction of the particles (void fraction) in the
    !! computatioational domain on each time step.
    !! @param qtime Current time
    impure subroutine s_write_void_evol(qtime, replace)

        real(wp), intent(in)                 :: qtime
        logical, intent(in)                  :: replace
        real(wp)                             :: volcell, voltot
        real(wp)                             :: lag_void_max, lag_void_avg, lag_vol
        real(wp)                             :: void_max_glb, void_avg_glb, vol_glb
        real(wp)                             :: aux_glb, nBubs_all
        integer                              :: i, j, k
        character(LEN=path_len + 2*name_len) :: file_loc
        logical                              :: file_exist
        character(len=512)                   :: line

        if (proc_rank == 0) then
            write (file_loc, '(A)') 'voidfraction.dat'
            file_loc = trim(case_dir) // '/D/' // trim(file_loc)
            inquire (FILE=trim(file_loc), EXIST=file_exist)
            if (.not. file_exist .or. replace) then
                open (12, FILE=trim(file_loc), form='formatted', position='rewind')
                if (hifu) then
                    write (12, '(A)') 'mytime,dt,nbubs,mean_rad,max_rad,min_rad,sum_vol_bubs,avg_void,max_void,euler_vol'
                end if
            else
                open (12, FILE=trim(file_loc), form='formatted', position='append')
            end if
        end if

        lag_void_max = 0._wp
        lag_void_avg = 0._wp
        lag_vol = 0._wp
        $:GPU_PARALLEL_LOOP(private='[volcell]', collapse=3, reduction='[[lag_vol, lag_void_avg], [lag_void_max]]', &
                            & reductionOp='[+, MAX]', copy='[lag_vol, lag_void_avg, lag_void_max]')
        do k = 0, p
            do j = 0, n
                do i = 0, m
                    lag_void_max = max(lag_void_max, 1._wp - q_beta%vf(1)%sf(i, j, k))
                    call s_get_char_vol(i, j, k, volcell)
                    if ((1._wp - q_beta%vf(1)%sf(i, j, k)) > 5.0d-11) then
                        lag_void_avg = lag_void_avg + (1._wp - q_beta%vf(1)%sf(i, j, k))*volcell
                        lag_vol = lag_vol + volcell
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        nBubs_all = real(nBubs, wp)

        $:GPU_UPDATE(host='[Rmax_glb, Rmin_glb, Rmean_glb, lag_vol_glb]')

#ifdef MFC_MPI
        if (num_procs > 1) then
            call s_mpi_allreduce_max(lag_void_max, void_max_glb)
            lag_void_max = void_max_glb
            call s_mpi_allreduce_sum(lag_vol, vol_glb)
            lag_vol = vol_glb
            call s_mpi_allreduce_sum(lag_void_avg, void_avg_glb)
            lag_void_avg = void_avg_glb
            call s_mpi_allreduce_max(Rmax_glb, aux_glb)
            Rmax_glb = aux_glb
            call s_mpi_allreduce_min(Rmin_glb, aux_glb)
            Rmin_glb = aux_glb
            call s_mpi_allreduce_sum(Rmean_glb, aux_glb)
            Rmean_glb = aux_glb
            call s_mpi_allreduce_sum(nBubs_all, aux_glb)
            nBubs_all = aux_glb
            call s_mpi_allreduce_sum(lag_vol_glb, aux_glb)
            lag_vol_glb = aux_glb
        end if
#endif
        voltot = lag_void_avg

        ! This voidavg value does not reflect the real void fraction in the cloud since the cell which does not have bubbles are not
        ! accounted
        if (lag_vol > 0._wp) lag_void_avg = lag_void_avg/lag_vol

        if (proc_rank == 0) then
            if (hifu) then
                write (line, &
                       & '(ES24.16,",",ES24.16,",",I0,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') qtime, &
                       & dt, int(nBubs_all), Rmean_glb/nBubs_all, Rmax_glb, Rmin_glb, lag_vol_glb, lag_void_avg, lag_void_max, &
                       & voltot
                write (12, '(A)') trim(line)
            else
                write (12, '(6X,4e24.8)') qtime, lag_void_avg, lag_void_max, voltot
            end if
            close (12)
        end if

    end subroutine s_write_void_evol

    !> Write restart files for the Lagrangian bubble solver
    impure subroutine s_write_restart_lag_bubbles(t_step)

        ! Generic string used to store the address of a particular file
        integer, intent(in)                  :: t_step
        character(LEN=path_len + 2*name_len) :: file_loc
        logical                              :: file_exist
        integer                              :: bub_id, tot_part
        integer                              :: i, k

#ifdef MFC_MPI
        ! For Parallel I/O
        integer                                :: ifile, ierr
        integer, dimension(MPI_STATUS_SIZE)    :: status
        integer(KIND=MPI_OFFSET_KIND)          :: disp
        integer                                :: view
        integer, dimension(2)                  :: gsizes, lsizes, start_idx_part
        integer, allocatable                   :: proc_bubble_counts(:)
        real(wp), dimension(1:1,1:lag_io_vars) :: dummy

        dummy = 0._wp

        bub_id = 0._wp
        if (nBubs /= 0) then
            do k = 1, nBubs
                if (particle_in_domain_physical(mtn_pos(k,1:3,1))) then
                    bub_id = bub_id + 1
                end if
            end do
        end if

        if (.not. parallel_io) return

        allocate (proc_bubble_counts(num_procs))

        lsizes(1) = bub_id
        lsizes(2) = lag_io_vars

        ! Total number of particles
        call MPI_ALLREDUCE(bub_id, tot_part, 1, MPI_integer, MPI_SUM, MPI_COMM_WORLD, ierr)

        call MPI_ALLGATHER(bub_id, 1, MPI_INTEGER, proc_bubble_counts, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)

        ! Calculate starting index for this processor's particles
        call MPI_EXSCAN(lsizes(1), start_idx_part(1), 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
        if (proc_rank == 0) start_idx_part(1) = 0
        start_idx_part(2) = 0

        gsizes(1) = tot_part
        gsizes(2) = lag_io_vars

        write (file_loc, '(A,I0,A)') 'lag_bubbles_', t_step, '.dat'
        file_loc = trim(case_dir) // '/restart_data' // trim(mpiiofs) // trim(file_loc)

        ! Clean up existing file
        if (proc_rank == 0) then
            inquire (FILE=trim(file_loc), EXIST=file_exist)
            if (file_exist) then
                call MPI_FILE_DELETE(file_loc, mpi_info_int, ierr)
            end if
        end if

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        if (proc_rank == 0) then
            call MPI_FILE_OPEN(MPI_COMM_SELF, file_loc, ior(MPI_MODE_WRONLY, MPI_MODE_CREATE), mpi_info_int, ifile, ierr)

            ! Write header using MPI I/O for consistency
            call MPI_FILE_WRITE(ifile, tot_part, 1, MPI_INTEGER, status, ierr)
            call MPI_FILE_WRITE(ifile, mytime, 1, mpi_p, status, ierr)
            call MPI_FILE_WRITE(ifile, dt, 1, mpi_p, status, ierr)
            call MPI_FILE_WRITE(ifile, num_procs, 1, MPI_INTEGER, status, ierr)
            call MPI_FILE_WRITE(ifile, proc_bubble_counts, num_procs, MPI_INTEGER, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)
        end if

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        if (bub_id > 0) then
            allocate (MPI_IO_DATA_lag_bubbles(max(1, bub_id),1:lag_io_vars))

            i = 1
            do k = 1, nBubs
                if (particle_in_domain_physical(mtn_pos(k,1:3,1))) then
                    MPI_IO_DATA_lag_bubbles(i, 1) = real(lag_id(k, 1))
                    MPI_IO_DATA_lag_bubbles(i,2:4) = mtn_pos(k,1:3,1)
                    MPI_IO_DATA_lag_bubbles(i,5:7) = mtn_posPrev(k,1:3,1)
                    MPI_IO_DATA_lag_bubbles(i,8:10) = mtn_vel(k,1:3,1)
                    MPI_IO_DATA_lag_bubbles(i, 11) = intfc_rad(k, 1)
                    MPI_IO_DATA_lag_bubbles(i, 12) = intfc_vel(k, 1)
                    MPI_IO_DATA_lag_bubbles(i, 13) = bub_R0(k)
                    MPI_IO_DATA_lag_bubbles(i, 14) = Rmax_stats(k)
                    MPI_IO_DATA_lag_bubbles(i, 15) = Rmin_stats(k)
                    MPI_IO_DATA_lag_bubbles(i, 16) = bub_dphidt(k)
                    MPI_IO_DATA_lag_bubbles(i, 17) = gas_p(k, 1)
                    MPI_IO_DATA_lag_bubbles(i, 18) = gas_mv(k, 1)
                    MPI_IO_DATA_lag_bubbles(i, 19) = gas_mg(k)
                    MPI_IO_DATA_lag_bubbles(i, 20) = gas_betaT(k)
                    MPI_IO_DATA_lag_bubbles(i, 21) = gas_betaC(k)
                    ! Marmotant
                    MPI_IO_DATA_lag_bubbles(i, 22) = mrmtnt_shell(k, 1)
                    MPI_IO_DATA_lag_bubbles(i, 23) = mrmtnt_Rbuck(k)
                    MPI_IO_DATA_lag_bubbles(i, 24) = mrmtnt_Rrupt(k)
                    ! hifu
                    MPI_IO_DATA_lag_bubbles(i, 25) = bub_qvis(k)
                    MPI_IO_DATA_lag_bubbles(i, 26) = bub_qth(k)
                    MPI_IO_DATA_lag_bubbles(i, 27) = intfc_ac(k, 1)
                    MPI_IO_DATA_lag_bubbles(i, 28) = bub_hifu_rad(k)
                    i = i + 1
                end if
            end do

            call MPI_TYPE_CREATE_SUBARRAY(2, gsizes, lsizes, start_idx_part, MPI_ORDER_FORTRAN, mpi_p, view, ierr)
            call MPI_TYPE_COMMIT(view, ierr)

            call MPI_FILE_OPEN(MPI_COMM_WORLD, file_loc, ior(MPI_MODE_WRONLY, MPI_MODE_CREATE), mpi_info_int, ifile, ierr)

            ! Skip header (written by rank 0)
            disp = int(sizeof(tot_part) + 2*sizeof(mytime) + sizeof(num_procs) + num_procs*sizeof(proc_bubble_counts(1)), &
                       & MPI_OFFSET_KIND)
            call MPI_FILE_SET_VIEW(ifile, disp, mpi_p, view, 'native', mpi_info_int, ierr)

            call MPI_FILE_WRITE_ALL(ifile, MPI_IO_DATA_lag_bubbles, lag_io_vars*bub_id, mpi_p, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)

            deallocate (MPI_IO_DATA_lag_bubbles)
        else
            call MPI_TYPE_CONTIGUOUS(0, mpi_p, view, ierr)
            call MPI_TYPE_COMMIT(view, ierr)

            call MPI_FILE_OPEN(MPI_COMM_WORLD, file_loc, ior(MPI_MODE_WRONLY, MPI_MODE_CREATE), mpi_info_int, ifile, ierr)

            ! Skip header (written by rank 0)
            disp = int(sizeof(tot_part) + 2*sizeof(mytime) + sizeof(num_procs) + num_procs*sizeof(proc_bubble_counts(1)), &
                       & MPI_OFFSET_KIND)
            call MPI_FILE_SET_VIEW(ifile, disp, mpi_p, view, 'native', mpi_info_int, ierr)

            call MPI_FILE_WRITE_ALL(ifile, dummy, 0, mpi_p, status, ierr)

            call MPI_FILE_CLOSE(ifile, ierr)
        end if

        deallocate (proc_bubble_counts)
#endif

    end subroutine s_write_restart_lag_bubbles

    !> Compute the maximum and minimum radius of each bubble
    subroutine s_calculate_lag_bubble_stats()

        integer :: k

        Rmax_glb = min(dflt_real, -dflt_real)
        Rmin_glb = max(dflt_real, -dflt_real)
        Rmean_glb = 0._wp; lag_vol_glb = 0._wp
        $:GPU_UPDATE(device='[Rmax_glb, Rmin_glb, Rmean_glb, lag_vol_glb]')

        $:GPU_PARALLEL_LOOP(private='[k]', reduction='[[Rmax_glb], [Rmin_glb], [Rmean_glb, lag_vol_glb]]', reductionOp='[MAX, &
                            & MIN, +]', copy='[Rmax_glb, Rmin_glb, Rmean_glb, lag_vol_glb]')
        do k = 1, nBubs
            Rmax_glb = max(Rmax_glb, intfc_rad(k, 1))
            Rmin_glb = min(Rmin_glb, intfc_rad(k, 1))
            Rmean_glb = Rmean_glb + intfc_rad(k, 1)
            lag_vol_glb = lag_vol_glb + (4._wp/3._wp)*pi*intfc_rad(k, 1)**3._wp
            Rmax_stats(k) = max(Rmax_stats(k), intfc_rad(k, 1)/bub_R0(k))
            Rmin_stats(k) = min(Rmin_stats(k), intfc_rad(k, 1)/bub_R0(k))
        end do
        $:END_GPU_PARALLEL_LOOP()

    end subroutine s_calculate_lag_bubble_stats

    !> Write the maximum and minimum radius statistics for each bubble
    impure subroutine s_write_lag_bubble_stats()

        integer                              :: k
        character(LEN=path_len + 2*name_len) :: file_loc
        character(len=20)                    :: FMT

        write (file_loc, '(A,I0,A)') 'stats_lag_bubbles_', proc_rank, '.dat'
        file_loc = trim(case_dir) // '/D/' // trim(file_loc)

        $:GPU_UPDATE(host='[Rmax_glb, Rmin_glb]')

        if (precision == precision_single) then
            FMT = "(A10,A14,5A16)"
        else
            FMT = "(A10,A14,5A24)"
        end if

        open (13, FILE=trim(file_loc), form='formatted', position='rewind')
        write (13, FMT) 'proc_rank', 'particleID', 'x', 'y', 'z', 'Rmax_glb', 'Rmin_glb'

        if (precision == precision_single) then
            FMT = "(I10,I14,5F16.8)"
        else
            FMT = "(I10,I14,5F24.16)"
        end if

        do k = 1, nBubs
            write (13, FMT) proc_rank, lag_id(k, 1), mtn_pos(k, 1, 1), mtn_pos(k, 2, 1), mtn_pos(k, 3, 1), Rmax_stats(k), &
                   & Rmin_stats(k)
        end do

        close (13)

    end subroutine s_write_lag_bubble_stats

    !> Remove a specific Lagrangian bubble when dt becomes too small
    impure subroutine s_remove_lag_bubble(bub_id)

        integer, intent(in) :: bub_id
        integer             :: i

        $:GPU_LOOP(parallelism='[seq]')
        do i = bub_id, nBubs - 1
            if (i == bub_id) print *, 'In loop remove bub:', i
            lag_id(i, 1) = lag_id(i + 1, 1)
            bub_R0(i) = bub_R0(i + 1)
            Rmax_stats(i) = Rmax_stats(i + 1)
            Rmin_stats(i) = Rmin_stats(i + 1)
            gas_mg(i) = gas_mg(i + 1)
            gas_betaT(i) = gas_betaT(i + 1)
            gas_betaC(i) = gas_betaC(i + 1)
            bub_dphidt(i) = bub_dphidt(i + 1)
            gas_p(i,1:2) = gas_p(i + 1,1:2)
            gas_mv(i,1:2) = gas_mv(i + 1,1:2)
            intfc_rad(i,1:2) = intfc_rad(i + 1,1:2)
            intfc_vel(i,1:2) = intfc_vel(i + 1,1:2)
            intfc_ac(i,1:2) = intfc_ac(i + 1,1:2)
            mtn_pos(i,1:3,1:2) = mtn_pos(i + 1,1:3,1:2)
            mtn_posPrev(i,1:3,1:2) = mtn_posPrev(i + 1,1:3,1:2)
            mtn_vel(i,1:3,1:2) = mtn_vel(i + 1,1:3,1:2)
            mtn_s(i,1:3,1:2) = mtn_s(i + 1,1:3,1:2)
            intfc_draddt(i,1:lag_num_ts) = intfc_draddt(i + 1,1:lag_num_ts)
            intfc_dveldt(i,1:lag_num_ts) = intfc_dveldt(i + 1,1:lag_num_ts)
            gas_dpdt(i,1:lag_num_ts) = gas_dpdt(i + 1,1:lag_num_ts)
            gas_dmvdt(i,1:lag_num_ts) = gas_dmvdt(i + 1,1:lag_num_ts)
            mtn_dposdt(i,1:3,1:lag_num_ts) = mtn_dposdt(i + 1,1:3,1:lag_num_ts)
            mtn_dveldt(i,1:3,1:lag_num_ts) = mtn_dveldt(i + 1,1:3,1:lag_num_ts)
            mrmtnt_shell(i,1:2) = mrmtnt_shell(i + 1,1:2)
            mrmtnt_Rbuck(i) = mrmtnt_Rbuck(i + 1)
            mrmtnt_Rrupt(i) = mrmtnt_Rrupt(i + 1)
            bub_qvis(i) = bub_qvis(i + 1)
            bub_qth(i) = bub_qth(i + 1)
            bub_hifu_rad(i) = bub_hifu_rad(i + 1)
        end do

        nBubs = nBubs - 1
        dt = 5._wp*dt
        $:GPU_UPDATE(device='[nBubs, dt]')

        print *, 'Bubble removed, nBubs now: in processor: ', nBubs, proc_rank

    end subroutine s_remove_lag_bubble

    ! subroutine s_free_memory_stg3()

    ! @:DEALLOCATE(Rmax_stats) @:DEALLOCATE(Rmin_stats) @:DEALLOCATE(gas_mg) @:DEALLOCATE(gas_betaT) @:DEALLOCATE(gas_betaC)
    ! @:DEALLOCATE(bub_dphidt) @:DEALLOCATE(gas_p) @:DEALLOCATE(gas_mv) @:DEALLOCATE(intfc_ac) @:DEALLOCATE(mtn_vel)
    ! @:DEALLOCATE(intfc_draddt) @:DEALLOCATE(intfc_dveldt) @:DEALLOCATE(gas_dpdt) @:DEALLOCATE(gas_dmvdt) @:DEALLOCATE(mtn_dposdt)
    ! @:DEALLOCATE(mtn_dveldt) ! Marmotant model @:DEALLOCATE(mrmtnt_shell) @:DEALLOCATE(mrmtnt_Rbuck) @:DEALLOCATE(mrmtnt_Rrupt) !
    ! bubble interaction @:DEALLOCATE(bub_interact) if (lag_params%pressure_corrector .and. any(lag_params%interaction_model == (/2,
    ! 3/))) then @:DEALLOCATE(bub_int_ids) end if !@:DEALLOCATE(bub_lambda_c) if (hifu_params%moments) then
    ! @:DEALLOCATE(moments_bubs) end if if (hifu_params%power_balance) then @:DEALLOCATE(acPw_bubs) end if

    ! end subroutine s_free_memory_stg3

    !> The purpose of this subroutine is to deallocate variables
    impure subroutine s_finalize_lagrangian_solver()

        integer :: i

        do i = 1, q_beta_idx
            @:DEALLOCATE(q_beta%vf(i)%sf)
        end do
        @:DEALLOCATE(q_beta%vf)

        ! Deallocating space
        @:DEALLOCATE(lag_id)
        @:DEALLOCATE(bub_R0)
        @:DEALLOCATE(intfc_rad)
        @:DEALLOCATE(intfc_vel)
        @:DEALLOCATE(mtn_pos)
        @:DEALLOCATE(mtn_posPrev)
        @:DEALLOCATE(mtn_s)
        ! hifu
        @:DEALLOCATE(bub_qvis)
        @:DEALLOCATE(bub_qth)
        @:DEALLOCATE(bub_hifu_rad)
        @:DEALLOCATE(bub_rho)

        ! if (.not. hifu_params%heatSolver) then
        @:DEALLOCATE(Rmax_stats)
        @:DEALLOCATE(Rmin_stats)
        @:DEALLOCATE(gas_mg)
        @:DEALLOCATE(gas_betaT)
        @:DEALLOCATE(gas_betaC)
        @:DEALLOCATE(bub_dphidt)
        @:DEALLOCATE(gas_p)
        @:DEALLOCATE(gas_mv)
        @:DEALLOCATE(intfc_ac)
        @:DEALLOCATE(mtn_vel)
        @:DEALLOCATE(intfc_draddt)
        @:DEALLOCATE(intfc_dveldt)
        @:DEALLOCATE(gas_dpdt)
        @:DEALLOCATE(gas_dmvdt)
        @:DEALLOCATE(mtn_dposdt)
        @:DEALLOCATE(mtn_dveldt)
        ! Marmotant model
        @:DEALLOCATE(mrmtnt_shell)
        @:DEALLOCATE(mrmtnt_Rbuck)
        @:DEALLOCATE(mrmtnt_Rrupt)
        ! bubble interaction
        @:DEALLOCATE(bub_interact)
        if (lag_params%pressure_corrector .and. any(lag_params%interaction_model == (/2, 3/))) then
            @:DEALLOCATE(bub_int_ids)
        end if
        !@:DEALLOCATE(bub_lambda_c)
        if (hifu_params%moments) then
            @:DEALLOCATE(moments_bubs)
        end if
        if (hifu_params%power_balance) then
            @:DEALLOCATE(acPw_bubs)
        end if
        ! end if

    end subroutine s_finalize_lagrangian_solver

end module m_bubbles_EL
