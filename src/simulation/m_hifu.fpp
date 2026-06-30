!>
!! @file m_hifu.f90
!! @brief Contains module m_hifu

#:include 'macros.fpp'

!> @brief The module contains the subroutines used to study HIFU
module m_hifu

    use m_derived_types         !< Definitions of the derived types
    use m_global_parameters     !< Definitions of the global parameters
    use m_mpi_proxy             !< Message passing interface (MPI) module proxy
    use m_variables_conversion  !< State variables type conversion procedures
    use m_bubbles_EL
    use m_bubbles_EL_kernels
    use m_helper

    implicit none

    type(vector_field) :: q_hifu  !< HIFU vector fields
    $:GPU_DECLARE(create='[q_hifu]')

    real(wp), allocatable, dimension(:) :: shear_viscous_fluids, bulk_viscous_fluids, abs_coef_fluids, rho_cp_fluids, tdiff_fluids
    $:GPU_DECLARE(create='[shear_viscous_fluids, bulk_viscous_fluids, abs_coef_fluids, rho_cp_fluids, tdiff_fluids]')

    integer :: bc_pole, sys_size_hyd
    $:GPU_DECLARE(create='[bc_pole, sys_size_hyd]')

contains

    !> Initializes the hifu model
    subroutine s_initialize_HIFU_module()

        integer :: i

        sys_size_hyd = sys_size

        ! Allocating the cell-average RHS variables
        @:ALLOCATE(q_hifu%vf(1:sys_size_hifu))

        do i = 1, sys_size_hifu
            @:ALLOCATE(q_hifu%vf(i)%sf(idwbuff(1)%beg:idwbuff(1)%end, idwbuff(2)%beg:idwbuff(2)%end, idwbuff(3)%beg:idwbuff(3)%end))
        end do
        @:ACC_SETUP_VFs(q_hifu)

        ! Fluids' properties needed (GPU)
        @:ALLOCATE(shear_viscous_fluids(1: num_fluids))
        @:ALLOCATE(bulk_viscous_fluids(1: num_fluids))
        @:ALLOCATE(abs_coef_fluids(1: num_fluids))
        @:ALLOCATE(rho_cp_fluids(1: num_fluids))
        @:ALLOCATE(tdiff_fluids(1: num_fluids))

        do i = 1, num_fluids
            shear_viscous_fluids(i) = fluid_pp(i)%Re(1)
            bulk_viscous_fluids(i) = fluid_pp(i)%Re(2)
            abs_coef_fluids(i) = fluid_pp(i)%absCoef
            rho_cp_fluids(i) = fluid_pp(i)%rho_cp
            tdiff_fluids(i) = fluid_pp(i)%tdiff
        end do
        $:GPU_UPDATE(device='[sys_size_hyd, shear_viscous_fluids, bulk_viscous_fluids, abs_coef_fluids, rho_cp_fluids, tdiff_fluids]')

    end subroutine s_initialize_HIFU_module

    subroutine s_start_HIFU_indexes(stg)

        integer, intent(in) :: stg

        if (stg == 2) then
            hifu_idx%qac = 1
            hifu_idx%qac_prms = 2
            hifu_idx%tsamp = 3
            hifu_idx%P = 4
        end if

        if (stg == 3) then
            hifu_idx%T = 1
            hifu_idx%qac = 3
            hifu_idx%qvis = 4
            hifu_idx%qth = 5
        end if

    end subroutine s_start_HIFU_indexes

    !> Populate HIFU vars with user inputs and zeroing the time-averaged vars.
    subroutine s_initialize_sampling_vars()

        integer :: i, j, k, l

        ! Zeroing all the hifu variables

        $:GPU_PARALLEL_LOOP(private='[i, j, k, l]', collapse=4)
        do l = 1, sys_size_hifu
            do k = idwbuff(3)%beg, idwbuff(3)%end
                do j = idwbuff(2)%beg, idwbuff(2)%end
                    do i = idwbuff(1)%beg, idwbuff(1)%end
                        q_hifu%vf(l)%sf(i, j, k) = 0._wp
                    end do
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        ! Initialize Pmin and Pmax
        $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
        do k = idwbuff(3)%beg, idwbuff(3)%end
            do j = idwbuff(2)%beg, idwbuff(2)%end
                do i = idwbuff(1)%beg, idwbuff(1)%end
                    q_hifu%vf(hifu_idx%P)%sf(i, j, k) = min(dflt_real, -dflt_real)
                    q_hifu%vf(hifu_idx%P + 1)%sf(i, j, k) = max(dflt_real, -dflt_real)
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        call s_open_run_time_information_samplingHIFU()

        if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 2: sampling heat sources'

    end subroutine s_initialize_sampling_vars

    subroutine s_restart_hifu_stages()

        !!!!>>> NO GPU NEEDED <<<!!!!!!!

        ! Starting fresh
        if (cfl_dt) then
            if (n_start == 0) then
                if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 1: developing hydrodynamic field'
                return
            end if
        else
            if (t_step_start == 0) then
                if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 1: developing hydrodynamic field'
                return
            end if
        end if

        ! Restart during stg1
        if (hifu_params%stg1) then
            if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 1 -> restarting'
            return
        end if

        ! Restart during stg2
        if (hifu_params%stg2) then
            if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 2 -> restarting'
            call s_start_HIFU_indexes(stg=2)
            hifu_params%sampling = .true.
            hifu_params%heatSolver = .false.
            if (cfl_dt) then
                t_stop = hifu_params%t_stop_stg2
            else
                t_step_stop = hifu_params%t_step_stop_stg2
            end if
            return
        end if

        ! Restart during stg3
        if (hifu_params%stg3) then
            call s_start_HIFU_indexes(stg=3)
            hifu_params%sampling = .false.
            hifu_params%heatSolver = .true.
            cfl_dt = .false.
            dt = hifu_params%dt_stg3
            if (f_is_default(hifu_params%dt_stg3)) call s_mpi_abort('dt_stg3 not defined')
            t_step_save = hifu_params%t_step_save_stg3
            t_step_stop = hifu_params%t_step_stop_stg3 - 1
            finaltime = t_step_stop*dt

            if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 3 -> restarting'
        end if

    end subroutine s_restart_hifu_stages

    !> The idea is to jump form one stage into the other with this procedure
    subroutine s_HIFU_stages(t_step, hifu_write_output, exitFlag)

        integer, intent(inout) :: t_step
        logical, intent(out)   :: hifu_write_output
        logical, intent(inout) :: exitFlag

        hifu_write_output = .false.

        ! 1st to 2nd stage
        if (cfl_dt) then
            if (mytime >= hifu_params%t_stop_stg1 .and. .not. hifu_params%sampling) then
                ! Define params to start stage 2 (Apadt dt) Stg 2 uses the same dt as in stg 1
                if (.not. hifu_params%stg2) return

                dt = hifu_params%dt_stg2
                t_stop = hifu_params%t_stop_stg2
                call s_start_HIFU_indexes(stg=2)
                hifu_params%sampling = .true.
                hifu_params%heatSolver = .false.
                $:GPU_UPDATE(device='[dt, hifu_params, hifu_idx]')

                call s_initialize_sampling_vars()
                hifu_write_output = .true.
                exitFlag = .false.
                return
            end if
        else
            if (t_step == hifu_params%t_step_stop_stg1 .and. .not. hifu_params%sampling) then
                ! Define params to start stage 2 (constant dt) Stg 2 uses the same dt as in stg 1
                if (.not. hifu_params%stg2) return

                dt = hifu_params%dt_stg2
                t_step_stop = hifu_params%t_step_stop_stg2
                finaltime = t_step_stop*dt
                call s_start_HIFU_indexes(stg=2)
                hifu_params%sampling = .true.
                hifu_params%heatSolver = .false.
                $:GPU_UPDATE(device='[hifu_params, dt, hifu_idx]')

                call s_initialize_sampling_vars()
                hifu_write_output = .true.
                exitFlag = .false.
                return
            end if
        end if

        ! 2nd to 3rd stage
        if ((cfl_dt .and. mytime >= hifu_params%t_stop_stg2 .and. .not. hifu_params%heatSolver) .or. (.not. cfl_dt &
            & .and. t_step == hifu_params%t_step_stop_stg2 .and. .not. hifu_params%heatSolver)) then
            ! Define params to start stage 3 (constant dt only)
            call s_close_run_time_information_samplingHIFU()
            if (.not. hifu_params%stg3) return

            hifu_params%sampling = .false.
            hifu_params%heatSolver = .true.
            cfl_dt = .false.
            dt = hifu_params%dt_stg3
            if (f_is_default(hifu_params%dt_stg3)) call s_mpi_abort('dt_stg3 not defined')
            t_step_save = hifu_params%t_step_save_stg3
            t_step_stop = hifu_params%t_step_stop_stg3 - 1
            finaltime = t_step_stop*dt

            mytime = 0._wp
            t_step_start = 0
            t_step = 0

            if (p == 0) then
                if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 3: solving heat equation (2D)'
            else
                if (proc_rank == 0) print *, 'WARNING :: HIFU -> Stage 3: solving heat equation (3D)'
                if (bc_x%beg == -20) bc_x%beg = -6
                $:GPU_UPDATE(device='[bc_x]')
            end if

            exitFlag = .false.

            $:GPU_UPDATE(device='[hifu_params, dt, hifu_idx]')
        end if

    end subroutine s_HIFU_stages

    !> The purpose of this procedure is to take samples needed to calculate the time averaged heat sources. It calculates the
    !! generated heat source "q_us_ac", from the primary ultrasound source.
    !! @param q_cons_vf Conservative variables
    !! @param q_prim_vf Primitive variables
    !! @param t_step Current time step @param hdid Physical period advanced in the last time step.
    subroutine s_update_HIFU_vars_sampling(q_cons_vf, q_prim_vf, t_step, hdid)

        type(scalar_field), dimension(sys_size), intent(in) :: q_cons_vf
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        integer, intent(in)                                 :: t_step
        real(wp), intent(in)                                :: hdid
        real(wp)                                            :: rho_h, pres_h, gamma_h, pi_inf_h, T_h, c_c_h
        real(wp), dimension(num_dims)                       :: vel_h
        real(wp)                                            :: qv_h, cson_h
        real(wp), dimension(2)                              :: Re_h
        real(wp), dimension(eqn_idx%cont%end)               :: myalpha_rho, myalpha
        real(wp)                                            :: rhoYks_h(1:num_species)
        logical                                             :: axialCondition, radialCondition, condition
        real(wp)                                            :: shearVisc, bulkVisc, absCoef
        real(wp)                                            :: varA, varB
        real(wp)                                            :: duxdx, duxdr, durdx, durdr, ep11, ep22, ep33, ep12, ep13, ep23
        real(wp), dimension(3)                              :: duxdn, duydn, duzdn
        real(wp)                                            :: intensity_ac, sum_qac, tmp, intensity_ac_prms, sum_qac_prms
        integer                                             :: i, j, k, l, s, mtd_idx
        integer                                             :: abortFlag, abortFlag_max
        real(wp), dimension(1:4)                            :: mom_qac
        real(wp)                                            :: dist_radial, vol_cell, xb_Rc
        integer                                             :: n_sgn
        real(wp)                                            :: acPw, acPw_qac, acPw_cmprssv, acPw_kntc
        real(wp), dimension(1:6)                            :: acPw_in_dt, acPw_out_dt
        logical                                             :: flg_cell_in_cv
        character(len=512)                                  :: line

        if (hifu_params%moments) mom_qac(1:4) = 0._wp

        sum_qac = 0._wp; sum_qac_prms = 0._wp

        if (hifu_params%power_balance) then
            acPw_in_dt(1:6) = 0._wp; acPw_out_dt(1:6) = 0._wp; acPw_qac = 0._wp
            acPw_cmprssv = 0._wp; acPw_kntc = 0._wp
        end if

        if (bubbles_lagrange .and. .not. adap_dt) call s_compute_bubble_heat_sources_HIFU(hdid)

        abortFlag_max = 0

        if (cyl_coord .and. p == 0) then  ! Axysimetric

#ifdef MFC_DEBUG
            if (proc_rank == 0) print*, 'Computing axysimetric acoustic damping', mytime, hdid
#endif

            $:GPU_PARALLEL_LOOP(collapse=3, reduction='[[abortFlag_max], [sum_qac, sum_qac_prms]]', reductionOp='[MAX, +]', &
                                & private='[myalpha_rho, myalpha, vel_h, Re_h, rhoYks_h]', copy='[sum_qac, sum_qac_prms, abortFlag_max]')
            do l = 0, p
                do k = 0, n
                    do j = 0, m
                        abortFlag = 0

                        ! Get viscosities (and absorption coeff.) which are user inputs
                        shearVisc = 0._wp
                        bulkVisc = 0._wp
                        absCoef = 0._wp

                        $:GPU_LOOP(parallelism='[seq]')
                        do i = 1, num_fluids
                            shearVisc = shearVisc + q_prim_vf(eqn_idx%E + i)%sf(j, k, l)*shear_viscous_fluids(i)
                            bulkVisc = bulkVisc + q_prim_vf(eqn_idx%E + i)%sf(j, k, l)*bulk_viscous_fluids(i)
                            absCoef = absCoef + q_prim_vf(eqn_idx%E + i)%sf(j, k, l)*abs_coef_fluids(i)
                        end do

                        shearVisc = 1._wp/shearVisc
                        bulkVisc = 1._wp/bulkVisc

                        if (f_is_default(absCoef)) then
                            print*, "HIFU: Check absCoef values!"
                            abortFlag = 1
                        end if

                        !> > Get the strain rate tensor (using central finite difference)
                        varA = 0._wp
                        varB = 0._wp

                        ! Only for axysimmetric assumption
                        duxdx = (q_prim_vf(eqn_idx%cont%end + 1)%sf(j + 1, k, 0) - q_prim_vf(eqn_idx%cont%end + 1)%sf(j - 1, k, &
                                 & 0))/(x_cc(j + 1) - x_cc(j - 1))
                        duxdr = (q_prim_vf(eqn_idx%cont%end + 1)%sf(j, k + 1, 0) - q_prim_vf(eqn_idx%cont%end + 1)%sf(j, k - 1, &
                                 & 0))/(y_cc(k + 1) - y_cc(k - 1))

                        durdx = (q_prim_vf(eqn_idx%cont%end + 2)%sf(j + 1, k, 0) - q_prim_vf(eqn_idx%cont%end + 2)%sf(j - 1, k, &
                                 & 0))/(x_cc(j + 1) - x_cc(j - 1))
                        durdr = (q_prim_vf(eqn_idx%cont%end + 2)%sf(j, k + 1, 0) - q_prim_vf(eqn_idx%cont%end + 2)%sf(j, k - 1, &
                                 & 0))/(y_cc(k + 1) - y_cc(k - 1))

                        !> > Get pressure, density and speed of sound
                        call s_compute_species_fraction(q_prim_vf, j, k, l, myalpha_rho, myalpha)
                        call s_convert_species_to_mixture_variables_acc(rho_h, gamma_h, pi_inf_h, qv_h, myalpha, myalpha_rho, Re_h)

                        $:GPU_LOOP(parallelism='[seq]')
                        do s = 1, num_dims
                            vel_h(s) = q_cons_vf(s + eqn_idx%cont%end)%sf(j, k, l)/rho_h
                        end do

                        call s_compute_pressure(q_cons_vf(eqn_idx%E)%sf(j, k, l), 0._wp, 0.5_wp*rho_h*dot_product(vel_h, vel_h), &
                                                & pi_inf_h, gamma_h, rho_h, qv_h, rhoYks_h, pres_h, T_h)

                        call s_compute_speed_of_sound(pres_h, rho_h, gamma_h, pi_inf_h, &
                                                      & ((gamma_h + 1._wp)*pres_h + pi_inf_h + qv_h)/rho_h, myalpha, 0._wp, &
                                                      & 0._wp, cson_h, qv_h)

                        ! Obtaining Pmax and Pmin fields
                        q_hifu%vf(hifu_idx%P)%sf(j, k, l) = max(q_hifu%vf(hifu_idx%P)%sf(j, k, l), pres_h)
                        q_hifu%vf(hifu_idx%P + 1)%sf(j, k, l) = min(q_hifu%vf(hifu_idx%P + 1)%sf(j, k, l), pres_h)

                        !> > Compute intensity form acoustic damping

                        ! PRMS method (calculate only during the last time step in stage2 -> need developed Pmax field)
                        intensity_ac_prms = 0._wp
                        if (cfl_dt) then
                            if (mytime + dt >= t_stop) then
                                intensity_ac_prms = absCoef*(q_hifu%vf(hifu_idx%P)%sf(j, k, &
                                                             & l) - hifu_params%atmPres)**2._wp/(rho_h*cson_h)
                                if (proc_rank == 0 .and. j == 0 .and. k == 0 .and. l == 0) print*, 'Calculated intensity_ac_prms'
                            end if
                        else
                            if (t_step == t_step_stop - 1) then
                                intensity_ac_prms = absCoef*(q_hifu%vf(hifu_idx%P)%sf(j, k, &
                                                             & l) - hifu_params%atmPres)**2._wp/(rho_h*cson_h)
                                if (proc_rank == 0 .and. j == 0 .and. k == 0 .and. l == 0) print*, 'Calculated intensity_ac_prms'
                            end if
                        end if

                        ! Shear stress method
                        intensity_ac = 0._wp
                        ep11 = durdr
                        ep22 = vel_h(2)/y_cc(k)
                        ep33 = duxdx
                        ep13 = 0.5_wp*(durdx + duxdr)
                        varA = ep11**2._wp + ep22**2._wp + ep33**2._wp
                        varB = (8._wp/3._wp)*varA - (4._wp/3._wp)*(ep11*ep22 + ep11*ep33 + ep22*ep33) + 6._wp*(ep13**2._wp)
                        intensity_ac = intensity_ac + bulkVisc*varA + 2._wp*shearVisc*varB  ! intensity is "q_us_ac"

                        ! Update total sampling time
                        q_hifu%vf(hifu_idx%tsamp)%sf(j, k, l) = q_hifu%vf(hifu_idx%tsamp)%sf(j, k, l) + hdid
                        ! Sampling acoustic intensity
                        q_hifu%vf(hifu_idx%qac)%sf(j, k, l) = q_hifu%vf(hifu_idx%qac)%sf(j, k, l) + intensity_ac*hdid
                        ! Sampling acoustic intensity (prms)
                        q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l) = intensity_ac_prms*q_hifu%vf(hifu_idx%tsamp)%sf(j, k, l)

                        ! Checking for NaNs
                        if (q_hifu%vf(hifu_idx%qac)%sf(j, k, l) /= q_hifu%vf(hifu_idx%qac)%sf(j, k, l)) then
                            print*, 'Acoustic intensity is NaN', j, k, l, hdid, intensity_ac
                            print*, 'viscosities (bulk & shear)', bulkVisc, shearVisc
                            print*, 'var: A, B', varA, varB, ep11, ep22, ep33, ep13
                            print*, 'ep22:', vel_h(2), y_cc(k), rho_h
                            abortFlag = 1
                        end if

                        if (q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l) /= q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l)) then
                            print*, 'Acoustic intensity PRMS is NaN', j, k, l, hdid, intensity_ac_prms, absCoef, &
                                & q_hifu%vf(hifu_idx%P)%sf(j, k, l), hifu_params%atmPres, rho_h, cson_h
                            abortFlag = 1
                        end if

                        abortFlag_max = max(abortFlag_max, abortFlag)

                        ! Update average velocities for streaming if (hifu_params%streaming) then q_hifu%vf(hifu_idx%u)%sf(j,
                        ! k, l) = q_hifu%vf(hifu_idx%u)%sf(j, k, & & l) + vel_h(1)*hdid ! Sampling x-vel
                        ! q_hifu%vf(hifu_idx%v)%sf(j, k, l) = q_hifu%vf(hifu_idx%v)%sf(j, k, & & l) + vel_h(2)*hdid !
                        ! Sampling y-vel end if

                        ! Intensity summation through the domain
                        sum_qac = sum_qac + q_hifu%vf(hifu_idx%qac)%sf(j, k, l)
                        sum_qac_prms = sum_qac_prms + q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l)
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            if (abortFlag_max > 0) stop "NaNs in Acoustic intensity (prms)"

            if (num_procs > 1) then
                tmp = sum_qac
                call s_mpi_allreduce_sum(tmp, sum_qac)

                tmp = sum_qac_prms
                call s_mpi_allreduce_sum(tmp, sum_qac_prms)
            end if

            $:GPU_UPDATE(host='[q_hifu%vf(hifu_idx%tsamp)%sf]')

            if (proc_rank == 0) then
                write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime, q_hifu%vf(hifu_idx%tsamp)%sf(0, 0, 0), &
                       & sum_qac, sum_qac_prms
                write (99, '(A)') trim(line)
            end if
        else if (.not. cyl_coord .and. p > 0) then  ! Cartesian 3D
#ifdef MFC_DEBUG
            if (proc_rank == 0) print*, 'Computing cartesian 3D acoustic damping', mytime, hdid
#endif

            $:GPU_PARALLEL_LOOP(collapse=3, reduction='[[abortFlag_max], [sum_qac, sum_qac_prms, acPw_qac, acPw_cmprssv, &
                                & acPw_kntc], [mom_qac(1:4), acPw_in_dt(1:6), acPw_out_dt(1:6)]]', reductionOp='[MAX, +, +]', &
                                & private='[i, j, k, l, myalpha_rho, myalpha, vel_h, Re_h, rhoYks_h, duxdn, duydn, duzdn, xb_Rc, &
                                & vol_cell]', copy='[abortFlag_max, sum_qac, sum_qac_prms, acPw_qac, acPw_cmprssv, acPw_kntc, &
                                & mom_qac(1:4), acPw_in_dt(1:6), acPw_out_dt(1:6)]')
            do l = 0, p
                do k = 0, n
                    do j = 0, m
                        abortFlag = 0

                        ! Filter the cells inside the spherical bubble cloud
                        if (hifu_params%moments) dist_radial = sqrt((x_cc(j) - hifu_params%cloud_center(1))**2._wp + (y_cc(k) &
                            & - hifu_params%cloud_center(2))**2._wp + (z_cc(l) - hifu_params%cloud_center(3))**2._wp)

                        ! Get viscosities (and absorption coeff.) which are user inputs
                        shearVisc = 0._wp
                        bulkVisc = 0._wp
                        absCoef = 0._wp

                        $:GPU_LOOP(parallelism='[seq]')
                        do i = 1, num_fluids
                            shearVisc = shearVisc + q_prim_vf(eqn_idx%E + i)%sf(j, k, l)*shear_viscous_fluids(i)
                            bulkVisc = bulkVisc + q_prim_vf(eqn_idx%E + i)%sf(j, k, l)*bulk_viscous_fluids(i)
                            absCoef = absCoef + q_prim_vf(eqn_idx%E + i)%sf(j, k, l)*abs_coef_fluids(i)
                        end do
                        shearVisc = 1._wp/shearVisc
                        bulkVisc = 1._wp/bulkVisc

                        if (f_is_default(absCoef)) then
                            abortFlag = 1
                            print*, "HIFU: Check absCoef values!"
                        end if

                        !> > Get the strain rate tensor (using central finite difference)
                        varA = 0._wp
                        varB = 0._wp

                        mtd_idx = 2
                        call s_space_derivative(q_prim_vf(eqn_idx%cont%end + 1), j, k, l, duxdn, mtd_idx)
                        call s_space_derivative(q_prim_vf(eqn_idx%cont%end + 2), j, k, l, duydn, mtd_idx)
                        call s_space_derivative(q_prim_vf(eqn_idx%cont%end + 3), j, k, l, duzdn, mtd_idx)

                        !> > Get pressure, density and speed of sound
                        $:GPU_LOOP(parallelism='[seq]')
                        do i = 1, eqn_idx%cont%end
                            myalpha_rho(i) = q_prim_vf(i)%sf(j, k, l)
                            myalpha(i) = q_prim_vf(eqn_idx%E + i)%sf(j, k, l)
                        end do

                        call s_convert_species_to_mixture_variables_acc(rho_h, gamma_h, pi_inf_h, qv_h, myalpha, myalpha_rho, Re_h)

                        $:GPU_LOOP(parallelism='[seq]')
                        do s = 1, num_dims
                            vel_h(s) = q_cons_vf(s + eqn_idx%cont%end)%sf(j, k, l)/rho_h
                        end do
                        call s_compute_pressure(q_cons_vf(eqn_idx%E)%sf(j, k, l), 0._wp, 0.5_wp*rho_h*dot_product(vel_h, vel_h), &
                                                & pi_inf_h, gamma_h, rho_h, qv_h, rhoYks_h, pres_h, T_h)
                        call s_compute_speed_of_sound(pres_h, rho_h, gamma_h, pi_inf_h, &
                                                      & ((gamma_h + 1._wp)*pres_h + pi_inf_h + qv_h)/rho_h, myalpha, 0._wp, &
                                                      & 0._wp, cson_h, qv_h)

                        ! Obtaining Pmax and Pmin fields
                        q_hifu%vf(hifu_idx%P)%sf(j, k, l) = max(q_hifu%vf(hifu_idx%P)%sf(j, k, l), pres_h)
                        q_hifu%vf(hifu_idx%P + 1)%sf(j, k, l) = min(q_hifu%vf(hifu_idx%P + 1)%sf(j, k, l), pres_h)

                        !> > Compute intensity form acoustic damping

                        ! PRMS method (calculate only during the last time step in stage2 -> need developed Pmax field)
                        intensity_ac_prms = 0._wp
                        if (cfl_dt) then
                            if (mytime + dt >= t_stop) then
                                intensity_ac_prms = absCoef*(q_hifu%vf(hifu_idx%P)%sf(j, k, &
                                                             & l) - hifu_params%atmPres)**2._wp/(rho_h*cson_h)
                                if (proc_rank == 0 .and. j == 0 .and. k == 0 .and. l == 0) print*, 'Calculated intensity_ac_prms'
                            end if
                        else
                            if (t_step == t_step_stop - 1) then
                                intensity_ac_prms = absCoef*(q_hifu%vf(hifu_idx%P)%sf(j, k, &
                                                             & l) - hifu_params%atmPres)**2._wp/(rho_h*cson_h)
                                if (proc_rank == 0 .and. j == 0 .and. k == 0 .and. l == 0) print*, 'Calculated intensity_ac_prms'
                            end if
                        end if

                        ! Shear stress method: Intensity is "q_us_ac"
                        intensity_ac = 0._wp
                        ep11 = duxdn(1)
                        ep22 = duydn(2)
                        ep33 = duzdn(3)
                        ep12 = 0.5_wp*(duxdn(2) + duydn(1))
                        ep13 = 0.5_wp*(duxdn(3) + duzdn(1))
                        ep23 = 0.5_wp*(duydn(3) + duzdn(2))

                        varA = ep11**2._wp + ep22**2._wp + ep33**2._wp + 2._wp*(ep11*ep22 + ep11*ep33 + ep22*ep33)
                        varB = ep11**2._wp + ep22**2._wp + ep33**2._wp + 2._wp*(ep12**2._wp + ep13**2._wp + ep23**2._wp)

                        ! intensity is "q_us_ac"
                        intensity_ac = intensity_ac + bulkVisc*varA + 2._wp*shearVisc*varB - (2._wp/3._wp)*shearVisc*varA

                        ! Update total sampling time
                        q_hifu%vf(hifu_idx%tsamp)%sf(j, k, l) = q_hifu%vf(hifu_idx%tsamp)%sf(j, k, l) + hdid
                        ! Sampling acoustic intensity
                        q_hifu%vf(hifu_idx%qac)%sf(j, k, l) = q_hifu%vf(hifu_idx%qac)%sf(j, k, l) + intensity_ac*hdid
                        ! Sampling acoustic intensity (prms)
                        q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l) = intensity_ac_prms*q_hifu%vf(hifu_idx%tsamp)%sf(j, k, l)

                        ! Checking for NaNs
                        if (q_hifu%vf(hifu_idx%qac)%sf(j, k, l) /= q_hifu%vf(hifu_idx%qac)%sf(j, k, l)) then
                            print*, 'Acoustic intensity is NaN', j, k, l, hdid, intensity_ac
                            print*, 'viscosities (bulk & shear)', bulkVisc, shearVisc
                            print*, 'var: A, B', varA, varB, ep11, ep22, ep33, ep13
                            print*, 'ep22:', vel_h(2), y_cc(k), rho_h
                            abortFlag = 1
                        end if

                        if (q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l) /= q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l)) then
                            print*, 'Acoustic intensity PRMS is NaN', j, k, l, hdid, intensity_ac_prms, absCoef, &
                                & q_hifu%vf(hifu_idx%P)%sf(j, k, l), hifu_params%atmPres, rho_h, cson_h
                            abortFlag = 1
                        end if

                        abortFlag_max = max(abortFlag_max, abortFlag)

                        ! Update average velocities for streaming if (hifu_params%streaming) then q_hifu%vf(hifu_idx%u)%sf(j,
                        ! k, l) = q_hifu%vf(hifu_idx%u)%sf(j, k, & & l) + vel_h(1)*hdid ! Sampling x-vel
                        ! q_hifu%vf(hifu_idx%v)%sf(j, k, l) = q_hifu%vf(hifu_idx%v)%sf(j, k, & & l) + vel_h(2)*hdid !
                        ! Sampling y-vel end if

                        ! Intensity summation through the domain
                        sum_qac = sum_qac + q_hifu%vf(hifu_idx%qac)%sf(j, k, l)
                        sum_qac_prms = sum_qac_prms + q_hifu%vf(hifu_idx%qac_prms)%sf(j, k, l)

                        vol_cell = dx(j)*dy(k)*dz(l)
                        ! Calculate heat source moments inside the bubble cloud
                        if (hifu_params%moments) then
                            if (dist_radial <= hifu_params%R_cloud) then
                                xb_Rc = (x_cc(j) - hifu_params%cloud_center(1))/hifu_params%R_cloud

                                $:GPU_LOOP(parallelism='[seq]')
                                do i = 1, 4
                                    mom_qac(i) = mom_qac(i) + intensity_ac*vol_cell*(xb_Rc)**(i - 1)
                                end do
                            end if
                        end if

                        if (hifu_params%power_balance) then
                            flg_cell_in_cv = f_cell_in_cv(j, k, l)
                            if (flg_cell_in_cv) then
                                ! print*, 'Cell in CV for power balance:', proc_rank, j, k, l
                                acPw_qac = acPw_qac + intensity_ac*vol_cell
                                acPw_cmprssv = acPw_cmprssv + ((pres_h - hifu_params%atmPres)**2._wp/(2._wp*rho_h*cson_h**2._wp)) &
                                                               & *vol_cell
                                acPw_kntc = acPw_kntc + (0.5_wp*rho_h*dot_product(vel_h, vel_h))*vol_cell
                            end if

                            $:GPU_LOOP(parallelism='[seq]')
                            do i = 1, num_dims
                                n_sgn = f_is_on_cv_border(j, k, l, i)
                                if (n_sgn /= 0) then
                                    call s_compute_cv_acoustic_power(q_prim_vf, j, k, l, i, n_sgn, pres_h, vel_h, acPw)

                                    s = 2*i
                                    if (n_sgn == -1) s = s - 1
                                    if (acPw < 0._wp) then
                                        acPw_in_dt(s) = acPw_in_dt(s) + acPw
                                    else
                                        acPw_out_dt(s) = acPw_out_dt(s) + acPw
                                    end if
                                end if
                            end do
                        end if
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            if (abortFlag_max > 0) stop "NaNs in Acoustic intensity"

            call s_write_power_balance(acPw_in_dt, acPw_out_dt, acPw_qac, acPw_cmprssv, acPw_kntc, hdid)

            if (num_procs > 1) then
                tmp = sum_qac
                call s_mpi_allreduce_sum(tmp, sum_qac)
                tmp = sum_qac_prms
                call s_mpi_allreduce_sum(tmp, sum_qac_prms)
            end if

            $:GPU_UPDATE(host='[q_hifu%vf(hifu_idx%tsamp)%sf]')

            if (proc_rank == 0) then
                write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + dt, q_hifu%vf(hifu_idx%tsamp)%sf(0, 0, 0), &
                       & sum_qac, sum_qac_prms
                write (99, '(A)') trim(line)
            end if

            if (hifu_params%moments) call s_write_moments(mom_qac, idx=0)
        else
            call s_mpi_abort('Getting HIFU samples (stage 2) works only with axisymmetric assumption so far!')
        end if

    end subroutine s_update_HIFU_vars_sampling

    function f_cell_in_cv(j, k, l)

        $:GPU_ROUTINE(parallelism='[seq]')
        integer, intent(in) :: j, k, l
        logical             :: f_cell_in_cv

        f_cell_in_cv = .false.
        if ((hifu_params%cv_xb <= x_cb(j - 1)) .and. (x_cb(j) <= hifu_params%cv_xe) .and. (hifu_params%cv_yb <= y_cb(k - 1)) &
            & .and. (y_cb(k) <= hifu_params%cv_ye) .and. (hifu_params%cv_zb <= z_cb(l - 1)) .and. (z_cb(l) <= hifu_params%cv_ze)) &
            & then
            f_cell_in_cv = .true.
        end if

    end function f_cell_in_cv

    subroutine s_write_power_balance(acPw_in_dt, acPw_out_dt, acPw_qac, acPw_cmprssv, acPw_kntc, hdid)

        real(wp), dimension(6), intent(inout) :: acPw_in_dt, acPw_out_dt
        real(wp), intent(inout)               :: acPw_qac, acPw_cmprssv, acPw_kntc
        real(wp), intent(in)                  :: hdid
        real(wp)                              :: var_glb
        integer                               :: i
        character(len=512)                    :: line

        if (num_procs > 1) then
            do i = 1, 6
                call s_mpi_allreduce_sum(acPw_in_dt(i), var_glb)
                acPw_in_dt(i) = var_glb

                call s_mpi_allreduce_sum(acPw_out_dt(i), var_glb)
                acPw_out_dt(i) = var_glb
            end do

            call s_mpi_allreduce_sum(acPw_qac, var_glb)
            acPw_qac = var_glb

            call s_mpi_allreduce_sum(acPw_cmprssv, var_glb)
            acPw_cmprssv = var_glb

            call s_mpi_allreduce_sum(acPw_kntc, var_glb)
            acPw_kntc = var_glb
        end if

        if (proc_rank == 0) then
            write (line, &
                   & '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime &
                   & + hdid, hdid, acPw_in_dt(1), acPw_in_dt(2), acPw_in_dt(3), acPw_in_dt(4), acPw_in_dt(5), acPw_in_dt(6)
            write (89, '(A)') trim(line)

            write (line, &
                   & '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",", ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime &
                   & + hdid, hdid, acPw_out_dt(1), acPw_out_dt(2), acPw_out_dt(3), acPw_out_dt(4), acPw_out_dt(5), acPw_out_dt(6)
            write (88, '(A)') trim(line)

            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + hdid, hdid, acPw_qac, &
                   & acPw_cmprssv, acPw_kntc
            write (87, '(A)') trim(line)
        end if

    end subroutine s_write_power_balance

    subroutine s_compute_cv_acoustic_power(q_prim_vf, j, k, l, idx_dir, n_sgn, pres, vel, acPw)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(scalar_field), dimension(sys_size), intent(in) :: q_prim_vf
        integer, intent(in)                                 :: j, k, l, idx_dir, n_sgn
        real(wp), intent(inout)                             :: pres
        real(wp), dimension(num_dims), intent(inout)        :: vel
        real(wp), dimension(num_dims)                       :: n_vct
        real(wp), intent(out)                               :: acPw
        integer                                             :: i, aux_j, aux_k, aux_l
        real(wp)                                            :: Aface

        if (idx_dir == 1) then
            aux_j = j - n_sgn; aux_k = k; aux_l = l
            n_vct(1) = n_sgn; n_vct(2) = 0._wp; n_vct(3) = 0._wp
            Aface = dy(k)*dz(l)
        else if (idx_dir == 2) then
            aux_j = j; aux_k = k - n_sgn; aux_l = l
            n_vct(1) = 0._wp; n_vct(2) = n_sgn; n_vct(3) = 0._wp
            Aface = dx(j)*dz(l)
            if (bc_y%beg == BC_REFLECTIVE .and. hifu_params%cv_yb == y_cb(k - 1)) then
                aux_j = j; aux_k = k - 1; aux_l = l
            end if
        else if (idx_dir == 3) then
            aux_j = j; aux_k = k; aux_l = l - n_sgn
            n_vct(1) = 0._wp; n_vct(2) = 0._wp; n_vct(3) = n_sgn
            Aface = dx(j)*dy(k)
            if (bc_z%beg == BC_REFLECTIVE .and. hifu_params%cv_zb == z_cb(l - 1)) then
                aux_j = j; aux_k = k; aux_l = l - 1
            end if
        end if

        $:GPU_LOOP(parallelism='[seq]')
        do i = 1, num_dims
            vel(i) = (vel(i) + q_prim_vf(i + eqn_idx%cont%end)%sf(aux_j, aux_k, aux_l))*0.5_wp
        end do

        pres = (pres + q_prim_vf(eqn_idx%E)%sf(aux_j, aux_k, aux_l))*0.5_wp
        pres = pres - hifu_params%atmPres

        acPw = pres*dot_product(vel, n_vct)
        acPW = acPw*Aface

    end subroutine s_compute_cv_acoustic_power

    function f_is_on_cv_border(j, k, l, idx_dir)

        $:GPU_ROUTINE(parallelism='[seq]')
        integer, intent(in) :: j, k, l, idx_dir
        real(wp)            :: aux_xyz
        integer             :: f_is_on_cv_border

        f_is_on_cv_border = 0
        if (idx_dir == 1) then
            if (hifu_params%cv_xb <= x_cb(j) .and. hifu_params%cv_xb > x_cb(j - 1)) then
                if ((y_cb(k - 1) - hifu_params%cv_yb) >= 0._wp .and. (hifu_params%cv_ye - y_cb(k) >= 0._wp)) then
                    if ((z_cb(l - 1) - hifu_params%cv_zb) >= 0._wp .and. (hifu_params%cv_ze - z_cb(l) >= 0._wp)) then
                        f_is_on_cv_border = -1
                    end if
                end if
            end if
            if (hifu_params%cv_xe <= x_cb(j) .and. hifu_params%cv_xe > x_cb(j - 1)) then
                if ((y_cb(k - 1) - hifu_params%cv_yb) >= 0._wp .and. (hifu_params%cv_ye - y_cb(k) >= 0._wp)) then
                    if ((z_cb(l - 1) - hifu_params%cv_zb) >= 0._wp .and. (hifu_params%cv_ze - z_cb(l) >= 0._wp)) then
                        f_is_on_cv_border = 1
                    end if
                end if
            end if
        else if (idx_dir == 2) then
            if (hifu_params%cv_yb <= y_cb(k) .and. hifu_params%cv_yb > y_cb(k - 1)) then
                if ((x_cb(j - 1) - hifu_params%cv_xb) >= 0._wp .and. (hifu_params%cv_xe - x_cb(j) >= 0._wp)) then
                    if ((z_cb(l - 1) - hifu_params%cv_zb) >= 0._wp .and. (hifu_params%cv_ze - z_cb(l) >= 0._wp)) then
                        f_is_on_cv_border = -1
                    end if
                end if
            end if
            if (hifu_params%cv_ye <= y_cb(k) .and. hifu_params%cv_ye > y_cb(k - 1)) then
                if ((x_cb(j - 1) - hifu_params%cv_xb) >= 0._wp .and. (hifu_params%cv_xe - x_cb(j) >= 0._wp)) then
                    if ((z_cb(l - 1) - hifu_params%cv_zb) >= 0._wp .and. (hifu_params%cv_ze - z_cb(l) >= 0._wp)) then
                        f_is_on_cv_border = 1
                    end if
                end if
            end if
            if (bc_y%beg == BC_REFLECTIVE) then
                if (hifu_params%cv_yb == y_cb(k - 1)) then
                    if ((x_cb(j - 1) - hifu_params%cv_xb) >= 0._wp .and. (hifu_params%cv_xe - x_cb(j) >= 0._wp)) then
                        if ((z_cb(l - 1) - hifu_params%cv_zb) >= 0._wp .and. (hifu_params%cv_ze - z_cb(l) >= 0._wp)) then
                            f_is_on_cv_border = -1
                        end if
                    end if
                end if
            end if
        else if (idx_dir == 3) then
            if (hifu_params%cv_zb <= z_cb(l) .and. hifu_params%cv_zb > z_cb(l - 1)) then
                if ((x_cb(j - 1) - hifu_params%cv_xb) >= 0._wp .and. (hifu_params%cv_xe - x_cb(j) >= 0._wp)) then
                    if ((y_cb(k - 1) - hifu_params%cv_yb) >= 0._wp .and. (hifu_params%cv_ye - y_cb(k) >= 0._wp)) then
                        f_is_on_cv_border = -1
                    end if
                end if
            end if
            if (hifu_params%cv_ze <= z_cb(l) .and. hifu_params%cv_ze > z_cb(l - 1)) then
                if ((x_cb(j - 1) - hifu_params%cv_xb) >= 0._wp .and. (hifu_params%cv_xe - x_cb(j) >= 0._wp)) then
                    if ((y_cb(k - 1) - hifu_params%cv_yb) >= 0._wp .and. (hifu_params%cv_ye - y_cb(k) >= 0._wp)) then
                        f_is_on_cv_border = 1
                    end if
                end if
            end if
            if (bc_z%beg == BC_REFLECTIVE) then
                if (hifu_params%cv_zb == z_cb(l - 1)) then
                    if ((x_cb(j - 1) - hifu_params%cv_xb) >= 0._wp .and. (hifu_params%cv_xe - x_cb(j) >= 0._wp)) then
                        if ((y_cb(k - 1) - hifu_params%cv_yb) >= 0._wp .and. (hifu_params%cv_ye - y_cb(k) >= 0._wp)) then
                            f_is_on_cv_border = -1
                        end if
                    end if
                end if
            end if
        end if

    end function f_is_on_cv_border

    subroutine s_space_derivative(q_var, i, j, k, dumdn, mtd_idx)

        $:GPU_ROUTINE(parallelism='[seq]')
        type(scalar_field), intent(in)      :: q_var
        integer, intent(in)                 :: i, j, k, mtd_idx
        real(wp), dimension(3), intent(out) :: dumdn

        if (mtd_idx == 1) then
            !> First order centered difference approximation
            dumdn(1) = (q_var%sf(i + 1, j, k) - q_var%sf(i - 1, j, k))/(x_cc(i + 1) - x_cc(i - 1))
            dumdn(2) = (q_var%sf(i, j + 1, k) - q_var%sf(i, j - 1, k))/(y_cc(j + 1) - y_cc(j - 1))
            if (p > 0) dumdn(3) = (q_var%sf(i, j, k + 1) - q_var%sf(i, j, k - 1))/(z_cc(k + 1) - z_cc(k - 1))
        else if (mtd_idx == 2) then
            !> Second order centered difference approximation
            dumdn(1) = q_var%sf(i, j, k)*(dx(i + 1) - dx(i - 1)) + q_var%sf(i + 1, j, k)*(dx(i) + dx(i - 1)) - q_var%sf(i - 1, j, &
                  & k)*(dx(i) + dx(i + 1))
            dumdn(1) = dumdn(1)/((dx(i) + dx(i - 1))*(dx(i) + dx(i + 1)))

            dumdn(2) = q_var%sf(i, j, k)*(dy(j + 1) - dy(j - 1)) + q_var%sf(i, j + 1, k)*(dy(j) + dy(j - 1)) - q_var%sf(i, j - 1, &
                  & k)*(dy(j) + dy(j + 1))
            dumdn(2) = dumdn(2)/((dy(j) + dy(j - 1))*(dy(j) + dy(j + 1)))
            if (p > 0) then
                dumdn(3) = q_var%sf(i, j, k)*(dz(k + 1) - dz(k - 1)) + q_var%sf(i, j, k + 1)*(dz(k) + dz(k - 1)) - q_var%sf(i, j, &
                      & k - 1)*(dz(k) + dz(k + 1))
                dumdn(3) = dumdn(3)/((dz(k) + dz(k - 1))*(dz(k) + dz(k + 1)))
            end if
        end if

    end subroutine s_space_derivative

    !> The purpose of this procedure is to write the maximum and minimum pressure through time along the axisymmetric and radial
    !! axes
    !! @param save_count File identifier
    subroutine s_write_Pmax(save_count)

        integer, intent(in) :: save_count
        integer             :: j, k, l
        logical             :: axialCondition, radialCondition, condition
        character(len=512)  :: line

        do l = 0, p
            do k = 0, n
                do j = 0, m
                    ! Specify enough conditions for axial and radial probe lines
                    axialCondition = (dy(k) > abs(y_cc(k)) .and. abs(y_cc(k)) >= 0._wp)
                    if (p > 0) axialCondition = axialCondition .and. (dz(l) > abs(z_cc(l)) .and. abs(z_cc(l)) >= 0._wp)
                    radialCondition = (x_cb(j - 1) < acoustic_bc_params%focLen .and. acoustic_bc_params%focLen < x_cb(j))
                    ! if (p > 0) radialCondition = (z_cb(l - 1) < acoustic_bc_params%focLen .and. acoustic_bc_params%focLen <
                    ! z_cb(l))
                    if (p > 0) radialCondition = radialCondition .and. l == 0
                    condition = (axialCondition .or. radialCondition)
                    if (condition) then
                        if (p > 0) then
                            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + dt, &
                                   & x_cc(j), y_cc(k), z_cc(l), q_hifu%vf(hifu_idx%P)%sf(j, k, l), &
                                   & q_hifu%vf(hifu_idx%P + 1)%sf(j, k, l)
                            write (100, '(A)') trim(line)
                        else
                            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') mytime + dt, x_cc(j), &
                                   & y_cc(k), q_hifu%vf(hifu_idx%P)%sf(j, k, l), q_hifu%vf(hifu_idx%P + 1)%sf(j, k, l)
                            write (100, '(A)') trim(line)
                        end if
                    end if
                end do
            end do
        end do

    end subroutine s_write_Pmax

    subroutine s_initialize_pure_3D(q_cons_vf, bc_type)

        type(scalar_field), dimension(sys_size_hyd), intent(in) :: q_cons_vf
        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        real(wp) :: sum_val_qac, sum_val_qth, sum_val_qvis, sampledTime
        integer :: j, k, l, i
        integer :: qac_hs_idx
        real(wp) :: alpha, rho_cp, tdiff, q_ac, q_vis, q_th, abortFlag, CFL_heat, CFL_heat_max, val_tmp
        character(len=512) :: line

        qac_hs_idx = hifu_idx%qac
        if (hifu_params%intPrms) qac_hs_idx = hifu_idx%qac_prms

        do i = 1, sys_size_hifu
            $:GPU_UPDATE(host='[q_hifu%vf(i)%sf]')
        end do
        sampledTime = q_hifu%vf(hifu_idx%tsamp)%sf(0, 0, 0)

        if (f_approx_equal(sampledTime, 0._wp)) call s_mpi_abort("mbHF: time sampled is zero. Run stage 2.")

        if (bubbles_lagrange) then
            if (proc_rank == 0) print*, 'Adding bubbles in pure 3D domain'
            call s_mean_radius_hifu(sampledTime)
            hifu_idx%qvis = 1; hifu_idx%qth = hifu_idx%qvis + 1
            $:GPU_UPDATE(device='[hifu_params, hifu_idx]')

            $:GPU_PARALLEL_LOOP(private='[i, j, k]', collapse=3)
            do k = idwbuff(3)%beg, idwbuff(3)%end
                do j = idwbuff(2)%beg, idwbuff(2)%end
                    do i = idwbuff(1)%beg, idwbuff(1)%end
                        q_hifu%vf(sys_size_hifu)%sf(i, j, k) = q_hifu%vf(qac_hs_idx)%sf(i, j, k)
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            $:GPU_PARALLEL_LOOP(private='[i, j, k, l]', collapse=4)
            do l = 1, sys_size_hifu - 1
                do k = idwbuff(3)%beg, idwbuff(3)%end
                    do j = idwbuff(2)%beg, idwbuff(2)%end
                        do i = idwbuff(1)%beg, idwbuff(1)%end
                            q_hifu%vf(l)%sf(i, j, k) = 0._wp
                        end do
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            call s_smoothfunction(nBubs, bub_hifu_rad, intfc_vel, mtn_s, mtn_posPrev, q_hifu, bub_qvis, bub_qth)
            if (num_procs > 0) call s_populate_EL_buffers(q_hifu, bc_type, nVar=2)

            if (proc_rank == 0) print*, 'Here are the non-averaged source terms: q_src'
            call s_mpi_barrier()
            call s_print_hifu_source_stats(sys_size_hifu, sum_val_qac)
            call s_print_hifu_source_stats(hifu_idx%qvis, sum_val_qvis)
            call s_print_hifu_source_stats(hifu_idx%qth, sum_val_qth)
            if (proc_rank == 0) then
                write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') sampledTime, &
                       & 0._wp, 0._wp, 0._wp, 0._wp, sum_val_qvis, sum_val_qth
                write (98, '(A)') trim(line)
                close (98)
            end if

            call s_start_HIFU_indexes(stg=3)
            $:GPU_UPDATE(device='[hifu_params, hifu_idx]')

            abortFlag = 0._wp; CFL_heat_max = -100_wp

            $:GPU_PARALLEL_LOOP(private='[i, j, k, q_ac, q_vis, q_th, alpha, rho_cp, tdiff]', collapse=3, &
                                & reduction='[[abortFlag], [CFL_heat_max]]', reductionOp='[+, MAX]', copy='[abortFlag, &
                                & sampledTime, CFL_heat_max]')
            do k = idwbuff(3)%beg, idwbuff(3)%end
                do j = idwbuff(2)%beg, idwbuff(2)%end
                    do i = idwbuff(1)%beg, idwbuff(1)%end
                        q_ac = q_hifu%vf(sys_size_hifu)%sf(i, j, k)/sampledTime
                        q_vis = q_hifu%vf(1)%sf(i, j, k)/sampledTime
                        q_th = q_hifu%vf(2)%sf(i, j, k)/sampledTime

                        q_hifu%vf(hifu_idx%qac)%sf(i, j, k) = q_ac
                        q_hifu%vf(hifu_idx%qvis)%sf(i, j, k) = q_vis
                        q_hifu%vf(hifu_idx%qth)%sf(i, j, k) = q_th
                        q_hifu%vf(hifu_idx%T)%sf(i, j, k) = hifu_params%Tref

                        !> Get thermal properties
                        rho_cp = 0._wp; tdiff = 0._wp
                        $:GPU_LOOP(parallelism='[seq]')
                        do l = 1, num_fluids
                            alpha = q_cons_vf(eqn_idx%adv%beg + l - 1)%sf(i, j, k)
                            rho_cp = rho_cp + alpha*rho_cp_fluids(l)
                            tdiff = tdiff + alpha*tdiff_fluids(l)
                        end do

                        if (f_is_default(rho_cp) .or. f_is_default(tdiff)) abortFlag = abortFlag + 1._wp

                        q_hifu%vf(hifu_idx%T + 1)%sf(i, j, k) = (q_ac + q_vis + q_th)/rho_cp

                        CFL_heat = max(CFL_heat, tdiff*dt/(dx(i)**2.0_wp))
                        CFL_heat = max(CFL_heat, tdiff*dt/(dy(j)**2.0_wp))
                        if (p > 0) CFL_heat = max(CFL_heat, tdiff*dt/(dz(k)**2.0_wp))
                        CFL_heat_max = max(CFL_heat_max, CFL_heat)
                    end do
                end do
            end do
            $:END_GPU_PARALLEL_LOOP()

            if (proc_rank == 0) print*, 'Here are the averaged source terms: q_src*(1/tsampled)*(1/rho*cp)'
            call s_mpi_barrier()
            call s_print_hifu_source_stats(hifu_idx%qac, sum_val_qac)
            call s_print_hifu_source_stats(hifu_idx%qvis, sum_val_qvis)
            call s_print_hifu_source_stats(hifu_idx%qth, sum_val_qth)

            if (num_procs > 1) then
                val_tmp = abortFlag
                call s_mpi_allreduce_max(val_tmp, abortFlag)
                val_tmp = CFL_heat_max
                call s_mpi_allreduce_max(val_tmp, CFL_heat_max)
            end if

            if (proc_rank == 0) print*, 'Max CFL:', CFL_heat_max

            if (abortFlag > 0._wp) call s_mpi_abort("mbHF: not defined thermal properties (rho_cp or tdiff)")
        end if

    end subroutine s_initialize_pure_3D

    subroutine s_print_hifu_source_stats(idx, sum_val)

        integer, intent(in)   :: idx
        real(wp), intent(out) :: sum_val
        real(wp)              :: max_val, min_val, tmp_local, tmp_global, val_test, vol_cell
        integer               :: j, k, l

        max_val = -abs(dflt_real)
        min_val = abs(dflt_real)
        sum_val = 0._wp

        $:GPU_PARALLEL_LOOP(collapse=3, copyin='[idx]', reduction='[[max_val], [min_val], [sum_val]]', reductionOp='[MAX, MIN, &
                            & +]', copy='[max_val, min_val, sum_val]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    val_test = q_hifu%vf(idx)%sf(j, k, l)
                    vol_cell = dx(j)*dy(k)*dz(l)
                    sum_val = sum_val + val_test*vol_cell  ! W
                    max_val = max(max_val, val_test)  ! W/m^3
                    min_val = min(min_val, val_test)  ! W/m^3
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (num_procs > 1) then
            tmp_local = max_val
            call s_mpi_allreduce_max(tmp_local, tmp_global)
            max_val = tmp_global

            tmp_local = min_val
            call s_mpi_allreduce_min(tmp_local, tmp_global)
            min_val = tmp_global

            tmp_local = sum_val
            call s_mpi_allreduce_sum(tmp_local, tmp_global)
            sum_val = tmp_global
        end if

        if (proc_rank == 0) then
            if (idx == hifu_idx%qac) print*, 'q_us (min, max):', min_val, max_val
            if (idx == hifu_idx%qvis) print*, 'q_vis smeared (min, max):', min_val, max_val
            if (idx == hifu_idx%qth) print*, 'q_th smeared (min, max):', min_val, max_val
        end if

    end subroutine s_print_hifu_source_stats

    ! Compute the first, second, and third moments of the heat source from the acoustic field. Heat source calculated from the the
    ! strain rate tensor field.
    subroutine s_write_heat_stats(sampledTime)

        real(wp), intent(in)                 :: sampledTime
        real(wp)                             :: total_heat, heat_moment1, heat_moment2, heat_moment3
        real(wp)                             :: val_tmp, dist_radial
        integer                              :: i, j, k, l
        logical                              :: file_exist
        character(LEN=path_len + 2*name_len) :: file_loc
        character(len=512)                   :: line

        total_heat = 0._wp
        heat_moment1 = 0._wp
        heat_moment2 = 0._wp
        heat_moment3 = 0._wp

        $:GPU_PARALLEL_LOOP(collapse=3, reduction='[[total_heat, heat_moment1, heat_moment2, heat_moment3]]', &
                            & reductionOp='[MAX]', copy='[total_heat, heat_moment1, heat_moment2, heat_moment3]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    ! Filter the cells inside the spherical bubble cloud
                    dist_radial = sqrt((x_cc(j) - hifu_params%cloud_center(1))**2._wp + (y_cc(k) - hifu_params%cloud_center(2)) &
                                       & **2._wp + (z_cc(l) - hifu_params%cloud_center(3))**2._wp)

                    if (dist_radial <= hifu_params%R_cloud) then
                        total_heat = total_heat + q_hifu%vf(hifu_idx%qac)%sf(j, k, l)
                        heat_moment1 = heat_moment1 + q_hifu%vf(hifu_idx%qac)%sf(j, k, &
                                                                & l)*((x_cc(j) - hifu_params%cloud_center(1))/hifu_params%R_cloud)
                        heat_moment2 = heat_moment2 + q_hifu%vf(hifu_idx%qac)%sf(j, k, &
                                                                & l)*((x_cc(j) - hifu_params%cloud_center(1))/hifu_params%R_cloud) &
                                                                & **2._wp
                        heat_moment3 = heat_moment3 + q_hifu%vf(hifu_idx%qac)%sf(j, k, &
                                                                & l)*((x_cc(j) - hifu_params%cloud_center(1))/hifu_params%R_cloud) &
                                                                & **3._wp
                    end if
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (num_procs > 1) then
            val_tmp = total_heat
            call s_mpi_allreduce_sum(val_tmp, total_heat)
            val_tmp = heat_moment1
            call s_mpi_allreduce_sum(val_tmp, heat_moment1)
            val_tmp = heat_moment2
            call s_mpi_allreduce_sum(val_tmp, heat_moment2)
            val_tmp = heat_moment3
            call s_mpi_allreduce_sum(val_tmp, heat_moment3)
        end if

        ! Write the heat statistics to file

        write (file_loc, '(A,I0,A)') 'moments_qac.dat'
        file_loc = trim(case_dir) // '/D/' // trim(file_loc)
        inquire (FILE=trim(file_loc), EXIST=file_exist)

        if (proc_rank == 0) then
            open (11, FILE=trim(file_loc), form='formatted', position='append')
            write (line, '(ES24.16,",",ES24.16,",",ES24.16,",",ES24.16,",",ES24.16)') sampledTime, heat_moment1/total_heat, &
                   & heat_moment2/total_heat, heat_moment3/total_heat, total_heat
            write (11, '(A)') trim(line)
            close (11)
        end if

    end subroutine s_write_heat_stats

    ! Calculate the rhs value from heat transfer eqn discretized with finite volumes.
    subroutine s_rhs_heatEqn(q_cons_vf, rhs_vf, pb, mv, t_step, bc_type, time_avg)

        type(scalar_field), dimension(sys_size_hyd), intent(in) :: q_cons_vf
        type(scalar_field), dimension(sys_size_hyd), intent(inout) :: rhs_vf
        real(wp), optional, dimension(idwbuff(1)%beg:,idwbuff(2)%beg:,idwbuff(3)%beg:,1:,1:), intent(inout) :: pb, mv
        integer, intent(in) :: t_step
        real(wp), intent(inout) :: time_avg
        type(integer_field), dimension(1:num_dims,-1:1), intent(in) :: bc_type
        real(wp) :: dTdx_L, dTdx_R, dTdr_L, dTdr_R
        real(wp) :: dTdx, dTdr, dTdz
        real(wp) :: dTdz_L, dTdz_R, Tz_L, Tz_R
        real(wp) :: tdiff, alpha
        integer :: i, j, k, l
        real(wp) :: abortFlag, val_tmp
        logical :: hifu_on
        real(wp) :: t_start, t_finish

        call nvtxStartRange("COMPUTE-mbHF-RHS")
        call cpu_time(t_start)

        abortFlag = 0._wp
        hifu_on = .false.
        if (t_step < hifu_params%stepStopSource) hifu_on = .true.

        ! 3D cartesian (all stages)
        call s_populate_variables_buffers(bc_type, q_hifu%vf, pb, mv)

        $:GPU_PARALLEL_LOOP(collapse=3, copyin='[hifu_on]', reduction='[[abortFlag]]', reductionOp='[+]', copy='[abortFlag]')
        do l = 0, p
            do k = 0, n
                do j = 0, m
                    rhs_vf(1)%sf(j, k, l) = 0._stp

                    !> Temperature derivatives at the cell center. METHOD: Second order centered difference approximation
                    dTdx = (q_hifu%vf(hifu_idx%T)%sf(j + 1, k, l) - q_hifu%vf(hifu_idx%T)%sf(j - 1, k, &
                            & l))/(x_cc(j + 1) - x_cc(j - 1))
                    dTdx_L = (q_hifu%vf(hifu_idx%T)%sf(j, k, l) - q_hifu%vf(hifu_idx%T)%sf(j - 2, k, l))/(x_cc(j) - x_cc(j - 2))
                    dTdx_R = (q_hifu%vf(hifu_idx%T)%sf(j + 2, k, l) - q_hifu%vf(hifu_idx%T)%sf(j, k, l))/(x_cc(j + 2) - x_cc(j))

                    dTdr = (q_hifu%vf(hifu_idx%T)%sf(j, k + 1, l) - q_hifu%vf(hifu_idx%T)%sf(j, k - 1, &
                            & l))/(y_cc(k + 1) - y_cc(k - 1))
                    dTdr_L = (q_hifu%vf(hifu_idx%T)%sf(j, k, l) - q_hifu%vf(hifu_idx%T)%sf(j, k - 2, l))/(y_cc(k) - y_cc(k - 2))
                    dTdr_R = (q_hifu%vf(hifu_idx%T)%sf(j, k + 2, l) - q_hifu%vf(hifu_idx%T)%sf(j, k, l))/(y_cc(k + 2) - y_cc(k))

                    dTdz = (q_hifu%vf(hifu_idx%T)%sf(j, k, l + 1) - q_hifu%vf(hifu_idx%T)%sf(j, k, &
                            & l - 1))/(z_cc(l + 1) - z_cc(l - 1))
                    dTdz_L = (q_hifu%vf(hifu_idx%T)%sf(j, k, l) - q_hifu%vf(hifu_idx%T)%sf(j, k, l - 2))/(z_cc(l) - z_cc(l - 2))
                    dTdz_R = (q_hifu%vf(hifu_idx%T)%sf(j, k, l + 2) - q_hifu%vf(hifu_idx%T)%sf(j, k, l))/(z_cc(l + 2) - z_cc(l))

                    !> Find temperature derivatives at the faces of the cell
                    dTdx_L = (dTdx*(x_cc(j) - x_cb(j - 1)) + dTdx_L*(x_cb(j - 1) - x_cc(j - 1)))/(x_cc(j) - x_cc(j - 1))
                    dTdr_L = (dTdr*(y_cc(k) - y_cb(k - 1)) + dTdr_L*(y_cb(k - 1) - y_cc(k - 1)))/(y_cc(k) - y_cc(k - 1))
                    dTdz_L = (dTdz*(z_cc(l) - z_cb(l - 1)) + dTdz_L*(z_cb(l - 1) - z_cc(l - 1)))/(z_cc(l) - z_cc(l - 1))

                    dTdx_R = (dTdx*(x_cb(j) - x_cc(j)) + dTdx_R*(x_cc(j + 1) - x_cb(j)))/(x_cc(j + 1) - x_cc(j))
                    dTdr_R = (dTdr*(y_cb(k) - y_cc(k)) + dTdr_R*(y_cc(k + 1) - y_cb(k)))/(y_cc(k + 1) - y_cc(k))
                    dTdz_R = (dTdz*(z_cb(l) - z_cc(l)) + dTdz_R*(z_cc(l + 1) - z_cb(l)))/(z_cc(l + 1) - z_cc(l))

                    !> Get thermal properties
                    tdiff = 0._wp
                    $:GPU_LOOP(parallelism='[seq]')
                    do i = 1, num_fluids
                        alpha = q_cons_vf(eqn_idx%adv%beg + i - 1)%sf(j, k, l)
                        tdiff = tdiff + alpha*tdiff_fluids(i)
                    end do

                    !> Thermal diffusion
                    rhs_vf(1)%sf(j, k, l) = rhs_vf(1)%sf(j, k, &
                           & l) + tdiff*((1._wp/dx(j))*(dTdx_R - dTdx_L) + (1._wp/dy(k))*(dTdr_R - dTdr_L) + (1._wp/dz(l)) &
                           & *(dTdz_R - dTdz_L))

                    !> Adding the heat source terms avg(qac+qvis+qth)/rho_cp
                    if (hifu_on) then
                        rhs_vf(1)%sf(j, k, l) = rhs_vf(1)%sf(j, k, l) + q_hifu%vf(hifu_idx%T + 1)%sf(j, k, l)
                    end if

                    ! Checking NaNs
                    if (rhs_vf(1)%sf(j, k, l) /= rhs_vf(1)%sf(j, k, l)) abortFlag = abortFlag + 1._wp
                end do
            end do
        end do
        $:END_GPU_PARALLEL_LOOP()

        if (num_procs > 1) then
            val_tmp = abortFlag
            call s_mpi_allreduce_max(val_tmp, abortFlag)
        end if

        if (abortFlag > 0._wp) call s_mpi_abort("Errors found in s_rhs_heatEqn")

        if (proc_rank == 0 .and. t_step == hifu_params%stepStopSource) print *, 'WARNING :: Turn off HIFU source'

        call cpu_time(t_finish)

        if (t_step >= 2) then
            time_avg = (abs(t_finish - t_start) + (t_step - 2)*time_avg)/(t_step - 1)
        else
            time_avg = 0._wp
        end if

        call nvtxEndRange

    end subroutine s_rhs_heatEqn

    subroutine s_open_run_time_information_samplingHIFU()

        character(LEN=path_len + 3*name_len) :: file_path

        ! Open files to save Pmax data at the axial and radial axes

        write (file_path, '(A,I0,A)') '/D/Pmax_', proc_rank, '.dat'
        file_path = trim(case_dir) // trim(file_path)
        open (100, FILE=trim(file_path), form='formatted', STATUS='unknown')
        if (p > 0) then
            write (100, '(A)') 'mytime,x_cc,y_cc,z_cc,Pmax,Pmin'
        else
            write (100, '(A)') 'mytime,x_cc,y_cc,Pmax,Pmin'
        end if

        if (proc_rank == 0) then
            ! Open files to save intensity sampling information at focus
            write (file_path, '(A)') '/D/hifu_qac.dat'
            file_path = trim(case_dir) // trim(file_path)
            open (99, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
            write (99, '(A)') 'mytime,sampling,sum_qac,sum_qac_prms'

            ! Open files to save viscous and thermal intensity sampling information for a single bubble
            write (file_path, '(A,I0,A)') '/D/hifu_qbubs.dat'
            file_path = trim(case_dir) // trim(file_path)
            open (98, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
            write (98, '(A)') 'mytime,dt,sum_qvis,sum_qth,nbubs,sum_smear_qvis,sum_smear_qth'

            if (hifu_params%moments) then
                ! Open files to save heat sources and volume moments
                write (file_path, '(A,I0,A)') '/D/moments_qac.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (97, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (97, '(A)') 'mytime,normMom_f,normMom_s,normMom_t,totalHeat'

                write (file_path, '(A,I0,A)') '/D/moments_qvis.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (96, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (96, '(A)') 'mytime,normMom_f,normMom_s,normMom_t,totalHeat'

                write (file_path, '(A,I0,A)') '/D/moments_qth_pos.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (95, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (95, '(A)') 'mytime,normMom_f,normMom_s,normMom_t,totalHeat'

                write (file_path, '(A,I0,A)') '/D/moments_qth_neg.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (94, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (94, '(A)') 'mytime,normMom_f,normMom_s,normMom_t,totalHeat'

                write (file_path, '(A,I0,A)') '/D/moments_vol.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (93, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (93, '(A)') 'mytime,normMom_f,normMom_s,normMom_t,totalVolume'

                write (file_path, '(A,I0,A)') '/D/moments_ke.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (92, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (92, '(A)') 'mytime,normMom_f,normMom_s,normMom_t,totalKE'
            end if

            if (hifu_params%power_balance) then
                write (file_path, '(A,I0,A)') '/D/power_balance_in.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (89, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (89, '(A)') 'mytime,hdid,acPw_in_xb,acPw_in_xe,acPw_in_yb,acPw_in_ye,acPw_in_zb,acPw_in_ze'

                write (file_path, '(A,I0,A)') '/D/power_balance_out.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (88, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (88, '(A)') 'mytime,hdid,acPw_out_xb,acPw_out_xe,acPw_out_yb,acPw_out_ye,acPw_out_zb,acPw_out_ze'

                write (file_path, '(A,I0,A)') '/D/power_balance_qac.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (87, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (87, '(A)') 'mytime,hdid,qac,acPw_cmprssv,acPw_kntc'

                write (file_path, '(A,I0,A)') '/D/power_balance_qbub.dat'
                file_path = trim(case_dir) // trim(file_path)
                open (86, FILE=trim(file_path), form='formatted', POSITION='append', STATUS='replace')
                write (86, '(A)') 'mytime,hdid,nbubs,qvis,qth,ke'
            end if
        end if

    end subroutine s_open_run_time_information_samplingHIFU

    subroutine s_close_run_time_information_samplingHIFU()

        ! Close files to save Pmax data at the axial and radial axes
        close (100)

        if (proc_rank == 0) then
            ! Close file to save intensity sampling information at focus
            close (99)

            ! Close file to save viscous and thermal intensity sampling information for a single bubble close (98)

            ! Close file to save heat sources and volume moments
            close (97)
            close (96)
            close (95)
            close (94)
            close (93)
            close (92)

            close (89)
            close (88)
            close (87)
            close (86)
        end if

    end subroutine s_close_run_time_information_samplingHIFU

    subroutine s_finalize_HIFU_module()

        integer :: i

        do i = 1, sys_size_hifu
            @:DEALLOCATE(q_hifu%vf(i)%sf)
        end do
        @:DEALLOCATE(q_hifu%vf)

        @:DEALLOCATE(shear_viscous_fluids)
        @:DEALLOCATE(bulk_viscous_fluids)
        @:DEALLOCATE(abs_coef_fluids)
        @:DEALLOCATE(rho_cp_fluids)
        @:DEALLOCATE(tdiff_fluids)

        if (bc_x%beg == -6) bc_x%beg = -20
        $:GPU_UPDATE(device='[bc_x]')

    end subroutine s_finalize_HIFU_module

end module m_hifu
