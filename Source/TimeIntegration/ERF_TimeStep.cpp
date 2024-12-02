#include <ERF.H>
#include <Utils.H>

using namespace amrex;

/**
 * Function that coordinates the evolution across levels -- this calls Advance to do the
 * actual advance at this level,  then recursively calls itself at finer levels
 *
 * @param[in] lev level of refinement (coarsest level is 0)
 * @param[in] time start time for time advance
 * @param[in] iteration time step counter
 */

void
ERF::timeStep (int lev, Real time, int /*iteration*/)
{

amrex::Print() << "I AM AT THE BEGINNING OF timeSTEP " << std::endl;
amrex::Print() << "At the beginning of TimeStep, Processor " << amrex::ParallelDescriptor::MyProc() << " out of " << amrex::ParallelDescriptor::NProcs()
               << " checking ERF_USE_WW3_COUPLING." << std::endl;
if (regrid_int > 0)  // We may need to regrid
    {
        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid
        if (lev < max_level)
        {
            if ( (istep[lev] % regrid_int == 0) && (istep[lev] > last_regrid_step[lev]) )
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;

                regrid(lev, time);

#ifdef ERF_USE_PARTICLES
                if (finest_level != old_finest) {
                    particleData.Redistribute();
                }
#endif

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }
            } // if
        } // lev
    }

    // Update what we call "old" and "new" time
    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "ADVANCE from time = " << t_old[lev] << " to " << t_new[lev]
                       << " with dt = " << dt[lev] << std::endl;
    }

amrex::Print() << "Processor " << amrex::ParallelDescriptor::MyProc()
               << " is checking ERF_USE_WW3_COUPLING." << std::endl;

#ifdef ERF_USE_WW3_COUPLING
amrex::Print() << "ERF_USE_WW3_COUPLING is defined." << std::endl;
#else
amrex::Print() << "ERF_USE_WW3_COUPLING is NOT defined." << std::endl;
#endif


#ifdef ERF_USE_WW3_COUPLING
    int global_rank = amrex::ParallelDescriptor::MyProc();

    amrex::Print() <<  " I AM HERE " << std::endl;

    amrex::Print() << "Total number of MPI ranks (ERF time_step): " << amrex::ParallelDescriptor::NProcs() << std::endl;
    amrex::Print() << "Processor " << global_rank << " is about to call send_to_ww3." << std::endl;
    amrex::Print() << "Before calling send_to_ww3 the global mpi rank is:" << global_rank <<std::endl;
    

    amrex::Print() <<  " About to call send_to_ww3 from ERF_Timestep" << std::endl;
    send_to_ww3(lev);

    amrex::Print() << "After calling send_waves the global mpi rank is:" << global_rank <<std::endl;
    amrex::Print() <<  " About to call read_waves from ERF_Timestep"  << std::endl;
    read_waves(lev);
#endif

    // Advance a single level for a single time step
    Advance(lev, time, dt[lev], istep[lev], nsubsteps[lev]);

    ++istep[lev];

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            Real strt_time_for_fine = time + (i-1)*dt[lev+1];
            timeStep(lev+1, strt_time_for_fine, i);
        }
    }

    if (verbose && lev == 0) {
        if (solverChoice.moisture_type != MoistureType::None) {
            amrex::Print() << "Cloud fraction " << time << "  " << cloud_fraction(time) << std::endl;
        }
    }
}

/**
 * Function that advances the solution at one level for a single time step --
 * this does some preliminaries then calls erf_advance
 *
 * @param[in] lev level of refinement (coarsest level is 0)
 * @param[in] time start time for time advance
 * @param[in] dt_lev time step for this time advance
 */
