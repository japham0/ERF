#ifdef ERF_USE_WW3_COUPLING

#include <ERF.H>
#include <Utils.H>
#include <mpi.h>
#include <AMReX_MPMD.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <cmath>
#include <time.h>
#include <iostream>

using namespace amrex;


void
ERF::read_waves (int lev)
{
    for ( MFIter mfi(*Hwave_onegrid[lev],false); mfi.isValid(); ++mfi)
    {

         const auto & bx = mfi.validbox();

         //amrex::Print() <<  " Just called ERF::read_waves to receive from WW3 " << bx << std::endl;
         amrex::Array4<Real> my_H_arr = Hwave_onegrid[lev]->array(mfi);
         amrex::Array4<Real> my_L_arr = Lwave_onegrid[lev]->array(mfi);
        amrex::Print() << "my Valid box: (orig) ("
               << bx.smallEnd(0) << ", " << bx.smallEnd(1) << ") to ("
               << bx.bigEnd(0) << ", " << bx.bigEnd(1) << ")" << std::endl;
         Real* my_H_ptr = my_H_arr.dataPtr();
         Real* my_L_ptr = my_L_arr.dataPtr();
         

         int rank_offset = amrex::MPMD::MyProc() - amrex::ParallelDescriptor::MyProc();
         int this_root, other_root;
         if (rank_offset == 0) { // First program
             this_root = 0;
             other_root = amrex::ParallelDescriptor::NProcs();
         } else {
             this_root = rank_offset;
             other_root = 0;
         }


         int nx=2147483647;
         int ny=2147483647; // sanity check

         //JUST RECEIVED
         if (amrex::MPMD::MyProc() == this_root) {
             if (rank_offset == 0) // First program
             {
                 MPI_Recv(&nx, 1, MPI_INT, other_root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 MPI_Recv(&ny, 1, MPI_INT, other_root, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             }
             else // Second program
             {
                 MPI_Recv(&nx, 1, MPI_INT, other_root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 MPI_Recv(&ny, 1, MPI_INT, other_root, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             }
             //This may not be necessary
             //ParallelDescriptor::Bcast(&nx, 1);
             //ParallelDescriptor::Bcast(&ny, 1);
         }

         if((nx)*(ny) > 0) {
             int nsealm = (nx)*ny;

             if (amrex::MPMD::MyProc() == this_root) {
                 if (rank_offset == 0) // the first program
                 {
                     MPI_Recv(my_H_ptr, nsealm, MPI_DOUBLE, other_root, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     MPI_Recv(my_L_ptr, nsealm, MPI_DOUBLE, other_root, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 }
                 else // the second program
                 {
                     MPI_Recv(my_H_ptr, nsealm, MPI_DOUBLE, other_root, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     MPI_Recv(my_L_ptr, nsealm, MPI_DOUBLE, other_root, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     amrex::PrintToFile("output_HS_cpp.txt")<<FArrayBox(my_H_arr)<<std::endl;
                    amrex::PrintToFile("output_L_cpp.txt")<<FArrayBox(my_L_arr)<<std::endl;
                    //amrex::Print() << "NSEALM: " << nsealm << std::endl;
                }
             }

         }
    }



// START
    int ioproc = ParallelDescriptor::IOProcessorNumber();

    const Box& bx = Hwave_onegrid[lev]->boxArray()[0];  // Single box in Hwave_onegrid[lev]
    int nx = bx.length(0);
    int ny = bx.length(1);
    std::vector<amrex::Real> temp_bufferH(nx * ny, 1.0);
    std::vector<amrex::Real> temp_bufferL(nx * ny, 1.0);
    amrex::AllPrint() << " NX ,NY: " << nx << " " << ny << std::endl; 
    //amrex::AllPrint() << "temp_bufferH size:" << temp_bufferH.size() << " from rank " << amrex::ParallelDescriptor::MyProc() << std::endl; 
    //amrex::AllPrint() << " temp_buffer size: " << temp_buffer.size() << " my rank is " << amrex::MPMD::MyProc()<<std::endl; 
   

    for (MFIter mfi(*Hwave_onegrid[lev]); mfi.isValid(); ++mfi) {
        const Array4<Real const>& arr_onegridH = Hwave_onegrid[lev]->const_array(mfi);
        int index = 0;
        const Box& bx = mfi.validbox();
        amrex::Print() << "my Valid box: ("
               << bx.smallEnd(0) << ", " << bx.smallEnd(1) << ") to ("
               << bx.bigEnd(0) << ", " << bx.bigEnd(1) << ")" << std::endl;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                    amrex::AllPrintToFile("debug_index.txt")<< index << " i, j: "<< i << ", " << j << " j "<< arr_onegridH(i, j, 0)<<std::endl;
                    temp_bufferH[index++] = arr_onegridH(i, j, 0);
            }
        }
    }
    for (MFIter mfi(*Lwave_onegrid[lev]); mfi.isValid(); ++mfi) {
        const Array4<Real const>& arr_onegridL = Lwave_onegrid[lev]->const_array(mfi);
        int index = 0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                    amrex::AllPrintToFile("debug_indexL.txt")<< index << " i, j: "<< i << ", " << j << " j "<< arr_onegridL(i, j, 0)<<std::endl;
                    temp_bufferL[index++] = arr_onegridL(i, j, 0);
            }
        }
    }

    ParallelDescriptor::Barrier();

    amrex::ParallelDescriptor::Bcast(temp_bufferH.data(), temp_bufferH.size(), 0);

    ParallelDescriptor::Barrier();
    amrex::ParallelDescriptor::Bcast(temp_bufferL.data(), temp_bufferL.size(), 0);
    
    for (int i = 0; i < 25; ++i){
        amrex::Print() << i << " temp_buffer from rank " << amrex::ParallelDescriptor::MyProc() << " (after) " << temp_bufferH[i] << std::endl;
    }
    

    ParallelDescriptor::Barrier();
    amrex::Print() << " temp_bufferH.size: " << temp_bufferH.size() << std::endl;

for (MFIter mfi(*Hwave[lev]); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.validbox();
    const Array4<Real>& arr_hwave = Hwave[lev]->array(mfi);

    const Geometry& geom_at_level = geom[lev];
    const Box& domain_box = geom_at_level.Domain();
    int NX = domain_box.length(0);
    int NY = domain_box.length(1);
    amrex::AllPrint() << " BIG NX ,NY: " << nx << " " << ny <<" my rank is " << amrex::ParallelDescriptor::MyProc()<< std::endl;
    
    int nx = bx.length(0);
    int ny = bx.length(1);
    amrex::AllPrint() << " small nx ,ny: " << nx << " " << ny <<" my rank is " << amrex::ParallelDescriptor::MyProc()<< std::endl;
    amrex::AllPrint() << " arr_hwave size: " << arr_hwave.size() << " my rank is " << amrex::ParallelDescriptor::MyProc()<< std::endl;
    amrex::AllPrint() << "my Valid box: (" 
               << bx.smallEnd(0) << ", " << bx.smallEnd(1) << ") to ("
               << bx.bigEnd(0) << ", " << bx.bigEnd(1) << ")" <<  " my rank is " << amrex::ParallelDescriptor::MyProc() << std::endl;
    

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // Calculate the global index for the temp_buffer
        
        int index = i + j * NX;
        Real valueH = temp_bufferH[index];
        IntVect iv(i, j, k);

        amrex::AllPrintToFile("check_temp.txt") << "Proc " << amrex::ParallelDescriptor::MyProc()
                      << ", IntVect: " << iv 
                      << ", index: " << index 
                      << ", value: " << temp_bufferH[index] << std::endl;
            
        // Assign value from temp_buffer to the corresponding element in Hwave[lev]
        arr_hwave(i, j, 0) = valueH;

        amrex::AllPrintToFile("arrH.txt") << "Proc " << amrex::ParallelDescriptor::MyProc()
                      << ", IntVect: " << iv
                      << ", value: " << valueH << std::endl;
    });
}
    Hwave[lev]->FillBoundary(geom[lev].periodicity());

for (MFIter mfi(*Lwave[lev]); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.validbox();
    const Array4<Real>& arr_lwave = Lwave[lev]->array(mfi);
    
    const Geometry& geom_at_level = geom[lev];
    const Box& domain_box = geom_at_level.Domain();
    int NX = domain_box.length(0);
    int NY = domain_box.length(1);

    int nx = bx.length(0);
    int ny = bx.length(1);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        
        int index = i + j * NX;
        Real valueL = temp_bufferL[index];
        arr_lwave(i, j, 0) = valueL;
        IntVect iv(i,j,k);

        amrex::AllPrintToFile("arrL.txt") << "Proc " << amrex::ParallelDescriptor::MyProc()
                      << ", IntVect: " << iv
                      << ", value: " << valueL << std::endl;

    });
}

    Lwave[lev]->FillBoundary(geom[lev].periodicity());
    ParallelDescriptor::Barrier();

    amrex::AllPrint() << "MADE IT PAST THE BARRIER" << std::endl;
// END

   // Hwave[lev]->ParallelCopy(*Hwave_onegrid[lev]);
   // Lwave[lev]->ParallelCopy(*Lwave_onegrid[lev]);

    //Lwave[lev]->FillBoundary(geom[lev].periodicity());
    //Hwave[lev]->FillBoundary(geom[lev].periodicity());
  //  amrex::Print() << "HWAVE BOX " << (*Hwave[lev])[0].box() << std::endl;
    
    for (MFIter mfi(*Hwave[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box bx = mfi.tilebox();


        const Array4<Real const>& Hwave_arr = Hwave[lev]->const_array(mfi);
        const Array4<int>& Lmask_arr = lmask_lev[lev][0]->array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            IntVect iv(i,j,k);
             //   amrex::Print() << "Hwave_arr (i,j) " << i << ", " << j << "): " << Hwave_arr(i,j,k) << std::endl;   

            if (Hwave_arr(i,j,k)<0) {
                Lmask_arr(i,j,k) = 1;
             } else {
                Lmask_arr(i,j,k) = 0;
            }
            
                int value = Lmask_arr(i,j,k);
                amrex::AllPrintToFile("landmask.txt") << "Proc " << amrex::ParallelDescriptor::MyProc()
                               << ", Index: " << iv << ", Lmask_arr = "
                               << value << std::endl;
                               
        });
    }
    

}

void
ERF::send_to_ww3 (int lev)
{
    int ncomp = 1; // number components
    auto& lev_new = vars_new[lev];
    const double PI = 3.1415926535897932384626433832795028841971693993751058209;

    int count_send = 0;
    int k_ref = 0;
    int nlevs_max = max_level + 1;
    const auto dz  = geom[lev].CellSize(2); //For 10m
    if (dz < 10){
        k_ref = std::floor( (10 / dz) - 0.5 );
    }

    // Access xvel, yvel from ABL
    MultiFab xvel_data(lev_new[Vars::xvel].boxArray(), lev_new[Vars::xvel].DistributionMap(), 1, lev_new[Vars::xvel].nGrowVect());

    MultiFab yvel_data(lev_new[Vars::yvel].boxArray(), lev_new[Vars::yvel].DistributionMap(), 1, lev_new[Vars::yvel].nGrowVect());


    // Make local copy of xvel, yvel
    MultiFab::Copy (xvel_data, lev_new[Vars::xvel], 0, 0, 1, lev_new[Vars::xvel].nGrowVect());
    MultiFab::Copy (yvel_data, lev_new[Vars::yvel], 0, 0, 1, lev_new[Vars::yvel].nGrowVect());


    // Initialize Multifabs to store x_avg, y_avg, u_mag, u_dir at cell centers
    MultiFab x_avg(lev_new[Vars::cons].boxArray(), lev_new[Vars::cons].DistributionMap(), 1, lev_new[Vars::cons].nGrowVect());

    MultiFab y_avg(lev_new[Vars::cons].boxArray(), lev_new[Vars::cons].DistributionMap(), 1, lev_new[Vars::cons].nGrowVect());

    MultiFab u_mag(lev_new[Vars::cons].boxArray(), lev_new[Vars::cons].DistributionMap(), 1, lev_new[Vars::cons].nGrowVect());


    MultiFab u_dir(lev_new[Vars::cons].boxArray(), lev_new[Vars::cons].DistributionMap(), 1, lev_new[Vars::cons].nGrowVect());


    x_avg.setVal(0.);
    y_avg.setVal(0.);
    u_mag.setVal(0.);
    u_dir.setVal(0.);

    for (MFIter mfi(x_avg,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        Box bx = mfi.tilebox();
        const Array4<Real>& u_vel = x_avg.array(mfi);
        const Array4<const Real>& velx_arr = xvel_data.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){
            u_vel(i,j,k)  = 0.5 *( velx_arr(i,j,k) + velx_arr(i+1,j,k) );

            amrex::AllPrintToFile("uvel.txt") << amrex::IntVect(i,j,k) << " [" <<velx_arr(i,j,k) << "| avg:  " << u_vel(i,j,k)<< " | " << velx_arr(i+1,j,k) << "]" <<std::endl;
        });
    }

    for (MFIter mfi(y_avg,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        Box bx = mfi.tilebox();
        const Array4<Real>& v_vel = y_avg.array(mfi);
        const Array4<const Real>& vely_arr = yvel_data.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            v_vel(i,j,k)  = 0.5 *( vely_arr(i,j,k) + vely_arr(i,j+1,k) );

            amrex::AllPrintToFile("vvel.txt") << "%ld" << amrex::IntVect(i,j,k) << " ["<<vely_arr(i,j,k)<<"| avg: " << v_vel(i,j,k)<< " | " << vely_arr(i,j+1,k) << "]" <<std::endl;
        });
    }


// START

    int num_procs = amrex::MPMD::NProcs();  // Total number of processors
    int num_procs_local = amrex::ParallelDescriptor::NProcs();
    int rank = amrex::MPMD::MyProc();       // Current processor rank

    int rank_offset = amrex::MPMD::MyProc() - amrex::ParallelDescriptor::MyProc();         
    int this_root, other_root;
         if (rank_offset == 0) { // First program
             this_root = 0;
             other_root = amrex::ParallelDescriptor::NProcs();
         } else {
             this_root = rank_offset;
             other_root = 0;
         }

    std::vector<amrex::Real> k_ref_vals;
    std::vector<amrex::Real> theta_values;
    std::vector<amrex::Real> magnitude_values;

    for (MFIter mfi(u_mag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        Box bx = mfi.tilebox();
        const Array4<Real>& magnitude = u_mag.array(mfi);
        const Array4<const Real>& u = x_avg.array(mfi);
        const Array4<const Real>& v = y_avg.array(mfi);
        const Array4<Real>& theta = u_dir.array(mfi);

        amrex::Vector<std::unique_ptr<amrex::MultiFab>> magnitude_onegrid;
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> theta_onegrid;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            double u_val = u(i, j, k);
            double v_val = v(i, j, k);

            if ( u_val == 0 ) {
                u_val = std::max( u_val, 1e-15 );  // Ensure u_val is non-zero
            }

            magnitude(i,j,k) = std::sqrt( pow(u_val, 2) + pow(v_val, 2) );

            if ( u_val < 0 && v_val > 0 || u_val < 0 && v_val < 0 ) {

                theta(i,j,k) = PI + ( atan( v_val / u_val ) );

            } else {

                theta(i,j,k) = atan ( v_val / u_val );
            }


            amrex::AllPrintToFile("mag_theta.txt") << amrex::IntVect(i,j,k) <<  " Magnitude: " << magnitude(i,j,k) << " Theta: " << theta(i,j,k) <<std::endl;
            });

            // Fill k_ref_vals
            for (amrex::BoxIterator bi(bx); bi.ok(); ++bi){
                amrex::IntVect iv = bi();
                if (iv[2] == k_ref){
                    //amrex::AllPrint() << iv << ": " << magnitude(iv) << " rank: " << rank << std::endl;
                    magnitude_values.push_back(magnitude(iv));
                    theta_values.push_back(theta(iv));
                }
            }
            //amrex::AllPrint() << "Size of k_ref_vals: " << magnitude_values.size() << "rank: " << rank << std::endl;
}

            int local_size = magnitude_values.size();
            int total_size = 0;
            amrex::Vector<amrex::Real> gathered_mags;
            amrex::Vector<amrex::Real> gathered_thetas;

            amrex::Vector<int> local_sizes(num_procs_local);
            amrex::ParallelDescriptor::Gather(&local_size, 1, local_sizes.data(), 1, 0);

            //amrex::AllPrint() << "My Local Size is " << local_size << " rank is " << rank << " local_sizes vector len" << local_sizes.size()<<  std::endl;    

            if (amrex::ParallelDescriptor::MyProc() == 0) {
                /*
                amrex::Print()<<"Local sizes after gather: ";
                for (int i = 0; i < num_procs_local; ++i){
                    amrex::Print() << local_sizes[i] << " ";
                }
                amrex::Print()<<std::endl;
                */
                for (int i = 0; i < num_procs_local; ++i){
                    total_size += local_sizes[i];
                }
                
                gathered_mags.resize(total_size);
                gathered_thetas.resize(total_size);
            }

            amrex::ParallelDescriptor::Bcast(&total_size, 1, 0);
            
            amrex::Vector<int> displacements(num_procs_local,0);
            if (amrex::ParallelDescriptor::MyProc() == 0){
                int displacement = 0;
                for (int i = 0; i < num_procs_local; ++i){
                    displacements[i] = displacement;
                    displacement += local_sizes[i];
                }
            }
            /*
            amrex::Print() << "DISPLACEMENT: ";
            for (int i = 0; i < displacements.size(); ++i){
                amrex::Print() << displacements[i] << " " << std::endl;
            }

            amrex::Print() << "DISPLACEMENTS SIZE: " << displacements.size();
            */

            amrex::ParallelDescriptor::Gatherv(magnitude_values.data(), local_size, gathered_mags.data(), local_sizes, displacements, 0);
            amrex::ParallelDescriptor::Gatherv(theta_values.data(), local_size, gathered_thetas.data(), local_sizes, displacements, 0);
           /* 
            amrex::Print() << "Gathered magnitude values: ";
            for (int i = 0; i < gathered_mags.size(); ++i){
                amrex::Print() << gathered_mags[i] << " ";
            }
            amrex::Print() << " My rank is " << rank << " and there are " << gathered_mags.size() << " elements." << std::endl;

            amrex::Print() << "Gathered theta values: ";
            for (int i = 0; i < gathered_thetas.size(); ++i){
                amrex::Print() << gathered_thetas[i] << " ";
            }
            amrex::Print() << " My rank is " << rank << " and there are " << gathered_mags.size() << " elements." << std::endl;
        */

        if (amrex::ParallelDescriptor::MyProc()==0){
        for (int j = 0; j < total_size; ++j) {
    amrex::AllPrintToFile("debug_send.txt")
        << "dz, k_ref " << dz << ", " << k_ref << " "
        //<< "Index: " << j
        //<< ", IntVect: (" << indices[j][0] << ", " << indices[j][1] << ", " << indices[j][2] << ")"
        << ", Magnitude: " << gathered_mags[j]
        << ", Theta: " << gathered_thetas[j]
        << std::endl;
}

        }
        

             if (amrex::MPMD::MyProc() == this_root) {
                if (rank_offset == 0) // First program
                {
                    MPI_Send(&total_size, 1, MPI_INT, other_root, 11, MPI_COMM_WORLD);
                    MPI_Send(magnitude_values.data(), total_size, MPI_DOUBLE, other_root, 13, MPI_COMM_WORLD);
                    MPI_Send(theta_values.data(), total_size, MPI_DOUBLE, other_root, 15, MPI_COMM_WORLD);
             }
             else // Second program
             {
                    MPI_Send(&total_size, 1, MPI_INT, other_root, 10, MPI_COMM_WORLD);
                    MPI_Send(magnitude_values.data(), total_size, MPI_DOUBLE, other_root, 12, MPI_COMM_WORLD);
                    MPI_Send(theta_values.data(), total_size, MPI_DOUBLE, other_root, 14, MPI_COMM_WORLD);
            }
         }

}
#endif

