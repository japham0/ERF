#include <iomanip>

#include "ERF.H"
#include "EOS.H"

using namespace amrex;

void
ERF::write_1D_profiles(Real time)
{
    BL_PROFILE("ERF::write_1D_profiles()");

    if (verbose <= 0)
      return;

    int datwidth = 14;
    int datprecision = 6;
    int timeprecision = 13; // e.g., 1-yr LES: 31,536,000 s with dt ~ 0.01 ==> min prec = 10

    if (verbose > 0 && NumDataLogs() > 1)
    {
        // Define the 1d arrays we will need
        Gpu::HostVector<Real> h_avg_u, h_avg_v, h_avg_w, h_avg_th;
        Gpu::HostVector<Real> h_avg_uth, h_avg_vth, h_avg_wth, h_avg_thth;
        Gpu::HostVector<Real> h_avg_uu, h_avg_uv, h_avg_uw, h_avg_vv, h_avg_vw, h_avg_ww;
        Gpu::HostVector<Real> h_avg_tau11, h_avg_tau12, h_avg_tau13, h_avg_tau22, h_avg_tau23, h_avg_tau33;
        Gpu::HostVector<Real> h_avg_k, h_avg_ku, h_avg_kv, h_avg_kw;
        Gpu::HostVector<Real> h_avg_p, h_avg_pu, h_avg_pv, h_avg_pw;

        if (NumDataLogs() > 1) {
            derive_diag_profiles(h_avg_u, h_avg_v, h_avg_w, h_avg_th,
                                 h_avg_uu, h_avg_uv, h_avg_uw, h_avg_vv, h_avg_vw, h_avg_ww,
                                 h_avg_uth, h_avg_vth, h_avg_wth, h_avg_thth,
                                 h_avg_k, h_avg_ku, h_avg_kv, h_avg_kw,
                                 h_avg_p, h_avg_pu, h_avg_pv, h_avg_pw);
        }

        if (NumDataLogs() > 3) {
            derive_stress_profiles(h_avg_tau11, h_avg_tau12, h_avg_tau13,
                                   h_avg_tau22, h_avg_tau23, h_avg_tau33);
        }

        auto const& dx = geom[0].CellSizeArray();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (NumDataLogs() > 1) {
                std::ostream& data_log1 = DataLog(1);
                if (data_log1.good()) {
                  // Write the quantities at this time
                  for (int k = 0; k < h_avg_u.size(); k++) {
                      Real z = (k + 0.5)* dx[2];
                      data_log1 << std::setw(datwidth) << std::setprecision(timeprecision) << time << " "
                                << std::setw(datwidth) << std::setprecision(datprecision) << z << " "
                                << h_avg_u[k]  << " " << h_avg_v[k] << " " << h_avg_w[k] << " " << h_avg_th[k]
                                << std::endl;
                  } // loop over z
                } // if good
            } // NumDataLogs

            if (NumDataLogs() > 2) {
                std::ostream& data_log2 = DataLog(2);
                if (data_log2.good()) {
                  // Write the perturbational quantities at this time
                  for (int k = 0; k < h_avg_u.size(); k++) {
                      Real z = (k + 0.5)* dx[2];
                      data_log2 << std::setw(datwidth) << std::setprecision(timeprecision) << time << " "
                                << std::setw(datwidth) << std::setprecision(datprecision) << z << " "
                                << h_avg_uu[k]   - h_avg_u[k]*h_avg_u[k]  << " "
                                << h_avg_uv[k]   - h_avg_u[k]*h_avg_v[k]  << " "
                                << h_avg_uw[k]   - h_avg_u[k]*h_avg_w[k]  << " "
                                << h_avg_vv[k]   - h_avg_v[k]*h_avg_v[k]  << " "
                                << h_avg_vw[k]   - h_avg_v[k]*h_avg_w[k]  << " "
                                << h_avg_ww[k]   - h_avg_w[k]*h_avg_w[k]  << " "
                                << h_avg_uth[k]  - h_avg_u[k]*h_avg_th[k] << " "
                                << h_avg_vth[k]  - h_avg_v[k]*h_avg_th[k] << " "
                                << h_avg_wth[k]  - h_avg_w[k]*h_avg_th[k] << " "
                                << h_avg_thth[k] - h_avg_th[k]*h_avg_th[k] << " "
                                << h_avg_ku[k]   - h_avg_k[k]*h_avg_u[k] << " "
                                << h_avg_kv[k]   - h_avg_k[k]*h_avg_v[k] << " "
                                << h_avg_kw[k]   - h_avg_k[k]*h_avg_w[k] << " "
                                << h_avg_pu[k]   - h_avg_p[k]*h_avg_u[k] << " "
                                << h_avg_pv[k]   - h_avg_p[k]*h_avg_v[k] << " "
                                << h_avg_pw[k]   - h_avg_p[k]*h_avg_w[k]
                                << std::endl;
                  } // loop over z
                } // if good
            } // NumDataLogs

            if (NumDataLogs() > 3) {
                std::ostream& data_log3 = DataLog(3);
                if (data_log3.good()) {
                  // Write the average stresses
                  for (int k = 0; k < h_avg_u.size(); k++) {
                      Real z = (k + 0.5)* dx[2];
                      data_log3 << std::setw(datwidth) << std::setprecision(timeprecision) << time << " "
                                << std::setw(datwidth) << std::setprecision(datprecision) << z << " "
                                << h_avg_tau11[k] << " " << h_avg_tau12[k] << " " << h_avg_tau13[k] << " "
                                << h_avg_tau22[k] << " " << h_avg_tau23[k] << " " << h_avg_tau33[k]
                                << std::endl;
                  } // loop over z
                } // if good
            } // NumDataLogs
        } // if IOProcessor
    } // if verbose
}

void
ERF::derive_diag_profiles(Gpu::HostVector<Real>& h_avg_u   , Gpu::HostVector<Real>& h_avg_v  ,
                          Gpu::HostVector<Real>& h_avg_w   , Gpu::HostVector<Real>& h_avg_th ,
                          Gpu::HostVector<Real>& h_avg_uu  , Gpu::HostVector<Real>& h_avg_uv , Gpu::HostVector<Real>& h_avg_uw,
                          Gpu::HostVector<Real>& h_avg_vv  , Gpu::HostVector<Real>& h_avg_vw , Gpu::HostVector<Real>& h_avg_ww,
                          Gpu::HostVector<Real>& h_avg_uth , Gpu::HostVector<Real>& h_avg_vth, Gpu::HostVector<Real>& h_avg_wth,
                          Gpu::HostVector<Real>& h_avg_thth,
                          Gpu::HostVector<Real>& h_avg_k,
                          Gpu::HostVector<Real>& h_avg_ku  , Gpu::HostVector<Real>& h_avg_kv , Gpu::HostVector<Real>& h_avg_kw,
                          Gpu::HostVector<Real>& h_avg_p,
                          Gpu::HostVector<Real>& h_avg_pu  , Gpu::HostVector<Real>& h_avg_pv , Gpu::HostVector<Real>& h_avg_pw)
{
    // We assume that this is always called at level 0
    int lev = 0;

    // This will hold theta, uu, uv, uw, vv, vw, ww, uth, vth, wth, thth, k, ku, kv, kw, p, pu, pv, pw
    MultiFab mf_out(grids[lev], dmap[lev], 19, 0);

    MultiFab mf_vels(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);

    MultiFab  u_cc(mf_vels, make_alias, 0, 1); // u at cell centers
    MultiFab  v_cc(mf_vels, make_alias, 1, 1); // v at cell centers
    MultiFab  w_cc(mf_vels, make_alias, 2, 1); // w at cell centers

    average_face_to_cellcenter(mf_vels,0,
            Array<const MultiFab*,3>{&vars_new[lev][Vars::xvel],&vars_new[lev][Vars::yvel],&vars_new[lev][Vars::zvel]});

    int zdir = 2;
    auto domain = geom[0].Domain();

    // Sum in the horizontal plane
    h_avg_u  = sumToLine(mf_vels ,0,1,domain,zdir);
    h_avg_v  = sumToLine(mf_vels ,1,1,domain,zdir);
    h_avg_w  = sumToLine(mf_vels ,2,1,domain,zdir);

    // Divide by the total number of cells we are averaging over
    Real area_z = static_cast<Real>(domain.length(0)*domain.length(1));
    for (int k = 0; k < h_avg_u.size(); ++k) {
        h_avg_u[k] /= area_z; h_avg_v[k] /= area_z; h_avg_w[k] /= area_z;
    }

    MultiFab mf_cons(vars_new[lev][Vars::cons], make_alias, 0, 2);

    MultiFab p_hse (base_state[lev], make_alias, 1, 1); // p_0  is second component

    for ( MFIter mfi(mf_cons,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Array4<Real>& fab_arr = mf_out.array(mfi);
        const Array4<Real>& u_cc_arr =  u_cc.array(mfi);
        const Array4<Real>& v_cc_arr =  v_cc.array(mfi);
        const Array4<Real>& w_cc_arr =  w_cc.array(mfi);
        const Array4<Real>& cons_arr = mf_cons.array(mfi);
#if defined(ERF_USE_MOISTURE)
        const Array4<Real>&   qv_arr = qv[lev].array(mfi);
#endif
        const Array4<Real>&   p0_arr = p_hse.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            Real theta = cons_arr(i,j,k,RhoTheta_comp) / cons_arr(i,j,k,Rho_comp);
            fab_arr(i, j, k, 0) = theta;
            fab_arr(i, j, k, 1) = u_cc_arr(i,j,k) * u_cc_arr(i,j,k);   // uu
            fab_arr(i, j, k, 2) = u_cc_arr(i,j,k) * v_cc_arr(i,j,k);   // uv
            fab_arr(i, j, k, 3) = u_cc_arr(i,j,k) * w_cc_arr(i,j,k);   // uw
            fab_arr(i, j, k, 4) = v_cc_arr(i,j,k) * v_cc_arr(i,j,k);   // vv
            fab_arr(i, j, k, 5) = v_cc_arr(i,j,k) * w_cc_arr(i,j,k);   // vw
            fab_arr(i, j, k, 6) = w_cc_arr(i,j,k) * w_cc_arr(i,j,k);   // ww
            fab_arr(i, j, k, 7) = u_cc_arr(i,j,k) * theta; // uth
            fab_arr(i, j, k, 8) = v_cc_arr(i,j,k) * theta; // vth
            fab_arr(i, j, k, 9) = w_cc_arr(i,j,k) * theta; // wth
            fab_arr(i, j, k,10) = theta * theta;           // thth
            Real tke = 0.5*(fab_arr(i,j,k,1) + fab_arr(i,j,k,4) + fab_arr(i,j,k,6));
            fab_arr(i, j, k,11) = tke;                     // k
            fab_arr(i, j, k,12) = tke * u_cc_arr(i,j,k);   // ku
            fab_arr(i, j, k,13) = tke * v_cc_arr(i,j,k);   // kv
            fab_arr(i, j, k,14) = tke * w_cc_arr(i,j,k);   // kw
#if defined(ERF_USE_MOISTURE)
            Real p = getPgivenRTh(cons_arr(i, j, k, RhoTheta_comp), qv_arr(i,j,k));
#else
            Real p = getPgivenRTh(cons_arr(i, j, k, RhoTheta_comp));
#endif
            p -= p0_arr(i,j,k);
            fab_arr(i, j, k,15) = p;                       // p'
            fab_arr(i, j, k,16) = p * u_cc_arr(i,j,k);     // p'u
            fab_arr(i, j, k,17) = p * v_cc_arr(i,j,k);     // p'v
            fab_arr(i, j, k,18) = p * w_cc_arr(i,j,k);     // p'w
        });
    }

    h_avg_th   = sumToLine(mf_out, 0,1,domain,zdir);
    h_avg_uu   = sumToLine(mf_out, 1,1,domain,zdir);
    h_avg_uv   = sumToLine(mf_out, 2,1,domain,zdir);
    h_avg_uw   = sumToLine(mf_out, 3,1,domain,zdir);
    h_avg_vv   = sumToLine(mf_out, 4,1,domain,zdir);
    h_avg_vw   = sumToLine(mf_out, 5,1,domain,zdir);
    h_avg_ww   = sumToLine(mf_out, 6,1,domain,zdir);
    h_avg_uth  = sumToLine(mf_out, 7,1,domain,zdir);
    h_avg_vth  = sumToLine(mf_out, 8,1,domain,zdir);
    h_avg_wth  = sumToLine(mf_out, 9,1,domain,zdir);
    h_avg_thth = sumToLine(mf_out,10,1,domain,zdir);
    h_avg_k    = sumToLine(mf_out,11,1,domain,zdir);
    h_avg_ku   = sumToLine(mf_out,12,1,domain,zdir);
    h_avg_kv   = sumToLine(mf_out,13,1,domain,zdir);
    h_avg_kw   = sumToLine(mf_out,14,1,domain,zdir);
    h_avg_p    = sumToLine(mf_out,15,1,domain,zdir);
    h_avg_pu   = sumToLine(mf_out,16,1,domain,zdir);
    h_avg_pv   = sumToLine(mf_out,17,1,domain,zdir);
    h_avg_pw   = sumToLine(mf_out,18,1,domain,zdir);

    // Divide by the total number of cells we are averaging over
    for (int k = 0; k < h_avg_u.size(); ++k) {
        h_avg_th[k]  /= area_z;  h_avg_thth[k] /= area_z;
        h_avg_uu[k]  /= area_z;  h_avg_uv[k]   /= area_z;  h_avg_uw[k]  /= area_z;
        h_avg_vv[k]  /= area_z;  h_avg_vw[k]   /= area_z;  h_avg_ww[k]  /= area_z;
        h_avg_uth[k] /= area_z;  h_avg_vth[k]  /= area_z;  h_avg_wth[k] /= area_z;
        h_avg_k[k]   /= area_z;
        h_avg_ku[k]  /= area_z;  h_avg_kv[k]   /= area_z;  h_avg_kw[k]  /= area_z;
        h_avg_p[k]   /= area_z;
        h_avg_pu[k]  /= area_z;  h_avg_pv[k]   /= area_z;  h_avg_pw[k]  /= area_z;
    }
}
void
ERF::derive_stress_profiles(Gpu::HostVector<Real>& h_avg_tau11, Gpu::HostVector<Real>& h_avg_tau12,
                            Gpu::HostVector<Real>& h_avg_tau13, Gpu::HostVector<Real>& h_avg_tau22,
                            Gpu::HostVector<Real>& h_avg_tau23, Gpu::HostVector<Real>& h_avg_tau33)
{
    int lev = 0;

    // This will hold the stress tensor components
    MultiFab mf_out(grids[lev], dmap[lev], 6, 0);

    for ( MFIter mfi(mf_out,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Array4<Real>& fab_arr = mf_out.array(mfi);

        // NOTE: These are from the last RK stage...
        const Array4<const Real>& tau11_arr = Tau11_lev[lev]->const_array(mfi);
        const Array4<const Real>& tau12_arr = Tau12_lev[lev]->const_array(mfi);
        const Array4<const Real>& tau13_arr = Tau13_lev[lev]->const_array(mfi);
        const Array4<const Real>& tau22_arr = Tau22_lev[lev]->const_array(mfi);
        const Array4<const Real>& tau23_arr = Tau23_lev[lev]->const_array(mfi);
        const Array4<const Real>& tau33_arr = Tau33_lev[lev]->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            fab_arr(i, j, k, 0) = tau11_arr(i,j,k);
            fab_arr(i, j, k, 1) = tau12_arr(i,j,k);
            fab_arr(i, j, k, 2) = tau13_arr(i,j,k);
            fab_arr(i, j, k, 3) = tau22_arr(i,j,k);
            fab_arr(i, j, k, 4) = tau23_arr(i,j,k);
            fab_arr(i, j, k, 5) = tau33_arr(i,j,k);
        });
    }

    int zdir = 2;
    auto domain = geom[0].Domain();

    h_avg_tau11 = sumToLine(mf_out,0,1,domain,zdir);
    h_avg_tau12 = sumToLine(mf_out,1,1,domain,zdir);
    h_avg_tau13 = sumToLine(mf_out,2,1,domain,zdir);
    h_avg_tau22 = sumToLine(mf_out,3,1,domain,zdir);
    h_avg_tau23 = sumToLine(mf_out,4,1,domain,zdir);
    h_avg_tau33 = sumToLine(mf_out,5,1,domain,zdir);

    // Divide by the total number of cells we are averaging over
    Real area_z = static_cast<Real>(domain.length(0)*domain.length(1));
    for (int k = 0; k < h_avg_tau11.size(); ++k) {
        h_avg_tau11[k] /= area_z; h_avg_tau12[k] /= area_z; h_avg_tau13[k] /= area_z;
        h_avg_tau22[k] /= area_z; h_avg_tau23[k] /= area_z; h_avg_tau33[k] /= area_z;
    }
}
