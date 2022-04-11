#include "ERF.H"

void
ERF::make_metrics(int lev)
{
    auto dx = geom[lev].CellSize();
    Real dz = dx[2];
    for ( MFIter mfi(z_phys_cc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.tilebox();
        Array4<Real const> z_nd = z_phys_nd[lev].const_array(mfi);
        Array4<Real      > z_cc = z_phys_cc[lev].array(mfi);
        Array4<Real      > detJ = detJ_cc[lev].array(mfi);
        amrex::ParallelFor(bx, [=]
           AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               z_cc(i, j, k) = .125 * (
                         z_nd(i,j,k  ) + z_nd(i+1,j,k  ) + z_nd(i,j+1,k  ) + z_nd(i+1,j+1,k  )
                        +z_nd(i,j,k+1) + z_nd(i+1,j,k+1) + z_nd(i,j+1,k+1) + z_nd(i+1,j+1,k+1) );
               detJ(i, j, k) = .25 * ( z_nd(i,j,k+1) + z_nd(i+1,j,k+1) + z_nd(i,j+1,k+1) + z_nd(i+1,j+1,k+1)
                                      -z_nd(i,j,k  ) - z_nd(i+1,j,k  ) - z_nd(i,j+1,k  ) - z_nd(i+1,j+1,k  ) ) ;
               });
    }
}