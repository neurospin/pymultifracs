/*******************************************************************************
 * compute_struct_func_mex
 *
 * Mex function to compute efficiently structure functions.
 *
 * Inputs:
 *   - coefs: vector of multiresolution quantity at given scale
 *   - q: vector of values of q
 *   - flag_zq (boolean): compute structure functions if true.
 *   - flag_dh (boolean): compute quantities for D(q) and h(q) if true.
 *       
 *   Inputs are assumend to be correct and checked by matlab wrapper.
 *    
 * Roberto Leonarduzzi
 ******************************************************************************/

#include <stdlib.h>
#include <math.h>
#ifdef  _OPENMP
#include <omp.h>
#endif
#include "mex.h"


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

    /*** Read input input ***/
    double* coef = mxGetPr (prhs[0]);
    size_t len_coef = mxGetNumberOfElements (prhs[0]);

    double* q = mxGetPr (prhs[1]);
    size_t len_q = mxGetNumberOfElements (prhs[1]);

    int flag_zq = mxIsLogicalScalarTrue (prhs[2]);
    int flag_dh = mxIsLogicalScalarTrue (prhs[3]);

#ifdef _OPENMP
    int num_threads = (int) mxGetScalar (prhs[4]);
    omp_set_dynamic (0);    
    omp_set_num_threads (num_threads);
#endif
    
    /*** Preallocate output ***/
    double* s_q = NULL;
    double* f_dq = NULL;
    double* f_hq = NULL;
    if (flag_zq) {
        plhs[0] = mxCreateDoubleMatrix (1, len_q, mxREAL);
         s_q  = mxGetPr (plhs[0]);
    }
    else
    {
        plhs[0] = mxCreateDoubleMatrix (0, 0, mxREAL);
        s_q = mxCalloc (len_q, sizeof (double));
    }
    
    if (flag_dh)
    {
        plhs[1] = mxCreateDoubleMatrix (1, len_q, mxREAL);
        plhs[2] = mxCreateDoubleMatrix (1, len_q, mxREAL);
        f_dq = mxGetPr (plhs[1]);
        f_hq = mxGetPr (plhs[2]);
    }
    else
    {
        plhs[1] = mxCreateDoubleMatrix (0, 0, mxREAL);
        plhs[2] = mxCreateDoubleMatrix (0, 0, mxREAL);
    }


    if (!(flag_zq || flag_dh))  /* Nothing to do! */
        return;
    
    /*** Make computations ***/
    #pragma omp parallel for shared(coef, len_q, len_coef)
    for (size_t iq = 0; iq < len_q; ++iq)
    {
        for (size_t k = 0.0; k < len_coef; ++k)
            s_q[iq] += pow (coef[k], q[iq]);
    }
    
    if (flag_dh)
    {
        #pragma omp parallel for shared(coef, len_q, len_coef)
        for (size_t iq = 0; iq < len_q; ++iq)
        {
            double sum_u = 0.0, sum_v = 0.0, rtmp = 0.0;
            for (size_t k = 0.0; k < len_coef; ++k)
            {
                rtmp = pow (coef[k], q[iq]) / s_q[iq];
                sum_u += rtmp * log2 (rtmp);
                sum_v += rtmp * log2 (coef[k]);
            }
            
            f_dq[iq] = sum_u + log2 (len_coef);
            f_hq[iq] = sum_v;
        }
    }  // if (flag_dh)

    if (flag_zq)
    {
        /* Finish structure functions */
        for (size_t iq = 0; iq < len_q; ++iq)
            s_q[iq] = log2 (s_q[iq] / len_coef);
    }
    else
    {
        /* s_q is not part of output array in this case. Free it.  */
        mxFree (s_q);
    }
    
}
