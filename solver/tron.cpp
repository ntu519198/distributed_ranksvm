#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <mpi.h>
#include <vector>
#include "tron.h"
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { QW, FW }; /* split_type */
extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

double dnrm2(int *, double *x, int *, int);
double ddot(int *, double *, int *, double *, int *, int);
int daxpy(int *, double *, double *, int *, double *, int *);
int dscal(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

// For timer
clock_t start_time;
clock_t stop_time;
clock_t comm_start_time;
clock_t comm_stop_time;
double total_sec;
double comm_sec;

template<typename T>
static void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
    std::vector<T> buf_reduced(count);
    MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
    for(int i=0;i<count;i++)
        buf[i] = buf_reduced[i];
}

static int mpi_get_rank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void TRON::info(const char *fmt,...)
{
    if(mpi_get_rank() != 0)
        return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, double eps, int max_iter, int split_type)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->split_type=split_type;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	// Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double delta, snorm, one=1.0, zero=0.0;
	double alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	int nr_pair; /*nr_pair > 0 for ranksvm*/
	double *result;
	double *s = new double[n];
	double *r = new double[n];
	double *w_new = new double[n];
	double *g = new double[n];

	for (i=0; i<n; i++)
		w[i] = 0;
	
	nr_pair = fun_obj->get_nr_pair();
	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	double gnorm1 = dnrm2(&n, g, &inc, split_type);
	
	comm_sec = 0;
	iter = 1;
	if(nr_pair > 0)
	{
		result = new double[2];
		fun_obj->eval(w, result);
		info("iter %3d f %5.15f PWACC %lf CG %3d time %lf\n", 
			iter, f*nr_pair, result[0], (int)zero, zero);
	}
	else
	{
		info("iter %3d f %5.15f CG %3d time %lf\n", 
			iter, f, 0, 0.0);
	}

	//start timer
	MPI_Barrier(MPI_COMM_WORLD);
	start_time = clock();

	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	if(nr_pair > 0)
		delta = gnorm1*nr_pair; //using gnorm1 as initial value of delta is too small
	else 
		delta = gnorm1;
	
	double gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	while (iter <= max_iter && search)
	{
		cg_iter = trcg(delta, g, s, r);

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		gs = ddot(&n, g, &inc, s, &inc, split_type);
		prered = -0.5*(gs-ddot(&n, s, &inc, r, &inc, split_type));
		fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
		actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		snorm = dnrm2(&n, s, &inc, split_type);
		if (iter == 1)
			delta = min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
		else
			delta = max(delta, min(alpha*snorm, sigma3*delta));
		
		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(double)*n);
			f = fnew;
			fun_obj->grad(w, g);

			gnorm = dnrm2(&n, g, &inc, split_type);
			MPI_Barrier(MPI_COMM_WORLD);
			stop_time = clock();
			total_sec = double(stop_time-start_time)/CLOCKS_PER_SEC;
		
			if(nr_pair > 0)
			{
				fun_obj->eval(w, result);
				info("iter %2d f %5.15f PWACC %lf CG %3d comm %lf time %lf\n", 
					iter, f*nr_pair, result[0], cg_iter, comm_sec, total_sec);
			}
			else
			{
				info("iter %2d f %5.15f CG %3d time %lf\n", 
					iter, f, cg_iter, total_sec);
			}
			if (gnorm <= eps*gnorm1)
            {
				info("total_sec = %10lf\n", total_sec);
				break;
            }
		}
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 0 && prered <= 0)
		{
			info("WARNING: actred and prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
	if(nr_pair > 0)
		delete[] result;
}
int TRON::trcg(double delta, double *g, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = 0.1*dnrm2(&n, g, &inc, split_type);

	int cg_iter = 0;
	rTr = ddot(&n, r, &inc, r, &inc, split_type);
	while (1)
	{
		if (dnrm2(&n, r, &inc, split_type) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot(&n, d, &inc, Hd, &inc, split_type);
		daxpy(&n, &alpha, d, &inc, s, &inc);
		if (dnrm2(&n, s, &inc, split_type) > delta)
		{
			info("cg reaches trust region boundary\n");
			alpha = -alpha;
			daxpy(&n, &alpha, d, &inc, s, &inc);

			double std = ddot(&n, s, &inc, d, &inc, split_type);
			double sts = ddot(&n, s, &inc, s, &inc, split_type);
			double dtd = ddot(&n, d, &inc, d, &inc, split_type);
			double dsq = delta*delta;
			double rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			daxpy(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			daxpy(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		daxpy(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot(&n, r, &inc, r, &inc, split_type);
		beta = rnewTrnew/rTr;
		dscal(&n, &beta, d, &inc);
		daxpy(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}

double dnrm2(int *n, double *x, int *incx, int split_type)
{
	double norm;

	if(split_type == FW)
	{
		
		double norm_ = dnrm2_(n,x,incx);
		norm = norm_*norm_;
		MPI_Barrier(MPI_COMM_WORLD);
		comm_start_time = clock();
		mpi_allreduce(&norm,1,MPI_DOUBLE,MPI_SUM);
		MPI_Barrier(MPI_COMM_WORLD);
		comm_stop_time = clock();
		comm_sec += (double)(comm_stop_time-comm_start_time)/CLOCKS_PER_SEC;
		norm = sqrt(norm);
		return norm;
	}
	
	return dnrm2_(n,x,incx);
}

double ddot(int *n, double *sx, int *incx, double *sy, int *incy, int split_type)
{
	double dot;
	if(split_type == FW)
	{
		dot = ddot_(n,sx,incx,sy,incy);
		MPI_Barrier(MPI_COMM_WORLD);
		comm_start_time = clock();
		mpi_allreduce(&dot,1,MPI_DOUBLE,MPI_SUM);
		MPI_Barrier(MPI_COMM_WORLD);
		comm_stop_time = clock();
		comm_sec += (double)(comm_stop_time-comm_start_time)/CLOCKS_PER_SEC;
		return dot;
	}
	return ddot_(n,sx,incx,sy,incy);
}

int daxpy(int *n, double *sa, double *sx, int *incx, double *sy, int *incy)
{
	return daxpy_(n,sa,sx,incx,sy,incy);
}

int dscal(int *n, double *sa, double *sx, int *incx)
{
	return dscal_(n,sa,sx,incx);
}
