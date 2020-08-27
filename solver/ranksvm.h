#ifndef _RANKSVM_H
#define _RANKSVM_H
#include "linear.h"
#include "tron.h"
#ifdef __cplusplus
extern "C" {
#endif
void eval_list(double *label, double *target, int *query, int l, double *result_ret, int split_type);
void rank_cross_validation(const problem *prob, const parameter *param, int nr_fold, double *result);

struct id_and_value
{
	int id;
	double value;
};

class l2r_l2_ranksvm_fun: public function
{
public:
	l2r_l2_ranksvm_fun(const problem *prob, const problem *prob_t, double C, int split_type, bool acc_Hv);
	~l2r_l2_ranksvm_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	void eval(double *w, double *result);

	int get_nr_variable(void);
	int get_nr_pair(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);
	void Xv_t(int w_size, double *v, double *Xv);
	void Mv(feature_node **M, double *v, double *Mv, int nr_row, int nr_col);
	void MTv(feature_node **M, double *v, double *MTv, int nr_row, int nr_col);
	
	void blockXv(double *v, double *Xv, int *perm_q, int count);
	void blockXTv(double *v, double *XTv, int *perm_q, int count);

	bool acc_Hv;
	int split_type;
	double C;
	double *z;
	int *l_plus;
	int *l_minus;
	double *gamma_plus;
	double *gamma_minus;
	double *ATAXw;
	double *ATe;
	int nr_query;
	int nr_pair;
	int *perm;
	int *start;
	int *count;
	id_and_value **order_perm;
	int *nr_class;
	int *int_y;
	const problem *prob;
	const problem *prob_t;
};

#ifdef __cplusplus
}
#endif

#endif /* _RANKSVM_H */

