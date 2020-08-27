#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include "linear.h"
#include "ranksvm.h"
#include <mpi.h>
#include <vector>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

extern int flag_cross_validation;
void print_null(const char *s) {}

void exit_with_help()
{
    if(mpi_get_rank() != 0)
        mpi_exit(1);
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 8)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for ranking\n"
	"	 8 -- L2-regularized L2-loss ranking support vector machine (primal)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"-S split : set split (default 0)\n"
	"  for distributed solver (only supported for ranksvm)\n"
	"	 0 -- Query-wise split\n"
	"	 1 -- Feature-wise split\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 8 and 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-t test_set_file : set test file name for showing performance per iteration\n"
	"-a : accelerate Hessian-vector product\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name, char *model_file_name);

// Reader of training data for the solvers except ranksvm
void read_problem(const char*filename);
// Reader of training data (query-wise)
void dist_read_problem(const char *filename);
void read_problem_csr(FILE *fp);
void read_problem_csc(FILE *fp);
// Reader of training data (feature-wise)
void dist_read_problem_fw(const char *filename);
void read_problem_fw_csr(FILE *fp);
void read_problem_fw_csc(FILE *fp);
// Reader of test data 
void read_problem_t(const char *filename);

void do_cross_validation();

struct feature_node *x_space;
struct parameter param;
struct problem prob, prob_t;
struct model* model_;
int flag_testing;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

	int global_n; 

	char input_file_name[1024];
	char test_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	parse_command_line(argc, argv, input_file_name, test_file_name, model_file_name);
	if(param.solver_type == L2R_L2LOSS_RANKSVM && mpi_get_size() > 1)
	{
		if(param.split_type == QW)
		{
			dist_read_problem(input_file_name);
			global_n = prob.n;
			mpi_allreduce(&global_n, 1, MPI_INT, MPI_MAX);
			prob.n = global_n;
		}
		else
		{
			dist_read_problem_fw(input_file_name);
		}
	
		if(flag_testing && mpi_get_rank() == 0)
			read_problem_t(test_file_name);
	}
	else
	{
		read_problem(input_file_name);
		if(param.solver_type == L2R_L2LOSS_RANKSVM && flag_testing)
			read_problem_t(test_file_name);
	}
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	if(flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model_=train(&prob, &prob_t, &param);
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(prob.query);
	free(x_space);
	free(line);

    MPI_Finalize();
	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	double result[2];


	if(param.solver_type == L2R_L2LOSS_RANKSVM)
	{
		rank_cross_validation(&prob,&param,nr_fold,result);
		if(mpi_get_rank() == 0)
		{
			printf("Cross Validation Pairwise Accuracy = %g%%\n",result[0]*100);
			printf("Cross Validation Mean NDCG = %g\n",result[1]);
		}
		free(target);
		return;
	}
	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		if(mpi_get_rank() == 0)
		{
			printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
			printf("Cross Validation Squared correlation coefficient = %g\n",
					((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
					((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
				  );
		}
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		if(mpi_get_rank() == 0)
			printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}

	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *test_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_L2LOSS_RANKSVM;
	param.split_type = QW;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.acc_Hv = false;
	flag_cross_validation = 0;
	flag_testing = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'S':
				param.split_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 't':
				flag_testing = 1;
				strcpy(test_file_name, argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'a':
				i--;
				param.acc_Hv = true;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	// priority of cross validation is higher than testing
	if(flag_cross_validation)
		flag_testing = 0;

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();
	
	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_RANKSVM:
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	prob.query = Malloc(int,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		prob.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid") == 0)
			{
				errno = 0;
				prob.query[i] = (int) strtol(val,&endptr,10);
				if(endptr == val || errno != 0 || *endptr != '\0')
					exit_input_error(i+1);
				continue;
			}
			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;
	fclose(fp);
}

void dist_read_problem(const char *filename)
{
	char *lptr, *nptr, *endptr;
	FILE *fp = fopen(filename,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}	

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	// read header (l and n)
	readline(fp);
	lptr = strtok(line," \t\n");
	prob.l = (int) strtol(lptr,&endptr,10);
	if(endptr == lptr || *endptr != '\0' || strtok(NULL," \t\n"))
		exit_input_error(1);
		
	readline(fp);
	nptr = strtok(line," \t\n");
	prob.n = (int) strtol(nptr,&endptr,10);
	if(endptr == nptr || *endptr != '\0' || strtok(NULL," \t\n"))
		exit_input_error(2);

	if(prob.l > prob.n/10)
	{
		prob.format = CSR;
		printf("CSR [%d]\n", mpi_get_rank());
		read_problem_csr(fp);
	}
	else
	{
		prob.format = CSC;
		printf("CSC [%d]\n", mpi_get_rank());
		read_problem_csc(fp);
	}
}
void read_problem_csr(FILE *fp)
{
	int inst_max_index, i;
	size_t elements, j;
	char *endptr;
	char *idx, *val, *label;
	elements = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
	}
	rewind(fp);

	prob.bias=bias;
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	prob.query = Malloc(int,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	// skip header (l and n)
	for(i=0;i<2;i++)
		readline(fp);
	
	j=0;
	for(i=0;i<prob.l;i++)
	{
		prob.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid") == 0)
			{
				errno = 0;
				prob.query[i] = (int) strtol(val,&endptr,10);
				if(endptr == val || errno != 0 || *endptr != '\0')
					exit_input_error(i+1);
				continue;
			}
			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n++;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}

	fclose(fp);
}
void read_problem_csc(FILE *fp)
{
	int i, inst_index;
	size_t elements, j;
	double inst_value;
	char *endptr;
	char *idx, *val, *label;
	elements = 0;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
	}
	rewind(fp);

	prob.bias=bias;
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.n);
	prob.query = Malloc(int,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.n);
	std::vector <std::vector<int> > feat_idx(prob.n);
	std::vector <std::vector<double> > feat_val(prob.n);

	// skip header
	for(i=0;i<2;i++)
		readline(fp);

	for(i=0;i<prob.l;i++)
	{
		prob.query[i] = 0;
		readline(fp);

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
		{
			exit_input_error(i+1);
		}
		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
		{
			exit_input_error(i+1);
		}
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid") == 0)
			{
				errno = 0;
				prob.query[i] = (int) strtol(val,&endptr,10);
				if(endptr == val || errno != 0 || *endptr != '\0')
					exit_input_error(i+1);
				continue;
			}

			errno = 0;
			inst_index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0')
				exit_input_error(i+1);
			feat_idx[inst_index-1].push_back(i);

			errno = 0;
			inst_value = strtod(val, &endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
			feat_val[inst_index-1].push_back(inst_value);

		}
	}

	std::vector <std::vector<int> >::iterator idx_i;
	std::vector <std::vector<double> >::iterator val_i;
	std::vector <int>::iterator idx_j;
	std::vector <double>::iterator val_j;
	i = 0, j = 0;
	for(idx_i=feat_idx.begin(),val_i=feat_val.begin(); 
		idx_i!=feat_idx.end(); idx_i++,val_i++)
	{
		prob.x[i] = &x_space[j];
		for(idx_j=(*idx_i).begin(),val_j=(*val_i).begin();
			idx_j!=(*idx_i).end(); idx_j++,val_j++)
		{
			x_space[j].index = (*idx_j)+1;
			x_space[j].value = (*val_j);
			j++;
		}
		//put -1 at the end
		x_space[j].index = -1;
		i++;
		j++;
	}
	if(prob.bias >= 0)
	{
		prob.n++;
		for(i=0;i<prob.l;i++)
		{
			x_space[j].index = prob.n-1;
			x_space[j].value = prob.bias;
			j++;
		}
		x_space[j].index = -1;
	}
	feat_idx.clear();
	feat_val.clear();
	fclose(fp);
}
void dist_read_problem_fw(const char *filename)
{
	char *lptr, *nptr, *endptr;
	FILE *fp = fopen(filename,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	
	max_line_len = 1024;
	line = Malloc(char,max_line_len);

	// read header (l and n)
	readline(fp);
	lptr = strtok(line," \t\n");
	prob.l = (int) strtol(lptr,&endptr,10);
	if(endptr == lptr || *endptr != '\0' || strtok(NULL," \t\n"))
		exit_input_error(1);
	readline(fp);
	nptr = strtok(line," \t\n");
	prob.n = (int) strtol(nptr,&endptr,10);
	if(endptr == nptr || *endptr != '\0' || strtok(NULL," \t\n"))
		exit_input_error(2);
	/*	
	if(prob.l > prob.n)
	{
		prob.format = CSC;
		printf("CSC [%d]\n", mpi_get_rank());
		read_problem_fw_csc(fp);
	}
	else
	*/
	{
		prob.format = CSR;
		printf("CSR [%d]\n", mpi_get_rank());
		read_problem_fw_csr(fp);
	}
	
}
void read_problem_fw_csr(FILE *fp)
{
	int feat_max_index, nr_header, i;
	size_t elements, j;
	char *endptr;
	char *idx, *val;
	elements = 0;
	
	// label and (qid)	
	nr_header = param.solver_type==L2R_L2LOSS_RANKSVM?2:1;
	while(readline(fp)!=NULL)
	{
		// instances
		char *p = strtok(line," \t");
		elements++;
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last instances
				break;
			elements++;
		}
	}
	rewind(fp);
	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.n);
	prob.query = Malloc(int,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.n);

	//skip header
	for(i=0;i<2;i++)
		readline(fp);
	//read label (qid)
	for(i=0;i<nr_header;i++)
	{
		readline(fp);
		char *ptr = strtok(line," \t");
		if(i == 0) 
			prob.y[0] = strtod(ptr,&endptr);
		else 
			prob.query[0] = (int) strtol(ptr,&endptr,10);
		if(endptr == ptr || *endptr != '\0')
			exit_input_error(i+1);

		j = 1;
		while(1)
		{
			ptr = strtok(NULL," \t");
			if(ptr == NULL || *ptr == '\n')
				break;
			if(i == 0) 
				prob.y[j] = strtod(ptr,&endptr);
			else 
				prob.query[j] = (int) strtol(ptr,&endptr,10);
			if(endptr == ptr || *endptr != '\0')
			{
				exit_input_error(i+1);
			}
			j++;
		}
	}

	j = 0;
	for(i=0;i<prob.n;i++)
	{
		feat_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		
		bool is_first = true;
		while(1)
		{
			if(is_first) 
			{
				idx = strtok(line,":");
				is_first = false;
			}
			else 
				idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(idx == NULL || val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);

			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= feat_max_index)
				exit_input_error(i+1);
			else
				feat_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
			
			++j;
		}

		x_space[j++].index = -1;
	}
	if(prob.bias >= 0)
	{
		prob.n++;
		for(i=1;i<=prob.l;i++)
		{
			x_space[j].index = i;
			x_space[j++].value = prob.bias;
		}
	}
	fclose(fp);
}
void read_problem_fw_csc(FILE *fp)
{
	int feat_index, nr_header, i;
	size_t elements, j;
	double feat_value;
	char *endptr;
	char *idx, *val;
	elements = 0;
	
	// label and (qid)
	nr_header = param.solver_type==L2R_L2LOSS_RANKSVM?2:1;
	while(readline(fp)!=NULL)
	{
		// instances
		char *p = strtok(line," \t");
		elements++;
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last instances
				break;
			elements++;
		}
	}
	rewind(fp);
	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	prob.query = Malloc(int,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);
	std::vector< std::vector<int> > inst_idx(prob.l);
	std::vector< std::vector<double> > inst_val(prob.l);

	//skip header (l and n)
	for(i=0;i<2;i++)
		readline(fp);
	//read label (qid)
	for(i=0;i<nr_header;i++)
	{
		readline(fp);
		char *ptr = strtok(line," \t");
		if(i == 0) 
			prob.y[0] = strtod(ptr,&endptr);
		else 
			prob.query[0] = (int) strtol(ptr,&endptr,10);
		if(endptr == ptr || *endptr != '\0')
			exit_input_error(i+1);
		
		j = 1;
		while(1)
		{
			ptr = strtok(NULL," \t");
			if(ptr == NULL || *ptr == '\n')
				break;
			if(i == 0) 
				prob.y[j] = strtod(ptr,&endptr);
			else 
				prob.query[j] = (int) strtol(ptr,&endptr,10);
			if(endptr == ptr || *endptr != '\0')
				exit_input_error(i+1);
			j++;
		}
	}

	j = 0;
	for(i=0;i<prob.n;i++)
	{
		readline(fp);
		bool is_first = true;
		while(1)
		{
			if(is_first) 
			{
				idx = strtok(line,":");
				is_first = false;
			}
			else 
				idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(idx == NULL || val == NULL)
				break;

			errno = 0;
			feat_index = (int) strtol(idx,&endptr,10);
			inst_idx[feat_index-1].push_back(i);
			if(endptr == idx || errno != 0 || *endptr != '\0')
				exit_input_error(i+1);

			errno = 0;
			feat_value = strtod(val, &endptr);
			inst_val[feat_index-1].push_back(feat_value);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
		}
	}
	
	std::vector <std::vector<int> >::iterator idx_i;
	std::vector <std::vector<double> >::iterator val_i;
	std::vector <int>::iterator idx_j;
	std::vector <double>::iterator val_j;
	
	i = 0, j = 0;
	for(idx_i=inst_idx.begin(),val_i=inst_val.begin(); 
		idx_i!=inst_idx.end(); idx_i++,val_i++)
	{
		prob.x[i] = &x_space[j];
		for(idx_j=(*idx_i).begin(),val_j=(*val_i).begin();
			idx_j!=(*idx_i).end(); idx_j++, val_j++)
		{
			x_space[j].index = (*idx_j)+1;
			x_space[j].value = (*val_j);
			j++;
		}
		//put -1 at the end
		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;
		x_space[j++].index = -1;
		i++;
	}
	if(prob.bias >= 0)
	{
		prob.n++;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	inst_idx.clear();
	inst_val.clear();
	fclose(fp);
}
void read_problem_t(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	prob_t.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob_t.l++;
	}
	rewind(fp);
	
	prob_t.bias=bias;
	prob_t.y = Malloc(double,prob_t.l);
	prob_t.x = Malloc(struct feature_node *,prob_t.l);
	prob_t.query = Malloc(int,prob_t.l);
	x_space = Malloc(struct feature_node,elements+prob_t.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob_t.l;i++)
	{
		prob_t.query[i] = 0;
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob_t.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob_t.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			if (strcmp(idx,"qid") == 0)
			{
				errno = 0;
				prob_t.query[i] = (int) strtol(val,&endptr,10);
				if(endptr == val || errno != 0 || *endptr != '\0')
					exit_input_error(i+1);
				continue;
			}
			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob_t.bias >= 0)
			x_space[j++].value = prob_t.bias;

		x_space[j++].index = -1;
	}

	if(prob_t.bias >= 0)
	{
		prob_t.n=max_index+1;
		for(i=1;i<prob_t.l;i++)
			(prob_t.x[i]-2)->index = prob_t.n;
		x_space[j-2].index = prob_t.n;
	}
	else
		prob_t.n=max_index;

	fclose(fp);
}
