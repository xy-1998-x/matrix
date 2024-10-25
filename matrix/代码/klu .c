#define _GNU_SOURCE
// #define MATRIX_TEST
#define REPEAT 1000000
#include <stdio.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <sched.h>
#include <emmintrin.h>
#include <klu.h>
#include <math.h>


#define EPS 1e-9

#define SRAND() (float)((double)rand() / ((double)RAND_MAX / 2.0F)) - 1.0F;
#define DRAND() ((double)rand() / ((double)RAND_MAX / 2.0F)) - 1.0F;


#define rte_mb() _mm_mfence()
#define rte_wmb() _mm_sfence()
#define rte_rmb() _mm_lfence()

#define rsn_pause(n)        \
    do                      \
    {                       \
        register int i = n; \
        while (i--)         \
        {                   \
            _mm_pause();    \
        }                   \
    } while (0)

static inline uint64_t rte_rdtsc(void)
{
    union
    {
        uint64_t tsc_64;
        struct
        {
            uint32_t lo_32;
            uint32_t hi_32;
        };
    } tsc;

    asm volatile("rdtsc" : "=a"(tsc.lo_32),
                           "=d"(tsc.hi_32));
    return tsc.tsc_64;
}

typedef struct
{
    bool verbose;
    int32_t core;
    int32_t matrix_size;
    int64_t repeat;
    int32_t warm_steps;
} App_ctx;

App_ctx g_app_ctx = {
    .verbose = false,
    .core = 6,
    .matrix_size = 57,
    .repeat = REPEAT,
    .warm_steps = 10};

typedef struct
{
    uint64_t stat_0_10000ns;
    uint64_t stat_10000_20000ns;
    uint64_t stat_20000_30000ns;
    uint64_t stat_30000_40000ns;
    uint64_t stat_40000_50000ns;
    uint64_t stat_50000_60000ns;
    uint64_t stat_60000_70000ns;
    uint64_t stat_70000_80000ns;
    uint64_t stat_80000_90000ns;
    uint64_t stat_90000_100000ns;
    uint64_t stat_over_100000ns;
} Stat;

static Stat total_stat, factorize_stat, solve_stat;

typedef struct {
	double* A;	
	double* original_B;
	double* B;
	int matrix_size;
	int *Ap;
	int *Ai;
	double *Ax;
	klu_symbolic *Symbolic ;
	klu_numeric *Numeric ;
	klu_common Common ;
} tFuncBlockDemoContext;

static tFuncBlockDemoContext* context; //static的变量定义

struct option long_options[] = {
    {"verbose", no_argument, NULL, 'v'},
    {"matrix", optional_argument, NULL, 'm'},
    {"repeat", optional_argument, NULL, 'r'},
    {"core", optional_argument, NULL, 'c'},
    {"help", no_argument, NULL, 'h'},
    {0, 0, 0, 0}};

static const char *short_options = "c:m:r:v:h";
static const char *usage = "\nuseage:\n"
                           "-v --verbese       verbose output\n"
                           "-m --matrix        the matrix size.\n"
                           "-r --repeat        the repeat number.\n"
                           "-c --core          running core.\n"
                           "-h --help          Show help information.\n";

//这是根据命令行参数来设置结构体App_ctx中相应变量的功能
static void parse_program_arg(int argc, char **argv) //解析程序的命令行参数
{
    int c;
    int option_index = 0;
    int count = 0;

    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) //处理命令行选项
    {
        count++;
        switch (c)
        {
        case 'v':
        {
            g_app_ctx.verbose = true;
            break;
        }
        case 'm':
        {
            g_app_ctx.matrix_size = atoi(optarg);
            count++;
            break;
        }
        case 'r':
        {
            g_app_ctx.repeat = atoi(optarg);
            count++;
            break;
        }
        case 'c':
        {
            g_app_ctx.core = atoi(optarg);
            count++;
            break;
        }
        case 'h':
        {
            printf("%s", usage);
            exit(0);
            break;
        }
        default:
            printf("unknown option!\n");
            break;
        }
    }
    if (count < argc - 1)
    {
        printf("%s", usage);
        exit(0);
    }
}

void bind_cpu_init()
{
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();
    CPU_ZERO(&cpuset);

    CPU_SET(g_app_ctx.core, &cpuset);

    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0)
    {
        perror("pthread_setaffinity_np");
        exit(-1);
    }
}

void sched_init()
{
    int sched_policy = SCHED_FIFO;
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(sched_policy);
    int rc = pthread_setschedparam(pthread_self(), sched_policy, &param);
    if (rc != 0)
    {
        perror("pthread_setschedparam");
        exit(-1);
    }
}

void print_matrix_rowmajor(double *mat, int m, int n, int ldm)
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            printf(" %1.9lf", mat[i * ldm + j]);
        printf("\n");
    }
}

void print_matrix_colmajor(double *mat, int m, int n, int ldm)
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            printf(" %1.9lf", mat[i + j * ldm]);
        printf("\n");
    }
}

void init_context()
{		
	context = malloc(sizeof(tFuncBlockDemoContext));
	if (!context)
	{
		printf("malloc tFuncBlockDemoContext error");
		exit(-1);
	}
	memset(context, 0x00, sizeof(tFuncBlockDemoContext));
	
	context->matrix_size =  g_app_ctx.matrix_size;
	
	context->A =  (double *)malloc(context->matrix_size * context->matrix_size * sizeof(double));
	if (!context->A)
	{
		printf("emt_local_malloc A error");
		exit(-1);
	}
	
	context->original_B =  (double *)malloc(context->matrix_size * sizeof(double));
	if (!context->original_B)
	{
		printf("emt_local_malloc original_B error");
		exit(-1);
	}

//context->B并没有用到 
	context->B =  (double *)malloc(context->matrix_size * sizeof(double));
	if (!context->B)
	{
		printf("emt_local_malloc B error");
		exit(-1);
	}
}

//将二维矩阵以一个一维数组存储
int denseToCscSparse(const double A[], int n, int **Ap, int **Ai, double **Ax)
{
    int nz = 0;
    for (int i = 0; i < n * n; i++)
    {   
        if(fabs(A[i]) >= EPS)
        {   
            nz += 1;
        }
    }

    int *_Ap = (int *)malloc((n + 1) * sizeof(int));     // A的列指针
    int *_Ai = (int *)malloc(nz * sizeof(int));          // A的行索引
    double *_Ax = (double *)malloc(nz * sizeof(double)); // A的非零元素值
	
	if(!_Ap || !_Ai || !_Ax)
	{
		return 0;
	}

    memset(_Ap, 0, (n + 1) * sizeof(int));
    memset(_Ai, 0, nz * sizeof(int));
    memset(_Ax, 0, nz * sizeof(double));

    int k = 0;
    for (int j = 0; j < n; j++) // j为列号
    {
        _Ap[j + 1] += _Ap[j];
        for (int i = 0; i < n; i++) // i为行号
        {
            double value = A[i * n + j]; // 第i行第j列的值
            //记录非0值的三元组信息
            if (fabs(value) >= EPS)
            {
                _Ap[j + 1]++;
                _Ai[k] = i;  //记录行号
                _Ax[k] = value; //记录对应的值
                k++;
            }
        }
    }

    *Ap = _Ap;
    *Ai = _Ai;
    *Ax = _Ax;
	
	return 1;
}


int hasBOM(FILE *stream) {
    unsigned char bom[3];
    if (fread(bom, 1, 3, stream) == 3 && bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF) {
        return 1;
    } else {
        // Rewind the file pointer to the beginning
        fseek(stream, 0, SEEK_SET);
        return 0;
    }
}


void init_matrix(tFuncBlockDemoContext *context)
{
    FILE *stream;
    char *line = NULL;
    size_t len = 0;
    ssize_t nread;

    /* ************************************************************* */
    //读取矩阵A 读取向量b
    char filename[64] = {0};
    snprintf(filename, 64, "matrix_%d.csv", context->matrix_size);
    stream = fopen(filename, "r"); //stream 表示文件指针 代表了打开的文件资源 
    if (stream == NULL) {
        printf("fopen matrix.csv error");
        exit(-1);
    }
    if (hasBOM(stream)) 
	{
        // Skip BOM if present
        fseek(stream, 3, SEEK_SET);
    }


    int idx = 0;
   
//通过strtok来分割 指向第一个以逗号分隔的子字符串
     while ((nread = getline(&line, &len, stream))!= -1) 
	{
        char *token = strtok(line, ",");
        while (token!= NULL) 
        {
            context->A[idx++]= atof(token);
            token = strtok(NULL, ","); //从上次分割后的位置开始查找下一个分割符
        }
    }
/*
     for (int i = 0 ; i < idx ; i++)
        {
          printf ("A [%d] = %g\n", i, context->A[i]) ;  
        } 
*/
    free(line);
    fclose(stream);

    //对矩阵进行CSC变换  AX是非0元素数组
	if(!denseToCscSparse(context->A, context->matrix_size, &context->Ap, &context->Ai, &context->Ax))
	{
		printf("dense matrix to csc matrix err\n");
		exit(-1);
	}
	
    //**************************************************************************** 读取向量b */
    FILE *stream_b;
    char *line_b = NULL;
    size_t len_b = 0;
    ssize_t nread_b;
    double val_b;
    int row_b = 0;

    char filename_b[64] = {0};
    snprintf(filename_b,64,"matrix_b_%d.csv",context->matrix_size);
    stream_b = fopen(filename_b,"r");
    if (stream_b == NULL) {
        printf("fopen matrix_b.csv error");
        exit(-1);
    }
       if (hasBOM(stream_b)) 
	{
        // Skip BOM if present
        fseek(stream_b, 3, SEEK_SET);
    }

    while( (nread_b = getline(&line_b,&len_b,stream_b)) != -1 ) //stream_b这个流中有多少行就会循环执行多少次
    {
   
        if( nread_b >0 && line_b[nread_b-1] == '\n' ) //这个line就是getline读取的一行的内容 
        {
            line_b[nread_b-1] = '\0';
        }
         val_b = atof(line_b);
         context->original_B[row_b] = val_b;
         row_b++;
    }

/*      for (int i = 0 ; i < row_b ; i++)
        {
          printf ("B [%d] = %g\n", i,  context->original_B[i]) ;  
        } 
*/
    free(line_b);
    fclose(stream_b);

    //**************************************************************************** 读取普通矩阵则不需要设值 */
	 int nnz = context->Ap[context->matrix_size];
	 for(int i = 0; i < nnz; i++)
	 {
	 	context->Ax[i] = DRAND(); //对CSC格式中的非0元素数组重新随机赋值
         }
   
     for(int i = 0; i < context->matrix_size; i++)
    {
       context->original_B[i] = DRAND(); //对向量B数据随机赋值
    }
   
   //**************************************************************************** 读取普通矩阵则不需要设值 */


	klu_defaults (&context->Common) ; //默认的库参数

    //符号矩阵 描述矩阵的结构和特性，为算法选择和优化提供关键信息
    context->Symbolic = klu_analyze (context->matrix_size, context->Ap, context->Ai, &context->Common) ;
	if(!context->Symbolic)
	{
		printf("klu_analyze err\n");
		exit(-1);
	}

    //因式分解 存储矩阵分解后的结果信息
    context->Numeric = klu_factor (context->Ap, context->Ai, context->Ax, context->Symbolic, &context->Common) ;
	if(!context->Numeric)
	{
		printf("klu_factor err\n");
		exit(-1);
	}
}


long long getSystemNanoSecond()
{
    struct timespec ts;
    (void)clock_gettime(CLOCK_REALTIME, &ts);
    long long milliseconds = (ts.tv_sec * 1000 * 1000) + (ts.tv_nsec / 1000);
    return milliseconds;
}

void solve()
{
    uint64_t time_begin;
    uint64_t time_end;
    uint64_t time_factorize;

    uint64_t time_used;
    uint64_t time_used_factorize;
    uint64_t time_used_solve;

    uint64_t min_time = 10000000000000ULL;
    uint64_t max_time = 0;
    uint64_t factorize_min_time = 10000000000000ULL;
    uint64_t factorize_max_time = 0;
    uint64_t solve_min_time = 10000000000000ULL;
    uint64_t solve_max_time = 0;
    uint64_t total_time = 0;
    uint64_t total_time_factorize = 0;
    uint64_t total_time_solve = 0;
    //int cpu_mhz = 3000;
    int cpu_ghz = 3;

    int nrhs = 1;
    

      double * tmp = malloc(context->matrix_size * sizeof(double));
      memcpy(tmp,context->original_B,context->matrix_size * sizeof(double));
      

      for (int r = 0; r < g_app_ctx.repeat + g_app_ctx.warm_steps ; ++r)
    {

	 for(int i = 0;i<context->matrix_size;i++)                                                                                          {                                                                                                                                    context->original_B[i]= DRAND();                                                                                                 }
	
        time_begin = rte_rdtsc();
        rte_rmb();

         ///////////////////
	 for(int i=0;i<(context->matrix_size)/4;i++)
	 {
	     tmp[i]=DRAND();	 
	 }
	 //////////////////	
		
        // klu_solve 返回值 TRUE 1  FALSE 0  调用 klu_solve 函数求解线性方程组后，b 数组中更新 存储了方程组的解
		if(!klu_solve (context->Symbolic, context->Numeric, context->matrix_size, nrhs, context->original_B, &context->Common))
		{
			printf("klu_rsolve err\n");
			exit(-1);
		}
       
        time_end = rte_rdtsc();
		rte_rmb();
	
        
/*
        time_used = (time_end - time_begin) / cpu_ghz;
        total_time += time_used;

        if (time_used > max_time)
            max_time = time_used;

        if (time_used < min_time)
            min_time = time_used;

        if (time_used < 10)
        {
            total_stat.stat_0_10us++;
        }
        else if (time_used >= 10 && time_used < 20)
        {
            total_stat.stat_10_20us++;
        }
        else if (time_used >= 20 && time_used < 30)
        {
            total_stat.stat_20_30us++;
        }
        else if (time_used >= 30 && time_used < 40)
        {
            total_stat.stat_30_40us++;
        }
        else if (time_used >= 40 && time_used < 50)
        {
            total_stat.stat_40_50us++;
        }
        else if (time_used >= 50 && time_used < 60)
        {
            total_stat.stat_50_60us++;
        }
        else if (time_used >= 60 && time_used < 70)
        {
            total_stat.stat_60_70us++;
        }
        else if (time_used >= 70 && time_used < 80)
        {
            total_stat.stat_70_80us++;
        }
        else if (time_used >= 80 && time_used < 90)
        {
            total_stat.stat_80_90us++;
        }
        else if (time_used >= 90 && time_used < 100)
        {
            total_stat.stat_90_100us++;
        }
        else
        {
            total_stat.stat_over_100us++;
        }
*/


        time_used_solve = (time_end - time_begin) / cpu_ghz;
        total_time_solve += time_used_solve;
        if (time_used_solve > solve_max_time)
            solve_max_time = time_used_solve;

        if (time_used_solve < solve_min_time)
            solve_min_time = time_used_solve;

        if (time_used_solve < 10000)
        {
            solve_stat.stat_0_10000ns++;
        } 
        else if (time_used_solve >= 10000 && time_used_solve < 20000)
        {
            solve_stat.stat_10000_20000ns++;
        }
        else if (time_used_solve >= 20000 && time_used_solve < 30000)
        {
            solve_stat.stat_20000_30000ns++;
        }
        else if (time_used_solve >= 30000 && time_used_solve < 40000)
        {
            solve_stat.stat_30000_40000ns++;
        }
        else if (time_used_solve >= 40000 && time_used_solve < 50000)
        {
            solve_stat.stat_40000_50000ns++;
        }
        else if (time_used_solve >= 50000 && time_used_solve < 60000)
        {
            solve_stat.stat_50000_60000ns++;
        }
        else if (time_used_solve >= 60000 && time_used_solve < 70000)
        {
            solve_stat.stat_60000_70000ns++;
        }
        else if (time_used_solve >= 70000 && time_used_solve < 80000)
        {
            solve_stat.stat_70000_80000ns++;
        }
        else if (time_used_solve >= 80000 && time_used_solve < 90000)
        {
            solve_stat.stat_80000_90000ns++;
        }
        else if (time_used_solve >= 90000 && time_used_solve < 100000)
        {
            solve_stat.stat_90000_100000ns++;
        }
        else
        {
            solve_stat.stat_over_100000ns++;
        }

        rsn_pause(1000);
    }
    

   /* printf("[Total Stat]\n (%d, %d)  min: %luus, max: %luus, avg: %luus\n", g_app_ctx.matrix_size, g_app_ctx.matrix_size, min_time, max_time, total_time / g_app_ctx.repeat);
    printf("0  ~ 10us:  %lu\n", total_stat.stat_0_10us);
    printf("10 ~ 20us:  %lu\n", total_stat.stat_10_20us);
    printf("20 ~ 30us:  %lu\n", total_stat.stat_20_30us);
    printf("30 ~ 40us:  %lu\n", total_stat.stat_30_40us);
    printf("40 ~ 50us:  %lu\n", total_stat.stat_40_50us);
    printf("50 ~ 60us:  %lu\n", total_stat.stat_50_60us);
    printf("60 ~ 70us:  %lu\n", total_stat.stat_60_70us);
    printf("70 ~ 80us:  %lu\n", total_stat.stat_70_80us);
    printf("80 ~ 90us:  %lu\n", total_stat.stat_80_90us);
    printf("90 ~ 100us: %lu\n", total_stat.stat_90_100us);
    printf("   > 100us: %lu\n", total_stat.stat_over_100us);
    printf("\n\n");
   */

    printf("[Solve Stat]\n (%d, %d)  min: %luns, max: %luns, avg: %luns\n", g_app_ctx.matrix_size, g_app_ctx.matrix_size, solve_min_time, solve_max_time, total_time_solve / g_app_ctx.repeat);
    printf("0  ~ 10000ns:  %lu\n", solve_stat.stat_0_10000ns);
    printf("10000 ~ 20000ns:  %lu\n", solve_stat.stat_10000_20000ns);
    printf("20000 ~ 30000ns:  %lu\n", solve_stat.stat_20000_30000ns);
    printf("30000 ~ 40000ns:  %lu\n", solve_stat.stat_30000_40000ns);
    printf("40000 ~ 50000ns:  %lu\n", solve_stat.stat_40000_50000ns);
    printf("50000 ~ 60000ns:  %lu\n", solve_stat.stat_50000_60000ns);
    printf("60000 ~ 70000ns:  %lu\n", solve_stat.stat_60000_70000ns);
    printf("70000 ~ 80000ns:  %lu\n", solve_stat.stat_70000_80000ns);
    printf("80000 ~ 90000ns:  %lu\n", solve_stat.stat_80000_90000ns);
    printf("90000 ~ 100000ns: %lu\n", solve_stat.stat_90000_100000ns);
    printf("   > 100000ns: %lu\n", solve_stat.stat_over_100000ns);
    printf("\n\n");

    return;
}

// gcc -O2 ./KLU-test.c -L ../SuiteSparse-install-dir/static-lib// -lklu -lbtf -lamd -lcolamd -lsuitesparseconfig -lm -lpthread -I ../SuiteSparse-install-dir/include/
int main(int argc, char *argv[])
{

    parse_program_arg(argc, argv);

    bind_cpu_init();
    sched_init();
    rsn_pause(300000000);
	init_context();
    init_matrix(context);
    solve();
}


