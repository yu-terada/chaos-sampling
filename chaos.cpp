#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <time.h>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/Eigen"
#include "MT.h"
#define N_res 100
#define N_ex 100
#define N_a 5
#define N_v 5
#define T_END 11e8
#define J0 0.0
#define dim_noise N_res
#define dim_noise2 N_ring
#define N_ring 5
#define transient_period 10
#define receptive_period (int)(190+transient_period)
#define error_history_period (int)(50*receptive_period)//150
#define record_period (int)(T_END/100)
using Eigen::MatrixXd;
using namespace Eigen;

enum CLARG {
  CLARG_CMD,
  CLARG_g,
  CLARG_D,
  CLARG_FLAG,
  CLARG_SEED,
  CLARG_OUTPUT_FILE_ERROR,
  CLARG_OUTPUT_FILE_COUPLING,
  CLARG_OUTPUT_FILE_DISTRIBUTION,
  CLARG_OUTPUT_FILE_CONDITION,
  NUM_OF_CLARGs
};
enum MSG {
  MSG_USAGE,
  NUM_OF_MSGs
};
char msg[NUM_OF_MSGs][256] = {
"\
Usage: %s\
 <g>\
 <D>\
 <flag>\
 <seed>\
 <outputfile_error>\
 <outputfile_coupling>\
 <outputfile_distribution>\
<outputfile_condition>\
\n"
};
typedef struct {
  double g;
  double D;
} param_t;
double gaussian_random();
int distance(int,int);
double true_likelihood_auditory(int,int);
double true_likelihood_visual(int,int);
void set_true_posterior_theta_given_patterns(int,int,double *);
MatrixXd dec_to_bin(int);
int tensor_to_scholar(int,int,int,int,int);
int vector_to_decimal(MatrixXd x,int num_pattern);
double difference_bet_probablities(MatrixXd x_a, MatrixXd x_v,double *true_posterior_theta,double *output_probablity,int num_pattern_a,int num_pattern_v);
void set_true_posterior_theta_x_a(int num_pattern_a,double *posterior_theta_a);
void set_true_posterior_theta_x_v(int num_pattern_v,double *posterior_theta_v);
double measuring_integration_a(double *true_posterior_theta,double *true_posterior_theta_a,int num_pattern_a,int num_pattern_v);
double measuring_integration_v(double *true_posterior_theta,double *true_posterior_theta_v,int num_pattern_a,int num_pattern_v);

FILE *sfopen(char *filename, char *openmode) {
  FILE *val;
  val = fopen(filename, openmode);
  if (val == NULL) {
    printf("The file cannot open.\n");
    exit(EXIT_FAILURE);
  }
  return val;
}
int main(int argc, char **argv){
  MatrixXd x_a(N_a,1),x_v(N_v,1),x(N_res,1),x_noise(N_res,1),x_nonoise(N_res,1),tmp_field(N_res,1),x_external(N_ex,1),tmp_field_external(N_ex,1),noise(N_res,1),resource(dim_noise,1),resource2(dim_noise2,1),noise_pert(N_ring,1),J(N_res,N_res),bias(N_res,1), W_a(N_res,N_a),W_v(N_res,N_v),W_readout(N_ring,N_res),bias_readout(N_ring,1),W_pert(N_res,dim_noise),W_pert2(N_ring,dim_noise2),J_external(N_ex,N_ex),W_external(N_res,N_ex),W_external_pert(N_ring,N_ex),W_feedback1(N_res,N_res),W_feedback2(N_ring,N_res),output_vector(N_ring,1);
  MatrixXf::Index row,col;
  std::complex<double> I(0,1);
  long t;
  double avg[2];
  param_t param;
  int state_history_offset=0;
  int i,j,k,t_past;
  int y_nonoise,y_noise;
  double b;
  FILE *fout_error,*fout_coupling,*fout_distribution,*fout_condition;
  int num_pattern_a=pow(2,N_a);
  int num_pattern_v=pow(2,N_v);
  double true_posterior_theta[N_ring*num_pattern_a*num_pattern_v];
  double true_posterior_theta_x_a[N_ring*num_pattern_a];
  double true_posterior_theta_x_v[N_ring*num_pattern_v];
  double output_probablity[N_ring],output_probablity_noise[N_ring];
  double prob_diff_noise,prob_diff_nonoise;
  double Error=0.0;
  double Error_base=0.0;
  int position=0;
  MatrixXd noise_history_for_learning(error_history_period,N_res),tanh_nonoise_history_for_learning(error_history_period,N_res),resource_history_for_learning(error_history_period,dim_noise),noise_pert_history_for_learning(error_history_period,N_ring);
  MatrixXd delta_J(N_res,N_res),delta_bias(N_res,1),delta_Wpert(N_res,dim_noise),delta_Wreadout(N_ring,N_res),delta_bias_readout(N_ring,1);
  double Error_mean=0.0;
  MatrixXd m_adam1(N_res,N_res),v_adam1(N_res,N_res),mhat_adam1(N_res,N_res),vhat_adam1(N_res,N_res),g_adam1(N_res,N_res);// adam for J
  MatrixXd m_adam2(N_res,1),v_adam2(N_res,1),mhat_adam2(N_res,1),vhat_adam2(N_res,1),g_adam2(N_res,1);// for bias
  MatrixXd m_adam3(N_res,dim_noise),v_adam3(N_res,dim_noise),mhat_adam3(N_res,dim_noise),vhat_adam3(N_res,dim_noise),g_adam3(N_res,dim_noise);// for Wp
  MatrixXd m_adam4(N_ring,N_res),v_adam4(N_ring,N_res),mhat_adam4(N_ring,N_res),vhat_adam4(N_ring,N_res),g_adam4(N_ring,N_res);// for Wreadout
  MatrixXd m_adam5(N_ring,1),v_adam5(N_ring,1),mhat_adam5(N_ring,1),vhat_adam5(N_ring,1),g_adam5(N_ring,1);// for bias_readout
  double ratio1=1.0;
  double ratio2=1.0;
  double error_record=0.0;
  VectorXd mean_noise1(N_res,1),mean_noise2(N_ring,1),dev_noise1(N_res,1),dev_noise2(N_ring,1);
  int flag;
  MatrixXd tmp_noise_field1(N_res,1),tmp_noise_field2(N_ring,1);
  int rnd_seed;
  double D;
  rnd_seed = atof(argv[CLARG_SEED]);
  
  init_genrand(17225+11*rnd_seed);
  if(argc!=NUM_OF_CLARGs){
    printf(msg[MSG_USAGE], argv[0]);
    return 0;
  }
  param.g = atof(argv[CLARG_g]);
  param.D = atof(argv[CLARG_D]);
  flag = atof(argv[CLARG_FLAG]);
  fout_error = sfopen(argv[CLARG_OUTPUT_FILE_ERROR],"w");
  fprintf(fout_error, "\n");
  set_true_posterior_theta_given_patterns(num_pattern_a,num_pattern_v,true_posterior_theta);
  set_true_posterior_theta_x_a(num_pattern_a,true_posterior_theta_x_a);
  set_true_posterior_theta_x_v(num_pattern_v,true_posterior_theta_x_v);
  for(i=0;i<N_res;i++){
    bias(i,0) = 0.0;
    if(i<N_ring)
      bias_readout(i,0) = 0.0;
    for(j=0;j<N_res;j++){
      if(i == j){
	J(i,j) = 0.0;
      }else{
	J(i,j) = J0 + param.g*gaussian_random()/sqrt((double)N_res);
      }
    }
  }
  for(i=0;i<N_a;i++){
    for(j=0;j<N_res;j++)
      W_a(j,i) = gaussian_random();
    if( flag == 6 || flag == 7 ){
      for(j=0;j<N_res;j++)
	W_a(j,i) = 0.0;
    }
  }
   for(i=0;i<N_v;i++){
     for(j=0;j<N_res;j++)
       W_v(j,i) = gaussian_random();
     if( flag == 5 || flag == 7 ){
       for(j=0;j<N_res;j++)
	 W_v(j,i) = 0.0;
     }
   }
   for(i=0;i<N_ring;i++){
     bias_readout(i,0) = 0.0;
     for(j=0;j<N_res;j++)
       W_readout(i,j) = gaussian_random()/sqrt((double)N_res);
   }
   for(i=0;i<N_res;i++)
     x(i,0) = gaussian_random();
   x_nonoise = x;

   fout_coupling = sfopen(argv[CLARG_OUTPUT_FILE_COUPLING],"w");
   fprintf(fout_coupling, "\n");
   for(i=0;i<N_res;i++){
     for(j=0;j<N_res;j++)
       fprintf(fout_coupling, "%d %d %f %f\n",i+1,j+1,J(i,j),bias(i,0));
   }
   fout_condition = sfopen(argv[CLARG_OUTPUT_FILE_CONDITION],"w");
   fprintf(fout_condition, "\n");
   for(i=0;i<N_ring;i++){
     for(j=0;j<N_res;j++)
       fprintf(fout_condition, "%d %d %f %f\n",i+1,j+1,W_readout(i,j),bias_readout(i,0));
   }

   // Initialization of output probabilities
   for(i=0;i<N_ring;i++){
     output_probablity[i] = 0.0;
     output_probablity_noise[i] = 0.0;	
   }
   // Start of the time evolution
   for( t=1 ; t<=T_END ; t++ ){
     for(i=0;i<N_res;i++)
       noise(i,0) = 2.0*param.D*(genrand_real3()-0.5); 
     for(i=0;i<N_ring;i++)
       noise_pert(i,0) = 2.0*param.D*(genrand_real3()-0.5);
     
     // Time evoluation of the network
     tmp_field = J*x_nonoise + bias + W_a*x_a + W_v*x_v;
     x_nonoise  = tmp_field.array().tanh().matrix();//r_t
     
     if( (t%receptive_period) == 0 || (t%receptive_period) > transient_period ){
       output_vector = W_readout*x_nonoise+bias_readout;//z_t
       b = output_vector.maxCoeff(&row,&col);
       y_nonoise = row;
       output_probablity[y_nonoise] += 1.0/((double)receptive_period-(double)transient_period);
       tmp_field += noise;
       x_noise = tmp_field.array().tanh().matrix();
       output_vector = W_readout*x_noise+bias_readout+noise_pert;
       b = output_vector.maxCoeff(&row,&col);
       y_noise = row;
       output_probablity_noise[y_noise] += 1.0/((double)receptive_period-(double)transient_period);
     }

     for(i=0;i<N_res;i++){
       noise_history_for_learning(((t-1)%error_history_period),i) = noise(i,0);//xi_t
       tanh_nonoise_history_for_learning(((t-1)%error_history_period),i) = x_nonoise(i,0);//r_t
     }
     for(i=0;i<N_ring;i++)
       noise_pert_history_for_learning(((t-1)%error_history_period),i) = noise_pert(i,0);//zeta_t
     for(i=0;i<dim_noise;i++)
       resource_history_for_learning(((t-1)%error_history_period),i) = resource(i,0);
    
     if((t % receptive_period) == 0){
       prob_diff_nonoise = difference_bet_probablities(x_a,x_v,true_posterior_theta,output_probablity,num_pattern_a,num_pattern_v);
       prob_diff_noise = difference_bet_probablities(x_a,x_v,true_posterior_theta,output_probablity_noise,num_pattern_a,num_pattern_v);
      
       Error +=  prob_diff_noise/((double)error_history_period/(double)receptive_period);
       Error_base +=  prob_diff_nonoise/((double)error_history_period/(double)receptive_period);
       //update of prey position
       position = (int)(5.0*genrand_real3());
       //set the states of auditory and visual neurons
       for(i=0;i<N_a;i++){
	 if(true_likelihood_auditory(position,i) > genrand_real3()){
	   x_a(i,0) = 1.0;
	 }else{
	   x_a(i,0) = 0.0;
	 }
       }
       for(i=0;i<N_v;i++){
	 if(true_likelihood_visual(position,i) > genrand_real3()){
	   x_v(i,0) = 1.0;
	 }else{
	   x_v(i,0) = 0.0;
	 }
       }
       // initialization of model probablities
       for(i=0;i<N_ring;i++){
	 output_probablity[i] = 0.0;
	 output_probablity_noise[i] = 0.0;	
       }
     }
    
     // learning process of internal connections and thresholds
     if((t % error_history_period) == 0){					   
       for(i=0;i<N_res;i++){
	 delta_bias(i,0) = 0.0;
	 if( i < N_ring )
	   delta_bias_readout(i,0) = 0.0;
	 for(j=0;j<N_res;j++){
	   delta_J(i,j) = 0.0;
	   if (j < dim_noise)
	     delta_Wpert(i,j) = 0.0;
	   if(i < N_ring)
	     delta_Wreadout(i,j) = 0.0;
	 }
       }
       for(t_past=transient_period;t_past<error_history_period;t_past++){
	 for(i=0;i<N_res;i++){
	   for(j=0;j<N_res;j++){
	     if(i == j){
	       delta_J(i,j) = 0.0;
	     }else{
	       if(t_past<error_history_period-1)
		 delta_J(i,j) -= (Error-Error_base)*noise_history_for_learning(t_past+1,i)*tanh_nonoise_history_for_learning(t_past,j);
	       if( flag == 3 )
		 delta_J(i,j) = 0.0;
	     }
	     if( i < N_ring ){
	       delta_Wreadout(i,j) -= (Error-Error_base)*noise_pert_history_for_learning(t_past,i)*tanh_nonoise_history_for_learning(t_past,j);
	     }	    
	   }
	   delta_bias(i,0) -= (Error-Error_base)*noise_history_for_learning(t_past,i);
	   if( i < N_ring )
	     delta_bias_readout(i,0) -= (Error-Error_base)*noise_pert_history_for_learning(t_past,i);
	   if( flag == 3 )
	     delta_bias(i,0) = 0.0;
	 }
       }
       ratio1 *= .9;
       ratio2 *= .999;
       for(i=0;i<N_res;i++){
	 for(j=0;j<N_res;j++){
	   g_adam1(i,j) = - delta_J(i,j);
	   m_adam1(i,j) = 0.9*m_adam1(i,j)+(1.0-0.9)*g_adam1(i,j);
	   v_adam1(i,j) = 0.999*v_adam1(i,j)+(1.0-0.999)*g_adam1(i,j)*g_adam1(i,j);
	   mhat_adam1(i,j) = m_adam1(i,j)/(1.0-ratio1);
	   vhat_adam1(i,j) = v_adam1(i,j)/(1.0-ratio2);
	   J(i,j) -= .001*mhat_adam1(i,j)/(sqrt(vhat_adam1(i,j))+1.0e-8);
	   if(J(i,j)==0.0 && i!=j)
	     printf("%ld %d %d\n",t,i,j);
	   if( i < N_ring ){
	     g_adam3(i,j) = - delta_Wreadout(i,j);
	     m_adam3(i,j) = 0.9*m_adam3(i,j)+(1.0-0.9)*g_adam3(i,j);
	     v_adam3(i,j) = 0.999*v_adam3(i,j)+(1.0-0.999)*g_adam3(i,j)*g_adam3(i,j);
	     mhat_adam3(i,j) = m_adam3(i,j)/(1.0-ratio1);
	     vhat_adam3(i,j) = v_adam3(i,j)/(1.0-ratio2);
	     W_readout(i,j) -= .001*mhat_adam3(i,j)/(sqrt(vhat_adam3(i,j))+1.0e-8);   
	   }
	 }
	 g_adam2(i,0) = - delta_bias(i,0);
	 m_adam2(i,0) = 0.9*m_adam2(i,0)+(1.0-0.9)*g_adam2(i,0);
	 v_adam2(i,0) = 0.999*v_adam2(i,0)+(1.0-0.999)*g_adam2(i,0)*g_adam2(i,0);
	 mhat_adam2(i,0) = m_adam2(i,0)/(1.0-ratio1);
	 vhat_adam2(i,0) = v_adam2(i,0)/(1.0-ratio2);
	 //	 bias(i,0) -= .001*mhat_adam2(i,0)/(sqrt(vhat_adam2(i,0))+1.0e-8);
	 if(isnan(bias(i,0)==1))
	   printf("%ld %d\n",t,i);
	 if( i < N_ring ){
	   g_adam5(i,0) = - delta_bias_readout(i,0);
	   m_adam5(i,0) = 0.9*m_adam5(i,0)+(1.0-0.9)*g_adam5(i,0);
	   v_adam5(i,0) = 0.999*v_adam5(i,0)+(1.0-0.999)*g_adam5(i,0)*g_adam5(i,0);
	   mhat_adam5(i,0) = m_adam5(i,0)/(1.0-ratio1);
	   vhat_adam5(i,0) = v_adam5(i,0)/(1.0-ratio2);
	   bias_readout(i,0) -= .001*mhat_adam5(i,0)/(sqrt(vhat_adam5(i,0))+1.0e-8);
	 }
       }
       error_record += Error_base/((((double)record_period)/((double)error_history_period)));
       if( (t%record_period) == 0 ){
	 printf("t=%ld Error=%f",t,error_record);
	 printf(" J_max=%f=J(%ld,%ld), J_min=%f=J(%ld,%ld), h_max=%f=h(%ld), h_min=%f=h(%ld), x_max=%f=x(%ld), x_min=%f=x(%ld)\n",J.maxCoeff(&row,&col),row,col,J.minCoeff(&row,&col),row,col,bias.maxCoeff(&row,&col),row,bias.minCoeff(&row,&col),row,x_nonoise.maxCoeff(&row,&col),row,x_nonoise.minCoeff(&row,&col),row);
	 fprintf(fout_error,"%ld %f\n",t,error_record);
	 error_record = 0.0;
       }
       Error = 0.0;
       Error_base = 0.0;
     }
   }
   fclose(fout_error);
   for(i=0;i<N_res;i++){
     for(j=0;j<N_res;j++)
       fprintf(fout_coupling, "%d %d %f %f\n",i+1,j+1,J(i,j),bias(i,0));
   }
   fclose(fout_coupling);
   

   //test run
   position = (int)(5.0*genrand_real3());
   //set the states of sensory neurons
   for(i=0;i<N_a;i++){
     if(true_likelihood_auditory(position,i) > genrand_real3()){
       x_a(i,0) = 1.0;
     }else{
       x_a(i,0) = 0.0;
     }
   }
   for(i=0;i<N_v;i++){
     if(true_likelihood_visual(position,i) > genrand_real3()){
       x_v(i,0) = 1.0;
     }else{
       x_v(i,0) = 0.0;
     }
   }
  
   // initialization of output probablity
   for(i=0;i<N_ring;i++)
     output_probablity[i] = 0.0;
 
   for(t=1;t<=receptive_period;t++){
     tmp_field = J*x_nonoise + bias + W_a*x_a + W_v*x_v;
     x_nonoise = tmp_field.array().tanh().matrix();
     output_vector = W_readout*x_nonoise+bias_readout;
     b = output_vector.maxCoeff(&row,&col);
     y_nonoise = row;

     if(  t > transient_period )
       output_probablity[y_nonoise] += 1.0/((double)receptive_period-(double)transient_period);
   }

   fout_distribution = sfopen(argv[CLARG_OUTPUT_FILE_DISTRIBUTION],"w");
   fprintf(fout_distribution, "x_a = ");
   for(i=0;i<N_a;i++)
     fprintf(fout_distribution, "%f ",x_a(i,0));
   fprintf(fout_distribution, "\n");
   fprintf(fout_distribution, "x_v = ");
   for(i=0;i<N_v;i++)
     fprintf(fout_distribution, "%f ",x_v(i,0));
   fprintf(fout_distribution, "\n");
   for(i=0;i<N_ring;i++){
     fprintf(fout_distribution, "%d %f %f\n",i+1,output_probablity[i],true_posterior_theta[tensor_to_scholar(i,vector_to_decimal(x_a,num_pattern_a),vector_to_decimal(x_v,num_pattern_v),num_pattern_a,num_pattern_v)]);
   }
   fclose(fout_distribution);

   fprintf(fout_condition, "\n");
   for(i=0;i<N_res;i++){
     for(j=0;j<N_a;j++)
       fprintf(fout_condition, "%d %d %f\n",i+1,j+1,W_a(i,j));
   }
   for(i=0;i<N_res;i++){
     for(j=0;j<N_v;j++)
       fprintf(fout_condition, "%d %d %f\n",i+1,j+1,W_v(i,j));
   }
   for(i=0;i<N_ring;i++){
     for(j=0;j<N_res;j++)
       fprintf(fout_condition, "%d %d %f %f\n",i+1,j+1,W_readout(i,j),bias_readout(i,0));
   }
   fclose(fout_condition);

   return 0; 
}
double gaussian_random(){
  double gauss_ran;
  gauss_ran = sqrt( -2.0*log(genrand_real3()) ) * sin( 2.0*M_PI*genrand_real3() );
  return gauss_ran;
}
int distance(int i,int j){
  int d=0;

  if(i < j){
    if(j > i+((int)N_ring/2))
	  d = -(j-i-N_ring);
	else
	  d = -(j-i);
  }else
  if(i > j){
    if(j-i < -((int)N_ring/2))
      d = -(j-i+N_ring);
    else
      d = -(j-i);
  }
  return d;
}
double true_likelihood_auditory(int i,int j){
  int d;
  double likelihood;
  d = abs(distance(i,j));
  if (d == 0)
    likelihood = 0.7;  
  if( d == 1)
    likelihood = 0.5;
  if( d == 2)
    likelihood = 0.3;
  return likelihood;
}
double true_likelihood_visual(int i,int j){
  int d;
  double likelihood;
  d = abs(distance(i,j));
  if (d == 0)
    likelihood = 0.8;  
  if( d == 1)
    likelihood = 0.5;
  if( d == 2)
    likelihood = 0.2;
  return likelihood;
}
// normalization constant for each theta
void set_true_posterior_theta_given_patterns(int num_pattern_a,int num_pattern_v,double *posterior_theta_given_patterns){
  int i,j;
  MatrixXd x_a(N_a,1),x_v(N_v,1);
  int a_dec,v_dec;
  int a_dec_tmp,v_dec_tmp;
  int a_bin,v_bin;
  int base;
  int theta;
  double tmp = 1.0;
  MatrixXd normalization_constant(num_pattern_a,num_pattern_v);
  int test;
  
  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    x_a = dec_to_bin(a_dec);
    for(v_dec=0;v_dec<num_pattern_v;v_dec++){
      x_v = dec_to_bin(v_dec);

      normalization_constant(a_dec,v_dec) = 0.0;
      for(theta=0;theta<N_ring;theta++){
	tmp = 1.0;
	for(i=0;i<N_a;i++){
	  if( x_a(i)>0.5 ){
	    tmp *= true_likelihood_auditory(i,theta);
	  }else{
	    tmp *= 1.0-true_likelihood_auditory(i,theta);
	  }
	}
	for(i=0;i<N_v;i++){
	  if( x_v(i)>0.5 ){
	    tmp *= true_likelihood_visual(i,theta);
	  }else{
	    tmp *= 1.0-true_likelihood_visual(i,theta);
	  }
	}
	normalization_constant(a_dec,v_dec) += tmp;
      }
    }
  }
  
  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    x_a = dec_to_bin(a_dec);    
    for(v_dec=0;v_dec<num_pattern_v;v_dec++){
      x_v = dec_to_bin(v_dec);
      for(theta=0;theta<N_ring;theta++){
	tmp = 1.0;
	for(i=0;i<N_a;i++){
	  if( x_a(i)>0.5 ){
	    tmp *= true_likelihood_auditory(i,theta);
	  }else{
	    tmp *= 1.0-true_likelihood_auditory(i,theta);
	  }
	}  
	for(i=0;i<N_v;i++){
	  if( x_v(i)>0.5 ){
	    tmp *= true_likelihood_visual(i,theta);
	  }else{
	    tmp *= 1.0-true_likelihood_visual(i,theta);
	  }
	}
	posterior_theta_given_patterns[tensor_to_scholar(theta,a_dec,v_dec,num_pattern_a,num_pattern_v)] = tmp / normalization_constant(a_dec,v_dec);
      }
    }
  }
  
  return;
}
// transform decimal index to binary vector
MatrixXd dec_to_bin(int dec){
  int i,base,bin,dec_tmp;
  MatrixXd x(5,1);
  
  base = 1;
  bin = 0;
  dec_tmp = dec;
  while( dec_tmp > 0 ){
    bin += (dec_tmp % 2)*base;
    dec_tmp /= 2;
    base *=10;
  }
  for(i=0;i<N_ring;i++)
    x(i,0) = (int)(((int)(bin/pow(10,4-i))) % 10);
  
  return x;
}
// transform 3 indices to scholar value
int tensor_to_scholar(int theta, int a_dec,int v_dec,int num_pattern_a,int num_pattern_v){
  int index;

  index = (theta*(num_pattern_a*num_pattern_v) + a_dec*num_pattern_v + v_dec);
  
  return index;
}
// transform vector to decimal
int vector_to_decimal(MatrixXd x,int num_pattern){
  int num,i,index;

  index=0;
  num = x.rows();//N_a or N_v
  for(i=0;i<num;i++)
    index += (int)(pow(2,num-1-i)*x(i,0));
  
  return index;
}
double difference_bet_probablities(MatrixXd x_a, MatrixXd x_v,double *true_posterior_theta,double *output_probablity,int num_pattern_a,int num_pattern_v){
  double ans=0.0;
  int theta,a_dec,v_dec,tmp_index;
  
  a_dec = vector_to_decimal(x_a,num_pattern_a);
  v_dec = vector_to_decimal(x_v,num_pattern_v);
  
  for(theta=0;theta<N_ring;theta++){
    tmp_index = theta*(num_pattern_a*num_pattern_v) + a_dec*num_pattern_v + v_dec;
    ans += (sqrt(true_posterior_theta[tmp_index]) - sqrt(output_probablity[theta]))*(sqrt(true_posterior_theta[tmp_index]) - sqrt(output_probablity[theta]));
  }
  ans = ans/2.0;
  return ans;
}
void set_true_posterior_theta_x_a(int num_pattern_a,double *posterior_theta_a){
  int a_dec,theta,i,tmp_index;
  double tmp = 1.0;
  MatrixXd x_a(N_a,1);
  MatrixXd normalization_constant(num_pattern_a,1);

  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    x_a = dec_to_bin(a_dec);
    normalization_constant(a_dec,0)= 0.0;
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_a;i++){
	if( x_a(i)>0.5 ){
	  tmp *= true_likelihood_auditory(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_auditory(i,theta);
	}
      }
      normalization_constant(a_dec,0) += tmp;
    }
  }
  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    x_a = dec_to_bin(a_dec);
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_a;i++){
	if( x_a(i)>0.5 ){
	  tmp *= true_likelihood_auditory(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_auditory(i,theta);
	}
      }
      tmp_index = theta*(num_pattern_a) + a_dec;
      posterior_theta_a[tmp_index] = tmp / normalization_constant(a_dec,0);
    }
  }
  return;
}
void set_true_posterior_theta_x_v(int num_pattern_v,double *posterior_theta_v){
  int v_dec,theta,i,tmp_index;
  double tmp = 1.0;
  MatrixXd x_v(N_v,1);
  MatrixXd normalization_constant(num_pattern_v,1);

  for(v_dec=0;v_dec<num_pattern_v;v_dec++){
    x_v = dec_to_bin(v_dec);
    normalization_constant(v_dec,0)= 0.0;
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_v;i++){
	if( x_v(i)>0.5 ){
	  tmp *= true_likelihood_visual(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_visual(i,theta);
	}
      }
      normalization_constant(v_dec,0) += tmp;
    }
  }
  for(v_dec=0;v_dec<num_pattern_v;v_dec++){
    x_v = dec_to_bin(v_dec);
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_v;i++){
	if( x_v(i)>0.5 ){
	  tmp *= true_likelihood_visual(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_visual(i,theta);
	}
      }
      tmp_index = theta*(num_pattern_v) + v_dec;
      posterior_theta_v[tmp_index] = tmp / normalization_constant(v_dec,0);
    }
  }
  return;
}
double measuring_integration_a(double *true_posterior_theta,double *true_posterior_theta_a,int num_pattern_a,int num_pattern_v){
  double ans=0.0;
  double h=0.0;
  int theta,a_dec,v_dec,tmp_index1,tmp_index2;
  MatrixXd p_a(num_pattern_a,1),p_v(num_pattern_v,1);
  MatrixXd x_a(N_a,1),x_v(N_v,1);
  double tmp = 1.0;
  int i;

  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    x_a = dec_to_bin(a_dec);
    p_a(a_dec,0) = 0.0;
    
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_a;i++){
	if( x_a(i)>0.5 ){
	  tmp *= true_likelihood_auditory(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_auditory(i,theta);
	}
      }
      p_a(a_dec,0) += tmp/((double)N_ring);
    }
  }
  for(v_dec=0;v_dec<num_pattern_v;v_dec++){
    x_v = dec_to_bin(v_dec);
    p_v(v_dec,0) = 0.0;
    
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_v;i++){
	if( x_v(i)>0.5 ){
	  tmp *= true_likelihood_visual(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_visual(i,theta);
	}
      }
      p_v(v_dec,0) += tmp/((double)N_ring);
    }
  }
  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    for(v_dec=0;v_dec<num_pattern_v;v_dec++){
      h=0.0;
      for(theta=0;theta<N_ring;theta++){
	tmp_index1 = tensor_to_scholar(theta,a_dec,v_dec,num_pattern_a,num_pattern_v);
	tmp_index2 = theta*(num_pattern_a) + a_dec;
	h += (sqrt(true_posterior_theta_a[tmp_index2]) - sqrt(true_posterior_theta[tmp_index1]))*(sqrt(true_posterior_theta_a[tmp_index2]) - sqrt(true_posterior_theta[tmp_index1]));
      }
      h = sqrt(h/2.0);
      ans += h*p_a(a_dec,0)*p_v(v_dec,0);
    }
  }
  return ans;
}
 double measuring_integration_v(double *true_posterior_theta,double *true_posterior_theta_v,int num_pattern_a,int num_pattern_v){
  double ans=0.0;
  double h=0.0;
  int theta,a_dec,v_dec,tmp_index1,tmp_index2;
  MatrixXd p_a(num_pattern_a,1),p_v(num_pattern_v,1);
  MatrixXd x_a(N_a,1),x_v(N_v,1);
  double tmp = 1.0;
  int i;

  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    x_a = dec_to_bin(a_dec);
    p_a(a_dec,0) = 0.0;
    
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_a;i++){
	if( x_a(i)>0.5 ){
	  tmp *= true_likelihood_auditory(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_auditory(i,theta);
	}
      }
      p_a(a_dec,0) += tmp/((double)N_ring);
    }
  }
  for(v_dec=0;v_dec<num_pattern_v;v_dec++){
    x_v = dec_to_bin(v_dec);
    p_v(v_dec,0) = 0.0;
    
    for(theta=0;theta<N_ring;theta++){
      tmp = 1.0;
      for(i=0;i<N_v;i++){
	if( x_v(i)>0.5 ){
	  tmp *= true_likelihood_visual(i,theta);
	}else{
	  tmp *= 1.0-true_likelihood_visual(i,theta);
	}
      }
      p_v(v_dec,0) += tmp/((double)N_ring);
    }
  }
 
  for(a_dec=0;a_dec<num_pattern_a;a_dec++){
    for(v_dec=0;v_dec<num_pattern_v;v_dec++){
      h=0.0;
      for(theta=0;theta<N_ring;theta++){
	tmp_index1 = tensor_to_scholar(theta,a_dec,v_dec,num_pattern_a,num_pattern_v);
	tmp_index2 = theta*(num_pattern_a) + v_dec;
	h += (sqrt(true_posterior_theta_v[tmp_index2]) - sqrt(true_posterior_theta[tmp_index1]))*(sqrt(true_posterior_theta_v[tmp_index2]) - sqrt(true_posterior_theta[tmp_index1]));
      }
      h = sqrt(h/2.0);
      ans += h*p_a(a_dec,0)*p_v(v_dec,0);
    }
  }
  return ans;
}
