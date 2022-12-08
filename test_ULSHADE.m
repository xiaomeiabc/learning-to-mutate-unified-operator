clear; 
para.dim=50;
para.p_best_rate=0.11 ;
para.Xmin=-100;
para.Xmax=100;
para.pop_size=round(para.dim*18);
para.arc_rate= 2.6; 
para.memo_size=6;
para.maxfe=10000*para.dim;
para.OPTIMUM =0;
para.EPSILON=10^(-8);
repeat=5;
hidden_num=100; 
pre_net1=load('ULSHADE_net\ULSHADE_net_50D_1');
pre_net1 = pre_net1.pre_net1;
pre_net2=load('ULSHADE_net\ULSHADE_net_50D_2');
pre_net2 = pre_net2.pre_net2;
pre_net3=load('ULSHADE_net\ULSHADE_net_50D_3');
pre_net3 = pre_net3.pre_net3;

Findex=[1,3:30];
tic
for ITER =11:13

num=Findex(ITER);

for i =1: repeat
   [results(i,ITER)] = ULSHADE(num, pre_net1,pre_net2,pre_net3 ,para,hidden_num);

end

disp(mean(results(:,ITER))-num*100);
end
toc