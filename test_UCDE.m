clear;
Findex=[1,3:30];
para.dim=30;
para.Xmin=-100;
para.Xmax=100;
para.pop_size=150;
para.maxfe=10000*para.dim;
repeat=11;

hidden_num=100;

pre_net=load('UCDE_net\UCDE_net_30D');
pre_net = pre_net.pre_net;

tic
for ITER =1:29
    
num=Findex(ITER);

for i =1: repeat
   [results(i),~] = UCDE( num, pre_net ,para);
end

disp(mean(results)-num*100);
end
toc