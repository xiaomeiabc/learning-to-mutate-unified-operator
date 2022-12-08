clear;
Findex=[1,3:30];
para.dim=30;
para.Xmin=-100;
para.Xmax=100;
para.pop_size=100;
para.maxfe=10000*para.dim;
repeat=11;

hidden_num=100;

pre_net=load('UJADE_net\UJADE_net_30D');
pre_net = pre_net.pre_net;

tic
for iter =1:9

num=Findex(iter);

for i =1: repeat
   [results(i)] = UJADE(num, pre_net ,para,hidden_num);
end

disp(mean(results)-num*100);
end
toc