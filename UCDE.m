function [ fitnessbestX, fig_deas ] = UCDE( FunIndex, net_w ,para)
maxIteration=para.maxfe/para.pop_size-1;
Generation=0;
CR=0.9;   %crossover rate 
mutationStrategy=10;
crossStrategy=1;
omega = 1;
gamma = 0;

%%
%step1 
%X represent population
%Generation=0;
X=(para.Xmax-para.Xmin)*rand(para.pop_size,para.dim)+para.Xmin;

%%
%step2 mutation,crossover,selection

    for i=1:para.pop_size
        fitnessX(i)=cec17_func(X(i,:)',FunIndex);
    end
while Generation<maxIteration


    [fitnessbestX,indexbestX]=min(fitnessX);
    bestX=X(indexbestX,:);

%%
%step2.1 mutation

       V=mutation_UDE(X,bestX,mutationStrategy,omega,gamma);

 %%   
%step2.2 crossover

    U=crossover_UDE(X,V,CR,crossStrategy);
    %CEC TEST functions are bounded in [-100,100] 
    U=U-U.*(U>100)+100*(U>100);
    U=U-U.*(U<-100)-100*(U<-100);
%%    
%step2.3 selection
    for i=1:para.pop_size
        fitnessU(i)=cec17_func(U(i,:)',FunIndex);
    end

    for i=1:para.pop_size
        if fitnessU(i)<=fitnessX(i)
            X(i,:)=U(i,:);
            fitnessX(i)=fitnessU(i);
            if fitnessU(i)<fitnessbestX
                bestX=U(i,:);
                fitnessbestX=fitnessU(i);
            end
        end
    end
%%
    Generation=Generation+1;
    
    bestfitnessG(Generation)=fitnessbestX;
    std_X(Generation) = mean(std(X));
    std_fit (Generation)= std(log(fitnessX));

    [deas1,deas2]= RL2([std_X(Generation),std_fit(Generation),1-(Generation/maxIteration)], net_w);
    omega = deas1;
    gamma = 1-deas2;
    fig_deas(Generation,:) = [omega,gamma];
end


end


function [deas1,deas2] = RL2(std_X, net_w)
             hidnet=std_X*net_w.layer{1}+net_w.bias{1};
             hid=(1+exp(-hidnet)).^(-1)-0.5;
             net=hid*net_w.layer{2}+net_w.bias{2};
             deas=((1+exp(-net)).^(-1))*1;
             deas1 = deas(1);deas2 = deas(2);
end
