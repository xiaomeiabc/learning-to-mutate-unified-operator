function [ bsf_fitness ,po] = ULSHADE( problem_index,net_w1,net_w2,net_w3,para,hidden_num )

    p_best_rate = para.p_best_rate;
    pop_size = para.pop_size;
    arc_size = round(pop_size*para.arc_rate);
    nfes = 0;
    
    lu = [para.Xmin*ones(1,para.dim); para.Xmax*ones(1,para.dim)];
    pop = repmat(lu(1, :), pop_size, 1) + rand(pop_size, para.dim) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
    

    fitness = cec17_func( pop' , problem_index);
    nfes=nfes+pop_size;
    [bsf_fitness,bsf_index]=min(fitness);
    
 %%%%%
    archive =[];
    memory_omega = 0.5*ones(para.memo_size, 1);
    memory_gamma = 0.5*ones(para.memo_size, 1);
    memory_cr = 0.5*ones(para.memo_size, 1); 
    
    memory_pos = 1;
    

    pop_cr=[];pop_omega=[]; pop_gamma=[];
    p_num = round(pop_size*p_best_rate);
    
    max_pop_size = pop_size;
    min_pop_size = 4;
    arc_ind_count = 0;
    iter = 0;
    
    %%%%%%%
    while nfes<para.maxfe 
        iter=iter+1;


        success_omega = [] ; % S_omega
        success_gamma = [] ; % S_gamma
        success_cr = []  ;% S_CR
        dif_fitness = []  ;% ¡÷fk
        [~,sorted_array] = sort(fitness);%
        
        [a,b]=size(pop);
        child = ones(a,b);
        
        for target=1:pop_size
            random_selected_period = unidrnd(para.memo_size);

            mu_omega = memory_omega(random_selected_period);
            mu_gamma = memory_gamma(random_selected_period);
            mu_cr = memory_cr(random_selected_period);
            
            if mu_cr == -1
                pop_cr(target) = 0;
            else
                pop_cr(target) = randn(1,1)*0.1+mu_cr;
                if pop_cr(target) > 1
                    pop_cr(target) = 1;
                elseif pop_cr(target) < 0
                    pop_cr(target)= 0;
                end
            end

            pop_omega(target) = cauchy_g(mu_omega, 0.1);
            while pop_omega(target) <= 0
                pop_omega(target) = cauchy_g(mu_omega, 0.1);  % until pop_omega[target]>0
            end
            if pop_omega(target) > 1
               pop_omega(target) = 1.0;
            end
            
            pop_gamma(target) = cauchy_g(mu_gamma, 0.1);
            while pop_gamma(target) <= 0
                pop_gamma(target) = cauchy_g(mu_gamma, 0.1);  %until pop_gamma[target]>0
            end
            if pop_gamma(target) > 1
               pop_gamma(target) = 1.0;
            end
            
            p_best_ind = sorted_array(unidrnd(p_num));
            %operate current-to-pbest-1-bin-withArchive
            child(target,:) = current2pbest(pop, target, p_best_ind, pop_omega(target),pop_gamma(target), pop_cr(target) ,archive, arc_ind_count, pop_size,para);

        end
     
        fitness_child = cec17_func( child' , problem_index);
        for i=1:pop_size

            nfes = nfes+1;
            if (fitness_child(i)<bsf_fitness)
                bsf_fitness = fitness_child(i);
            end
            if nfes>=para.maxfe 
                break
            end
        end


        
        
        %%%%generation alternation
        for i=1:pop_size
            if fitness_child(i) == fitness(i)
                fitness(i) = fitness_child(i);
                pop(i,:) = child(i,:);
            elseif fitness_child(i) < fitness(i)
                dif_fitness=[dif_fitness,abs(fitness(i)-fitness_child(i))];
                %successful parameters are preserved in S_omega,S_gamma and S_CR
                fitness(i) = fitness_child(i);
                pop(i,:) = child(i,:);                
                success_omega=[success_omega,pop_omega(i)];
                success_gamma = [success_gamma,pop_gamma(i)]; 
                success_cr=[success_cr,pop_cr(i)];

                if arc_size > 1
                    if arc_ind_count < arc_size 
                        archive(arc_ind_count+1,:) = pop(i,:);
                        arc_ind_count = arc_ind_count+1;
                    else  %Whenever the size of the archive exceeds, randomly selected elements are deleted to make space for the newly inserted 
                        random_selected_arc_ind = unidrnd(arc_size);
                        archive(random_selected_arc_ind,:) = pop(i,:);
                    end
                end
                
            end
        end
        


        num_success_params = length(success_omega);
        
        %if numeber of successful parameters > 0, historical memories are updated 
        if num_success_params > 0
            
            
            memory_omega(memory_pos) = 0;
            memory_gamma(memory_pos) = 0;
            memory_cr(memory_pos) = 0;
            temp_sum_omega = 0;
            temp_sum_gamma = 0;
            temp_sum_cr = 0;
            sum = 0;
            sum2 = 0;
             RL_fitness=RL(dif_fitness,net_w3,hidden_num);
             RL_succ_omega = RL( success_omega,net_w1,hidden_num);
             RL_succ_gamma = RL(success_gamma,net_w2,hidden_num);
            for i=1:num_success_params
                sum=sum+RL_fitness(i);
                sum2 = sum2+dif_fitness(i);
            end
            
            for i=1:num_success_params
                weight = RL_fitness(i) / sum;          
                weight2 = dif_fitness(i)/sum2; 

                memory_omega(memory_pos) = memory_omega(memory_pos)+weight * RL_succ_omega(i) * RL_succ_omega(i);
                temp_sum_omega = temp_sum_omega+weight * RL_succ_omega(i);   

                memory_gamma(memory_pos) = memory_gamma(memory_pos)+weight * RL_succ_gamma(i) * RL_succ_gamma(i);
                temp_sum_gamma = temp_sum_gamma+weight * RL_succ_gamma(i);

                memory_cr(memory_pos) = memory_cr(memory_pos)+weight2 * success_cr(i) * success_cr(i);
                temp_sum_cr = temp_sum_cr+weight2 * success_cr(i);
            end
            memory_omega(memory_pos) = memory_omega(memory_pos)/temp_sum_omega;
            memory_gamma(memory_pos) = memory_gamma(memory_pos)/temp_sum_gamma;
            if temp_sum_cr == 0 || memory_cr(memory_pos) == -1
                memory_cr(memory_pos) = -1;
            else
                memory_cr(memory_pos) = memory_cr(memory_pos)/temp_sum_cr;
            end
            
            memory_pos = memory_pos+1;
            
            if memory_pos > para.memo_size 
                memory_pos = 1; 
            end
            clear success_sf
            clear success_cr
            clear dif_fitness
        end
        
        %calculate the population size in the next generation
        plan_pop_size = round((((min_pop_size - max_pop_size) / para.maxfe) * nfes) + max_pop_size);
        if pop_size > plan_pop_size
            reduction_ind_num = int8(pop_size - plan_pop_size); 
            if pop_size - reduction_ind_num < min_pop_size
                reduction_ind_num = int(pop_size - min_pop_size);
            end
            [pop, fitness, pop_size ]= reducePS(pop, fitness, reduction_ind_num, pop_size);
            arc_size = round(pop_size*para.arc_rate); %%%%round
            if arc_ind_count > arc_size
                arc_ind_count = arc_size;
            end

            p_num = round(pop_size*p_best_rate);
            if p_num <= 1
                p_num = 2;
            end
        end

       po(iter)=pop_size;
%        his_omega(:,iter) = memory_omega;
%        his_gamma(:,iter) = memory_gamma;
    end

end

function [child]=current2pbest(pop, target, p_best_individual, omega, gamma, cross_rate,archive, arc_ind_count, pop_size,para)
    child = ones(para.dim,1);
    W = max((1-omega)*gamma,1-gamma);
    W = max(W,omega*gamma);
    r1 = unidrnd(pop_size);
    r2 = unidrnd(pop_size);
    r3 = unidrnd(pop_size+arc_ind_count);
    while r1 == target
        r1 = unidrnd(pop_size);
    end
    while r2 == target || r2 == r1
        r2 = unidrnd(pop_size);
    end
    
    while r3 == target || r3 == r1 || r3 == r2
        r3 = unidrnd(pop_size+arc_ind_count);
    end
    
    random_variable =unidrnd(para.dim);
    
    %%%%mutation crossover
    if r3 > pop_size
        r3 = r3-pop_size;
        
        for i=1:para.dim
            if rand(1) < cross_rate || i == random_variable

                  child(i) = (omega * pop(target, i) + (1-omega) * pop(p_best_individual, i)) * gamma + (1-gamma) * pop(r1, i) + W * (pop(r2, i) - archive(r3, i));
            else
                child(i) = pop(target, i);
            end
        end
        
    else
        for i=1:para.dim
            if rand(1) < cross_rate || i == random_variable

                  child(i) = (omega * pop(target, i) + (1-omega) * pop(p_best_individual, i)) * gamma + (1-gamma) * pop(r1, i) + W * (pop(r2, i) - pop(r3, i));
            else
                child(i) = pop(target, i);
            end
        end    
    end
    child = modifySolutionWithParent(child , pop(target,:),para);
end

function [child] = modifySolutionWithParent(child, parent, para)
    for j=1:para.dim
        if child(j) < para.Xmin
            child(j)  = (para.Xmin + parent(j) )/2.0;
        elseif child(j)  > para.Xmax
            child(j)  = (para.Xmax + parent(j) )/2.0;
        end
    end
end

function [pop, fitness, pop_size] = reducePS(pop, fitness, reduction_ind_num, pop_size) 
    for i=1:reduction_ind_num
        worst_ind = 1;
        for j=1:pop_size
            if fitness(j) > fitness(worst_ind)
                worst_ind = j;
            end 
        end
        pop(worst_ind,:)=[];
        fitness(worst_ind)=[];
        pop_size = pop_size-1;
    end
end

function [sam]=cauchy_g(mu, gamma)
    sam=mu + gamma * tan(pi * (rand(1) - 0.5));
end

function [act] = RL(F,net_w,hidden_num)
            F=(F)/max(F);
            hidnet=cell(1,hidden_num);
            hid=cell(1,hidden_num);
%             output=cell(1,hidden_num);
            for j=1:hidden_num
                 hidnet{j}=F*net_w.layer{1}(j)+net_w.bias{1}(j);
                 hid{j}=(1+exp(-hidnet{j})).^(-1)-0.5;
            end
            net=net_w.bias{2};
            for j=1:hidden_num
                 net=net+hid{j}*net_w.layer{2}(j);
            end
            output=((1+exp(-net)).^(-1))*1;
            act=output;
end
