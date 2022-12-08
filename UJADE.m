function [best_fvalue,best_solution,run_series,run_info] = UJADE(func_name,net_w,para,hidden_num )
%**************************************************************************************************
%Reference:  J. Zhang and A. C. Sanderson, "JADE: adaptive differential evolution
%                     with optional external archive," IEEE Trans. Evolut. Comput., vol. 13,
%                     no. 5, pp. 945-958, 2009.
%
% JADE with Archive
% 
% Author : Algo
% Email : zmdsn@126.com
% Date : 9/28/2016
%**************************************************************************

    %% ****************==- Initialization settings -==***********************

    mytime = cputime;
    popsize=para.pop_size;
    objfunc.Xmin=para.Xmin;objfunc.Xmax=para.Xmax;
    %% ****************==- main body  -==***********************
    n = para.dim;         
    maxFES=para.maxfe;
    
    lu = [objfunc.Xmin*ones(1,n); objfunc.Xmax*ones(1,n)];

    outcome = [];

    % Initialize the main population
   rand('seed', sum(100 * clock));
    popold = repmat(lu(1, :), popsize, 1) + rand(popsize, n) .* (repmat(lu(2, :) - lu(1, :), popsize, 1));
    for i=1:popsize

        valParents(i,:) = cec17_func( popold(i,:)' , func_name); 
    end

    c = 1/10;
    p = 0.05;

        CRm = 0.5;

        Omegam = 0.5;
        Gammam = 0.5;
        CRsigma = 0.1;

        Omegasigma = 0.1;
        Gammasigma = 0.1;
        Afactor = 1;

        archive.NP = Afactor * popsize; % the maximum size of the archive
        archive.pop = zeros(0, n); % the solutions stored in te archive
        archive.funvalues = zeros(0, 1); % the function value of the archived solutions

        %% the values and indices of the best solutions
        [valBest, indBest] = sort(valParents, 'ascend');
        

        SOmega = [];
        SGamma = [];
        SCR = [];        
        iter = 0;
        FES = 0;
        while FES < maxFES %& min(fit)>error_value(problem)
            iter =iter +1;
            pop = popold; % the old population becomes the current population
            if FES>1 && sum(SOmega)>0 && sum(SGamma)>0 && ~isempty(SCR)
                CRm = (1-c)*CRm + c*mean(SCR);

                  Omegam = (1-c)*Omegam + c*RL(SOmega,net_w,hidden_num);
                  Gammam = (1-c)*Gammam + c*RL(SGamma,net_w,hidden_num);
            end
            [Omegai,Gammai,CRi] = randFCR(popsize, CRm, CRsigma, Omegam,  Omegasigma, Gammam, Gammasigma );

            r0 = [1 : popsize];
            popAll = [pop; archive.pop];
            [r1, r2, r3] = gnR1R2(popsize,popsize ,size(popAll, 1), r0);

            % Find the p-best solutions
            pNP = max(round(p * popsize), 2); % choose at least two best solutions
            randindex = ceil(rand(1, popsize) * pNP); % select from [1, 2, 3, ..., pNP]
            randindex = max(1, randindex); % to avoid the problem that rand = 0 and thus ceil(rand) = 0
            pbest = pop(indBest(randindex), :);  % randomly choose one of the top 100p% solutions
            % == == == == == == == == == == == == == == == Mutation == == == == == == == == == == == == == 

            W=max((1-Omegai').*Gammai',1-Gammai');
            W=max(W,Omegai'.*Gammai'); W=W';
            vi = (Omegai(:, ones(1, n)).*pop+(1-Omegai(:, ones(1, n))).*pbest).*Gammai(:, ones(1, n))+(1-Gammai(:, ones(1, n))).*pop(r1,:)+W(:, ones(1, n)).*(pop(r2, :) - popAll(r3, :));
            vi = boundConstraint(vi, pop, lu);
            % == == == == = Crossover == == == == =

            mask = rand(popsize, n) > CRi(:, ones(1, n)); % mask is used to indicate which elements of ui comes from the parent
            rows = (1 : popsize)'; cols = floor(rand(popsize, 1) * n)+1; % choose one position where the element of ui doesn't come from the parent
            jrand = sub2ind([popsize n], rows, cols);
            mask(jrand) = false;
            ui = vi;
            ui(mask) = pop(mask);
            for i=1:popsize

                    valOffspring(i,:)=cec17_func( ui(i,:)',func_name);
            end

            FES = FES + popsize;
            
            % == == == == == == == == == == == == == == == Selection == == == == == == == == == == == == ==
            % I == 1: the parent is better; I == 2: the offspring is better
            [valParents, I] = min([valParents, valOffspring], [], 2);
            popold = pop;
            archive = updateArchive(archive, popold(I == 2, :), valParents(I == 2));
            popold(I == 2, :) = ui(I == 2, :);

            SOmega = Omegai(I == 2);
            SGamma = Gammai(I == 2);
            SCR = CRi(I == 2);
            [valBest indBest] = sort(valParents, 'ascend');
            outcome = [outcome ;min(valParents)];
                his_omega(iter) = Omegam;
                his_gamma(iter) = Gammam;
        end

        

    %% ****************==- collating the results -==*********************
    best_fvalue = min(valParents);
    best_solution = pop(indBest(1), :)';
    run_series = outcome;
    Altime = cputime - mytime;                  % ¼ÆËãÊ±¼ä
    run_info = [Altime];    
end


function [Omega,Gamma,CR] = randFCR(NP,CRm, CRsigma, Omegam,  Omegasigma, Gammam, Gammasigma)

% this function generate CR according to a normal distribution with mean "CRm" and sigma "CRsigma"
%           If CR > 1, set CR = 1. If CR < 0, set CR = 0.
% this function generate F  according to a cauchy distribution with location parameter "Fm" and scale parameter "Fsigma"
%           If F > 1, set F = 1. If F <= 0, regenrate F.
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang (jingqiao@gmail.com)
% Revised by Haotian Zhang (zht570795275@stu.xjtu.edu.cn) 12/1/2022
%  replace F with omega and gamma
    %% generate CR
    CR = CRm + CRsigma * randn(NP, 1);
    CR = min(1, max(0, CR));                % truncated to [0 1]

    %% generate Omega
    Omega = randCauchy(NP, 1, Omegam, Omegasigma);
    Omega = min(1, Omega);                          % truncation

    % we don't want Omega = 0. So, if Omega<=0, we regenerate Omega (instead of trucating it to 0)
    pos = find(Omega <= 0);
    while ~ isempty(pos)
        Omega(pos) = randCauchy(length(pos), 1 , Omegam, Omegasigma);
        Omega = min(1, Omega);                      % truncation
        pos = find(Omega <= 0);
    end
    
   %% generate Gamma
    Gamma = randCauchy(NP, 1, Gammam, Gammasigma);
    Gamma = min(1, Gamma);                          % truncation

    % we don't want Gamma = 0. So, if Gamma<=0, we regenerate Omega (instead of trucating it to 0)
    pos = find(Gamma <= 0);
    while ~ isempty(pos)
        Gamma(pos) = randCauchy(length(pos), 1 , Gammam, Gammasigma);
        Gamma = min(1, Gamma);                      % truncation
        pos = find(Gamma <= 0);
    end

end

% Cauchy distribution: cauchypdf = @(x, mu, delta) 1/pi*delta./((x-mu).^2+delta^2)
function result = randCauchy(m, n, mu, delta)

% http://en.wikipedia.org/wiki/Cauchy_distribution
% size(mu)
% size(delta * tan(pi * (rand(m, n) - 0.5)))
result = mu + delta * tan(pi * (rand(m, n) - 0.5));
end

function archive = updateArchive(archive, pop, funvalue)
% Update the archive with input solutions
%   Step 1: Add new solution to the archive
%   Step 2: Remove duplicate elements
%   Step 3: If necessary, randomly remove some solutions to maintain the archive size
%
% Version: 1.1   Date: 2008/04/02
% Written by Jingqiao Zhang (jingqiao@gmail.com)

    if archive.NP == 0, return; end

    if size(pop, 1) ~= size(funvalue,1), error('check it'); end

    % Method 2: Remove duplicate elements
    popAll = [archive.pop; pop ];
    funvalues = [archive.funvalues; funvalue ];
    [dummy IX]= unique(popAll, 'rows');
    if length(IX) < size(popAll, 1) % There exist some duplicate solutions
      popAll = popAll(IX, :);
      funvalues = funvalues(IX, :);
    end

    if size(popAll, 1) <= archive.NP   % add all new individuals
      archive.pop = popAll;
      archive.funvalues = funvalues;
    else                % randomly remove some solutions
      rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
      rndpos = rndpos(1 : archive.NP);

      archive.pop = popAll  (rndpos, :);
      archive.funvalues = funvalues(rndpos, :);
    end
end

function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

    [NP, D] = size(pop);  % the population size and the problem's dimension

    %% check the lower bound
    xl = repmat(lu(1, :), NP, 1);
    pos = vi < xl;
    vi(pos) = (pop(pos) + xl(pos)) / 2;

    %% check the upper bound
    xu = repmat(lu(2, :), NP, 1);
    pos = vi > xu;
    vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function [r1, r2, r3] = gnR1R2(NP1, NP2,NP3, r0)
 warning('off');
% gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
%    r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
%    r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
%
% Call:
%    [r1 r2 ...] = gnA1A2(NP1)   % r0 is set to be (1:NP1)'
%    [r1 r2 ...] = gnA1A2(NP1, r0) % r0 should be of length NP1
%
% Version: 2.1  Date: 2008/07/01
% Written by Jingqiao Zhang (jingqiao@gmail.com)
% Revised by Haotian Zhang (zht570795275@stu.xjtu.edu.cn) 12/1/2022
% Generate 3 random numners
    NP0 = length(r0);

    r1 = floor(rand(1, NP0) * NP1) + 1;
    for i = 1 : inf
        pos = (r1 == r0);
        if sum(pos) == 0
            break;
        else % regenerate r1 if it is equal to r0
            r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
        end
        if i > 1000, % this has never happened so far
            error('Can not genrate r1 in 1000 iterations');
        end
    end
    
    r2 = floor(rand(1, NP0) * NP2) + 1;
    for i = 1 : inf
        pos = ((r2 == r1) | (r2 == r0));
        if sum(pos) == 0
            break;
        else % regenerate r1 if it is equal to r0
            r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
        end
        if i > 1000, % this has never happened so far
            error('Can not genrate r1 in 1000 iterations');
        end
    end
    
    r3 = floor(rand(1, NP0) * NP3) + 1;
    for i = 1 : inf
        pos = ((r3 == r2) | (r3 == r1) |(r3 == r0));
        if sum(pos)==0
            break;
        else % regenerate r2 if it is equal to r0 or r1
            r3(pos) = floor(rand(1, sum(pos)) * NP3) + 1;
        end
        if i > 1000, % this has never happened so far
            error('Can not genrate r2 in 1000 iterations');
        end
    end
end

% mean_L = \frac{\sum_{F \in S_F}F^2}{\sum_{F \in S_F}F}
function result = meanL(F)
    result = F'*F/sum(F);
end

function [act] = RL(F,net_w,hidden_num) 
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
            act=mean(output);
end