function U=crossover_UDE(X,V,CR,crossStrategy)
[NP,Dim]=size(X);
switch crossStrategy
    %crossStrategy=1:binomial crossover
    case 1
        for i=1:NP
            jRand=randi([1,Dim]);%jRand¡Ê[1,Dim]
            for j=1:Dim
                k=rand;
                if k<=CR||j==jRand 
                    U(i,j)=V(i,j);
                else
                    U(i,j)=X(i,j);
                end     
            end    
        end

    otherwise
        error('err');
end


