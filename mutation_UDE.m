
function V=mutation_UDE(X,bestX,mutationStrategy, omega,gamma)
% NP=length(X);
[NP,Dim]=size(X);
for i=1:NP
    % Generating nrandI random numbers different from i in [1,NP]
    nrandI=5;
    r=randi([1,NP],1,nrandI);
    for j=1:nrandI
        equalr(j)=sum(r==r(j));
    end
    equali=sum(r==i);
    equalval=sum(equalr)+equali;
    while(equalval>nrandI) % if the random number are the same or equal to i, generate again!
        r=randi([1,NP],1,nrandI);
        for j=1:nrandI
            equalr(j)=sum(r==r(j));
        end
        equali=sum(r==i);
        equalval=sum(equalr)+equali;
    end
    
    switch mutationStrategy

        case 10
            F = max(abs(1-gamma),abs(1-omega)*gamma);
            FF = max(F,omega*gamma);

            V(i,:)=(omega*X(i,:)+ (1-omega)*bestX)*gamma+(1-gamma)*X(r(1),:)+FF*(X(r(2),:)-X(r(3),:));
        otherwise
            error('err');
    end
   
    
end