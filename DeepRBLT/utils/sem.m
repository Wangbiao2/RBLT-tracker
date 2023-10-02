function SEM = sem(response)

    Fmax = max(response(:));
    F_thre = 0.3*Fmax;
    [m,n] = size(response);
    mn = m*n;
    a = length(find(response>=F_thre));
    
    SEM = a/mn;
end